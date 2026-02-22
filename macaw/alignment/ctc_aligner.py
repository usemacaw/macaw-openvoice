"""CTC-based forced alignment using torchaudio.

Uses wav2vec2 to produce CTC logits, then ``torchaudio.functional.forced_align``
to compute character-level alignment. Character alignments can be optionally
merged into word-level timing.

The CTC model is lazy-loaded on first use and cached for subsequent calls.
Requires ``torchaudio>=2.1`` (where ``forced_align`` was introduced).

This module runs inside the TTS worker subprocess — it reuses the torch
runtime already loaded for the TTS engine.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import PCM_INT16_SCALE
from macaw.logging import get_logger

from .interface import Aligner

if TYPE_CHECKING:
    from macaw._types import TTSAlignmentItem

logger = get_logger("alignment.ctc")

# Wav2Vec2 base model expects 16 kHz audio.
_MODEL_SAMPLE_RATE = 16_000

# Wav2Vec2 base: convolutional feature extractor stride = 320 samples.
_HOP_LENGTH = 320

# Word separator token in wav2vec2 dictionaries (pipe character).
_WORD_SEPARATOR = "|"


class CTCAligner(Aligner):
    """CTC-based forced alignment using wav2vec2 + torchaudio.

    Lazy-loads the wav2vec2 model on first ``align()`` call. The model
    and dictionary are cached for the lifetime of the aligner instance.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._model: object | None = None
        self._labels: list[str] = []
        self._label_to_index: dict[str, int] = {}
        self._device = device
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        """Lazy-load wav2vec2 CTC model on first use (thread-safe)."""
        if self._model is not None:
            return
        with self._lock:
            # Double-checked locking: another thread may have loaded
            # the model while we waited for the lock.
            if self._model is not None:
                return

            import torchaudio  # type: ignore[import-untyped]

            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self._model = bundle.get_model().to(self._device)
            self._model.eval()  # type: ignore[union-attr]
            self._labels = list(bundle.get_labels())
            self._label_to_index = {c: i for i, c in enumerate(self._labels)}

            logger.info(
                "alignment_model_loaded",
                model="wav2vec2-base-960h",
                device=self._device,
                vocab_size=len(self._labels),
            )

    async def align(
        self,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str = "en",
        granularity: str = "word",
    ) -> tuple[TTSAlignmentItem, ...]:
        """Align text to audio using CTC forced alignment.

        Runs the alignment synchronously in an executor to avoid
        blocking the async event loop.
        """
        if not audio or not text.strip():
            return ()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._align_sync(audio, text, sample_rate, language, granularity),
        )

    def _align_sync(
        self,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str,
        granularity: str,
    ) -> tuple[TTSAlignmentItem, ...]:
        """Synchronous forced alignment (runs in executor thread)."""
        import torch
        import torchaudio

        from macaw._types import TTSAlignmentItem as _Item
        from macaw.workers.torch_utils import get_inference_context

        self._ensure_model()

        # --- PCM bytes to waveform tensor ---
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / PCM_INT16_SCALE
        waveform = torch.from_numpy(pcm).unsqueeze(0).to(self._device)  # [1, T]

        # Resample to model's expected sample rate if needed.
        if sample_rate != _MODEL_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, _MODEL_SAMPLE_RATE)

        # --- Extract CTC emissions ---
        with get_inference_context():
            emissions, _ = self._model(waveform)  # type: ignore[operator, misc]

        log_probs = torch.nn.functional.log_softmax(emissions[0], dim=-1)

        # --- Encode text to token indices ---
        tokens = _text_to_tokens(text, self._label_to_index)
        if not tokens:
            logger.warning("alignment_empty_tokens", text=text[:50])
            return ()

        targets = torch.tensor(tokens, dtype=torch.int32)
        input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.int32)
        target_lengths = torch.tensor([len(tokens)], dtype=torch.int32)

        # --- Run forced alignment ---
        try:
            aligned_tokens, _scores = torchaudio.functional.forced_align(
                log_probs=log_probs.unsqueeze(0),
                targets=targets.unsqueeze(0),
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=0,
            )
        except Exception as exc:
            logger.warning("forced_align_failed", error=str(exc), text=text[:50])
            return ()

        # --- Convert frame indices to character-level alignment ---
        frame_ms = (_HOP_LENGTH / _MODEL_SAMPLE_RATE) * 1000.0
        char_items = _path_to_char_items(
            path=aligned_tokens[0].tolist(),
            tokens=tokens,
            labels=self._labels,
            frame_ms=frame_ms,
        )

        if granularity == "character":
            return tuple(_Item(text=c, start_ms=s, duration_ms=d) for c, s, d in char_items)

        # --- Merge characters into words ---
        word_items = _merge_chars_to_words(char_items)
        return tuple(_Item(text=w, start_ms=s, duration_ms=d) for w, s, d in word_items)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def _text_to_tokens(
    text: str,
    label_to_index: dict[str, int],
) -> list[int]:
    """Convert text to CTC token indices.

    Wav2Vec2 dictionaries use uppercase letters + ``|`` for word separator.
    Characters not in the dictionary are silently skipped.
    """
    normalized = text.upper()
    tokens: list[int] = []
    separator_idx = label_to_index.get(_WORD_SEPARATOR)

    for char in normalized:
        if char == " " and separator_idx is not None:
            # Avoid consecutive separators.
            if not tokens or tokens[-1] != separator_idx:
                tokens.append(separator_idx)
        elif char in label_to_index:
            tokens.append(label_to_index[char])
        # Unknown characters (punctuation, digits) are skipped.

    # Strip leading/trailing separators.
    if tokens and separator_idx is not None:
        if tokens[0] == separator_idx:
            tokens = tokens[1:]
        if tokens and tokens[-1] == separator_idx:
            tokens = tokens[:-1]

    return tokens


def _path_to_char_items(
    path: list[int],
    tokens: list[int],
    labels: list[str],
    frame_ms: float,
) -> list[tuple[str, int, int]]:
    """Extract per-character timing from forced alignment path.

    The path list contains token indices at each frame.  Blank frames
    (value 0) separate character emissions.  We scan for runs of the same
    non-blank token to determine start frame and duration.

    Returns list of (character, start_ms, duration_ms).
    """
    items: list[tuple[str, int, int]] = []
    current_token: int | None = None
    start_frame = 0
    token_cursor = 0

    for frame_idx, token_id in enumerate(path):
        if token_id == 0:
            # Blank frame — finalize current token if any.
            if current_token is not None:
                if token_cursor < len(tokens):
                    label = labels[tokens[token_cursor]]
                    start_ms = int(start_frame * frame_ms)
                    duration_ms = int((frame_idx - start_frame) * frame_ms)
                    items.append((label, start_ms, max(duration_ms, 1)))
                    token_cursor += 1
                current_token = None
        elif token_id != current_token:
            # New token — finalize previous if any.
            if current_token is not None and token_cursor < len(tokens):
                label = labels[tokens[token_cursor]]
                start_ms = int(start_frame * frame_ms)
                duration_ms = int((frame_idx - start_frame) * frame_ms)
                items.append((label, start_ms, max(duration_ms, 1)))
                token_cursor += 1
            current_token = token_id
            start_frame = frame_idx

    # Finalize last token.
    if current_token is not None and token_cursor < len(tokens):
        label = labels[tokens[token_cursor]]
        start_ms = int(start_frame * frame_ms)
        duration_ms = int((len(path) - start_frame) * frame_ms)
        items.append((label, start_ms, max(duration_ms, 1)))

    return items


def _merge_chars_to_words(
    char_items: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Merge character-level items into word-level items.

    Word boundaries are detected by the ``|`` separator token.
    Returns list of (word, start_ms, duration_ms).
    """
    if not char_items:
        return []

    words: list[tuple[str, int, int]] = []
    current_word_chars: list[str] = []
    word_start_ms = 0

    for char, start_ms, _duration_ms in char_items:
        if char == _WORD_SEPARATOR:
            # Word boundary — emit accumulated word.
            if current_word_chars:
                word_text = "".join(current_word_chars)
                word_end_ms = start_ms  # separator start = previous word end
                words.append((word_text, word_start_ms, word_end_ms - word_start_ms))
                current_word_chars = []
        else:
            if not current_word_chars:
                word_start_ms = start_ms
            current_word_chars.append(char)

    # Emit final word.
    if current_word_chars:
        word_text = "".join(current_word_chars)
        last_char = char_items[-1]
        word_end_ms = last_char[1] + last_char[2]
        words.append((word_text, word_start_ms, word_end_ms - word_start_ms))

    return words
