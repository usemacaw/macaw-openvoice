"""TTS backend for Qwen3-TTS.

Implements TTSBackend for all Qwen3-TTS variants:
- CustomVoice: preset speakers with style control via instruct
- Base: 3-second voice cloning from reference audio
- VoiceDesign: natural language-driven voice design

qwen-tts is an optional dependency — import is guarded.

Qwen3-TTS API (qwen_tts):
  model = Qwen3TTSModel.from_pretrained(repo, device_map=..., dtype=...)
  wavs, _sr = model.generate_custom_voice(text, language, speaker, instruct=...)
  wavs, _sr = model.generate_voice_clone(text, language, ref_audio, ref_text)
  wavs, _sr = model.generate_voice_design(text, language, instruct)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import PCM_INT16_SCALE, PCM_INT32_SCALE
from macaw._types import TTSEngineCapabilities, VoiceInfo
from macaw.exceptions import ModelLoadError, TTSEngineError, TTSSynthesisError
from macaw.logging import get_logger
from macaw.server.constants import TTS_DEFAULT_SAMPLE_RATE
from macaw.workers.torch_utils import release_gpu_memory, resolve_device
from macaw.workers.tts.audio_utils import CHUNK_SIZE_BYTES, float32_to_pcm16_bytes
from macaw.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    from qwen_tts import Qwen3TTSModel as _Qwen3TTSModel
except ImportError:
    _Qwen3TTSModel = None

logger = get_logger("worker.tts.qwen3")


# Valid variant names
_VALID_VARIANTS = frozenset({"custom_voice", "base", "voice_design"})

# Fallback preset speakers for CustomVoice variant.
# Real speaker list is obtained at runtime via model.get_supported_speakers().
_CUSTOM_VOICE_SPEAKERS: list[VoiceInfo] = [
    VoiceInfo(voice_id="aiden", name="aiden", language="multi", gender="male"),
    VoiceInfo(voice_id="dylan", name="dylan", language="multi", gender="male"),
    VoiceInfo(voice_id="eric", name="eric", language="multi", gender="male"),
    VoiceInfo(voice_id="ono_anna", name="ono_anna", language="multi", gender="female"),
    VoiceInfo(voice_id="ryan", name="ryan", language="multi", gender="male"),
    VoiceInfo(voice_id="serena", name="serena", language="multi", gender="female"),
    VoiceInfo(voice_id="sohee", name="sohee", language="multi", gender="female"),
    VoiceInfo(voice_id="uncle_fu", name="uncle_fu", language="multi", gender="male"),
    VoiceInfo(voice_id="vivian", name="vivian", language="multi", gender="female"),
]


class Qwen3TTSBackend(TTSBackend):
    """TTS backend using Qwen3-TTS (LLM-based multi-codebook TTS).

    Supports three synthesis modes determined by the ``variant`` engine_config:
    - custom_voice: preset speakers with optional style instruct
    - base: 3-second voice cloning from reference audio
    - voice_design: natural language-driven voice design

    Note: This backend synthesizes the complete audio before chunking.
    TTFB equals total synthesis time. For low-latency streaming, use Kokoro.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""
        self._variant: str = "custom_voice"
        self._default_voice: str = "vivian"
        self._default_language: str = "English"
        self._sample_rate: int = TTS_DEFAULT_SAMPLE_RATE

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities(
            supports_streaming=False,
            supports_voice_cloning=self._variant == "base",
            supports_instruct=self._variant in ("custom_voice", "voice_design"),
        )

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if _Qwen3TTSModel is None:
            msg = "qwen-tts is not installed. Install with: pip install macaw-openvoice[qwen3-tts]"
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "auto"))
        dtype_str = str(config.get("dtype", "bfloat16"))
        attn_impl = str(config.get("attn_implementation", "sdpa"))
        variant = str(config.get("variant", "custom_voice"))

        if variant not in _VALID_VARIANTS:
            msg = f"Invalid variant: {variant}. Valid: {', '.join(sorted(_VALID_VARIANTS))}"
            raise ModelLoadError(model_path, msg)

        self._variant = variant
        self._default_voice = str(config.get("default_voice", "Chelsie"))
        self._default_language = str(config.get("default_language", "English"))

        device = resolve_device(device_str)

        loop = asyncio.get_running_loop()
        try:
            model, sample_rate = await loop.run_in_executor(
                None,
                lambda: _load_qwen3_model(model_path, device, dtype_str, attn_impl),
            )
        except ModelLoadError:
            raise
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model = model
        self._model_path = model_path
        self._sample_rate = sample_rate
        logger.info(
            "model_loaded",
            model_path=model_path,
            device=device,
            dtype=dtype_str,
            variant=variant,
            sample_rate=sample_rate,
        )

    # AsyncGenerator is a subtype of AsyncIterator but mypy doesn't recognize
    # yield-based overrides. See docs/ADDING_ENGINE.md.
    async def synthesize(  # type: ignore[override, misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = TTS_DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
        options: dict[str, object] | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            msg = "Model not loaded. Call load() first."
            raise ModelLoadError("unknown", msg)

        if not text.strip():
            raise TTSSynthesisError(self._model_path, "Empty text")

        if sample_rate != self._sample_rate:
            logger.warning(
                "sample_rate=%d ignored; engine outputs at %dHz",
                sample_rate,
                self._sample_rate,
            )

        # Extract options
        language = (
            str(options.get("language", self._default_language))
            if options
            else self._default_language
        )
        ref_audio = options.get("ref_audio") if options else None
        ref_text = str(options.get("ref_text", "")) if options else ""
        instruction = str(options.get("instruction", "")) if options else ""

        if voice == "default":
            voice = self._default_voice

        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(
                None,
                lambda: _synthesize_with_model(
                    model=self._model,
                    text=text,
                    language=language,
                    voice=voice,
                    variant=self._variant,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    instruction=instruction,
                ),
            )
        except (TTSSynthesisError, TTSEngineError):
            raise
        except Exception as exc:
            raise TTSEngineError(self._model_path, str(exc)) from exc

        if len(audio_data) == 0:
            msg = "Synthesis returned empty audio"
            raise TTSEngineError(self._model_path, msg)

        for i in range(0, len(audio_data), CHUNK_SIZE_BYTES):
            yield audio_data[i : i + CHUNK_SIZE_BYTES]

    async def voices(self) -> list[VoiceInfo]:
        if self._variant == "custom_voice":
            # Try runtime speaker list if model loaded
            if self._model is not None and hasattr(self._model, "get_supported_speakers"):
                try:
                    speakers = self._model.get_supported_speakers()
                    return [VoiceInfo(voice_id=s, name=s, language="multi") for s in speakers]
                except Exception as exc:
                    logger.warning(
                        "get_supported_speakers_failed",
                        model_path=self._model_path,
                        error=str(exc),
                    )
            return list(_CUSTOM_VOICE_SPEAKERS)
        return []

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            release_gpu_memory()
        self._model_path = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _get_torch_dtype(dtype_str: str) -> object:
    """Convert dtype string to torch dtype object."""
    import torch

    dtype_map: dict[str, object] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def _load_qwen3_model(
    model_path: str,
    device: str,
    dtype_str: str,
    attn_impl: str,
) -> tuple[object, int]:
    """Load Qwen3-TTS model (blocking, runs in executor).

    Returns:
        Tuple (model, sample_rate).
    """
    dtype = _get_torch_dtype(dtype_str)
    model = _Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    # Determine sample rate from a dummy generation or default
    sample_rate = TTS_DEFAULT_SAMPLE_RATE
    return model, sample_rate


def _decode_ref_audio(ref_audio: bytes | bytearray | memoryview) -> tuple[np.ndarray, int]:
    """Decode WAV reference audio bytes to (waveform, sample_rate).

    Qwen3-TTS expects ref_audio as (np.ndarray, int) or str (path/base64).
    Our API sends WAV bytes from the gRPC proto. This decodes them.
    """
    import io
    import wave

    raw = bytes(ref_audio)

    buf = io.BytesIO(raw)
    with wave.open(buf, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        pcm_bytes = wf.readframes(n_frames)
        width = wf.getsampwidth()

    if width == 2:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / PCM_INT16_SCALE
    elif width == 4:
        samples = np.frombuffer(pcm_bytes, dtype=np.int32).astype(np.float32) / PCM_INT32_SCALE
    else:
        from macaw.exceptions import AudioFormatError

        msg = f"Unsupported WAV sample width: {width} bytes (expected 2 or 4)"
        raise AudioFormatError(msg)

    return samples, sr


def _synthesize_with_model(
    *,
    model: object,
    text: str,
    language: str,
    voice: str,
    variant: str,
    ref_audio: object | None,
    ref_text: str,
    instruction: str,
) -> bytes:
    """Synthesize audio using Qwen3-TTS model (blocking).

    Dispatches to the appropriate generate method based on variant.
    Runs under torch.inference_mode() to eliminate autograd overhead.

    Returns:
        16-bit PCM audio as bytes.
    """
    from contextlib import nullcontext
    from typing import Any

    inference_ctx: Any = nullcontext()
    try:
        import torch

        inference_ctx = torch.inference_mode()
    except ImportError:
        pass

    with inference_ctx:
        if variant == "custom_voice":
            wavs, _sr = model.generate_custom_voice(  # type: ignore[attr-defined]
                text=text,
                language=language,
                speaker=voice,
                instruct=instruction if instruction else None,
            )
        elif variant == "base":
            if ref_audio is None:
                msg = "Voice cloning requires ref_audio"
                raise TTSSynthesisError("qwen3-tts", msg)
            # Decode WAV bytes to (np.ndarray, sample_rate) tuple
            ref_audio_decoded: object
            if isinstance(ref_audio, bytes):
                ref_audio_decoded = _decode_ref_audio(ref_audio)
            else:
                ref_audio_decoded = ref_audio
            wavs, _sr = model.generate_voice_clone(  # type: ignore[attr-defined]
                text=text,
                language=language,
                ref_audio=ref_audio_decoded,
                ref_text=ref_text,
            )
        elif variant == "voice_design":
            if not instruction:
                msg = "Voice design requires instruction"
                raise TTSSynthesisError("qwen3-tts", msg)
            wavs, _sr = model.generate_voice_design(  # type: ignore[attr-defined]
                text=text,
                language=language,
                instruct=instruction,
            )
        else:
            msg = f"Unknown variant: {variant}"
            raise TTSSynthesisError("qwen3-tts", msg)

    # wavs may be a list (batch) or a single numpy array
    if isinstance(wavs, list):
        audio_array = wavs[0] if len(wavs) > 0 else np.array([], dtype=np.float32)
    else:
        audio_array = wavs

    # Convert to numpy if torch tensor
    if hasattr(audio_array, "numpy"):
        audio_array = (
            audio_array.cpu().numpy() if hasattr(audio_array, "cpu") else audio_array.numpy()
        )

    audio_array = np.asarray(audio_array, dtype=np.float32)
    return float32_to_pcm16_bytes(audio_array)
