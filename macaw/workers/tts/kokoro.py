"""TTS backend for Kokoro.

Implements TTSBackend using Kokoro as the inference library.
Kokoro is an optional dependency -- the import is guarded.

Kokoro v0.9.4 API:
  model = kokoro.KModel(config='config.json', model='weights.pth')
  pipeline = kokoro.KPipeline(lang_code='a', model=model, device='cpu')
  for gs, ps, audio in pipeline(text, voice='path/voice.pt', speed=1.0):
      # audio is numpy float32 at 24kHz
"""

from __future__ import annotations

import asyncio
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np

from macaw._audio_constants import TTS_DEFAULT_SAMPLE_RATE
from macaw._types import TTSEngineCapabilities, VoiceInfo
from macaw.exceptions import ModelLoadError, TTSEngineError, TTSSynthesisError
from macaw.logging import get_logger
from macaw.workers.torch_utils import release_gpu_memory, resolve_device
from macaw.workers.tts.audio_utils import CHUNK_SIZE_BYTES, float32_to_pcm16_bytes
from macaw.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    import kokoro as kokoro_lib
except ImportError:
    kokoro_lib = None

logger = get_logger("worker.tts.kokoro")

# Voice prefix -> language code. Kokoro convention: first char = language.
_VOICE_LANG_MAP: dict[str, str] = {
    "a": "en",  # American English
    "b": "en",  # British English
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt",  # Portuguese
    "z": "zh",  # Chinese
}


class KokoroBackend(TTSBackend):
    """TTS backend using Kokoro v0.9.4 (KModel + KPipeline).

    Synthesizes text into 16-bit PCM audio via Kokoro. Inference is
    executed in an executor to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._pipeline: object | None = None
        self._model_path: str = ""
        self._voices_dir: str = ""
        self._default_voice: str = "af_heart"

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities(supports_streaming=True)

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if kokoro_lib is None:
            msg = "kokoro is not installed. Install with: pip install macaw-openvoice[kokoro]"
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "cpu"))
        device = resolve_device(device_str)
        lang_code = str(config.get("lang_code", "a"))
        self._default_voice = str(config.get("default_voice", "af_heart"))

        # Find config.json and weights file in model_path
        config_path = os.path.join(model_path, "config.json")
        weights_path = _find_weights_file(model_path)

        if not os.path.isfile(config_path):
            msg = f"config.json not found in {model_path}"
            raise ModelLoadError(model_path, msg)
        if weights_path is None:
            msg = f".pth file not found in {model_path}"
            raise ModelLoadError(model_path, msg)

        voices_dir = os.path.join(model_path, "voices")
        if os.path.isdir(voices_dir):
            self._voices_dir = voices_dir

        loop = asyncio.get_running_loop()
        try:
            model, pipeline = await loop.run_in_executor(
                None,
                lambda: _load_kokoro_model(
                    config_path,
                    weights_path,
                    lang_code,
                    device,
                ),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model = model
        self._pipeline = pipeline
        self._model_path = model_path
        logger.info(
            "model_loaded",
            model_path=model_path,
            device=device,
            lang_code=lang_code,
            voices_dir=self._voices_dir,
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
        """Synthesize text into audio, returning 16-bit PCM chunks.

        Streams segments as they are produced by KPipeline — each segment
        is converted to PCM and yielded immediately, so TTFB equals the
        time to produce the first segment (not total synthesis time).

        Args:
            text: Text to synthesize.
            voice: Voice identifier or "default".
            sample_rate: Output sample rate (ignored; Kokoro uses 24kHz).
            speed: Synthesis speed (0.25-4.0).

        Yields:
            16-bit PCM audio chunks.

        Raises:
            ModelLoadError: If the model is not loaded.
            TTSSynthesisError: If synthesis fails.
        """
        if self._pipeline is None:
            msg = "Model not loaded. Call load() first."
            raise ModelLoadError("unknown", msg)

        if not text.strip():
            raise TTSSynthesisError(self._model_path, "Empty text")

        if sample_rate != TTS_DEFAULT_SAMPLE_RATE:
            logger.warning(
                "sample_rate=%d ignored; engine outputs at %dHz",
                sample_rate,
                TTS_DEFAULT_SAMPLE_RATE,
            )

        voice_path = _resolve_voice_path(
            voice,
            self._voices_dir,
            self._default_voice,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        # Design: error_holder delays error detection to allow partial audio
        # streaming. This is intentional -- partial audio > no audio for
        # real-time TTS.
        error_holder: list[Exception | None] = [None]

        future = loop.run_in_executor(
            None,
            lambda: _stream_pipeline_segments(
                self._pipeline,
                text,
                voice_path,
                speed,
                queue,
                loop,
                error_holder,
            ),
        )

        has_audio = False
        while True:
            pcm_bytes = await queue.get()
            if pcm_bytes is None:
                break
            has_audio = True
            for i in range(0, len(pcm_bytes), CHUNK_SIZE_BYTES):
                yield pcm_bytes[i : i + CHUNK_SIZE_BYTES]

        await future

        if error_holder[0] is not None:
            exc = error_holder[0]
            if isinstance(exc, TTSSynthesisError | TTSEngineError):
                raise exc
            raise TTSEngineError(self._model_path, str(exc)) from exc

        if not has_audio:
            msg = "Synthesis returned empty audio"
            raise TTSEngineError(self._model_path, msg)

    async def voices(self) -> list[VoiceInfo]:
        if not self._voices_dir or not os.path.isdir(self._voices_dir):
            return [
                VoiceInfo(
                    voice_id=self._default_voice,
                    name=self._default_voice,
                    language="en",
                ),
            ]
        return _scan_voices_dir(self._voices_dir)

    async def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        release_gpu_memory()
        self._model_path = ""
        self._voices_dir = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _find_weights_file(model_path: str) -> str | None:
    """Find the .pth weights file in the model directory.

    Args:
        model_path: Path to the model directory.

    Returns:
        Full path to the .pth file, or None if not found.
    """
    if not os.path.isdir(model_path):
        return None
    for name in os.listdir(model_path):
        if name.endswith(".pth"):
            return os.path.join(model_path, name)
    return None


def _load_kokoro_model(
    config_path: str,
    weights_path: str,
    lang_code: str,
    device: str,
) -> tuple[object, object]:
    """Load the Kokoro model and create the pipeline (blocking).

    Args:
        config_path: Path to config.json.
        weights_path: Path to .pth file.
        lang_code: Language code ('a'=en, 'p'=pt, etc).
        device: Device string ("cpu", "cuda").

    Returns:
        Tuple (model, pipeline).
    """
    # Kokoro/PyTorch/spaCy emit harmless warnings during load:
    # - UserWarning about dropout in LSTM with num_layers=1
    # - FutureWarning about deprecated weight_norm
    # - DeprecationWarning about deprecated torch.jit.script (via spaCy/thinc)
    # Suppress to avoid pytest (filterwarnings=error) treating them as errors.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = kokoro_lib.KModel(config=config_path, model=weights_path)
        model = model.to(device).eval()
        pipeline = kokoro_lib.KPipeline(
            lang_code=lang_code,
            repo_id="hexgrad/Kokoro-82M",
            model=model,
            device=device,
        )
    return model, pipeline


def _resolve_voice_path(
    voice: str,
    voices_dir: str,
    default_voice: str,
) -> str:
    """Resolve voice name to the full .pt file path.

    Strategy:
    1. "default" -> default_voice name -> voices_dir/<default_voice>.pt
    2. Simple name (e.g., "af_heart") -> voices_dir/<voice>.pt
    3. Absolute path or with extension -> return as-is

    Args:
        voice: Voice name, "default", or full path.
        voices_dir: Voices directory for the model.
        default_voice: Default voice name (e.g., "af_heart").

    Returns:
        Path to the .pt voice file, or the name if no voices_dir exists.
    """
    if voice == "default":
        voice = default_voice

    # If already absolute path or has .pt extension, return as-is
    if os.path.isabs(voice) or voice.endswith(".pt"):
        return voice

    # If we have voices_dir, build full path
    if voices_dir:
        candidate = os.path.join(voices_dir, f"{voice}.pt")
        if os.path.isfile(candidate):
            return candidate

    # Fallback: return the name (KPipeline resolves internally)
    return voice


def _stream_pipeline_segments(
    pipeline: object,
    text: str,
    voice_path: str,
    speed: float,
    queue: asyncio.Queue[bytes | None],
    loop: asyncio.AbstractEventLoop,
    error_holder: list[Exception | None],
) -> None:
    """Stream KPipeline segments to an asyncio.Queue (blocking, runs in executor).

    Each pipeline segment is converted to PCM immediately and enqueued via
    call_soon_threadsafe, enabling true streaming — the async consumer yields
    audio as segments are produced instead of waiting for all synthesis to
    complete. TTFB = time-to-first-segment, not total synthesis time.

    Args:
        pipeline: KPipeline instance.
        text: Text to synthesize.
        voice_path: Path to the .pt voice file.
        speed: Synthesis speed.
        queue: asyncio.Queue to push PCM bytes into. None sentinel signals end.
        loop: Event loop for call_soon_threadsafe.
        error_holder: Mutable list to store any exception from the pipeline.
    """
    from contextlib import nullcontext
    from typing import Any

    inference_ctx: Any = nullcontext()
    try:
        import torch

        inference_ctx = torch.inference_mode()
    except ImportError:
        pass

    try:
        with inference_ctx:
            for _gs, _ps, audio in pipeline(text, voice=voice_path, speed=speed):  # type: ignore[operator]
                if audio is not None and len(audio) > 0:
                    # Kokoro v0.9.4 returns torch.Tensor, convert to numpy
                    arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                    pcm_bytes = float32_to_pcm16_bytes(arr)
                    loop.call_soon_threadsafe(queue.put_nowait, pcm_bytes)
    except Exception as exc:
        error_holder[0] = exc
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)


def _scan_voices_dir(voices_dir: str) -> list[VoiceInfo]:
    """List available voices by scanning the voices/ directory.

    Each .pt file is a voice. The filename (without extension)
    is the voice_id. The prefix determines language and gender.

    Args:
        voices_dir: Path to the voices/ directory.

    Returns:
        List of VoiceInfo sorted by voice_id.
    """
    voices: list[VoiceInfo] = []
    for name in sorted(os.listdir(voices_dir)):
        if not name.endswith(".pt"):
            continue
        voice_id = name[:-3]  # remove .pt
        language = _voice_id_to_language(voice_id)
        gender = _voice_id_to_gender(voice_id)
        voices.append(
            VoiceInfo(
                voice_id=voice_id,
                name=voice_id,
                language=language,
                gender=gender,
            ),
        )
    return voices


def _voice_id_to_language(voice_id: str) -> str:
    """Extract language from the voice_id prefix.

    Kokoro convention: first char = language.
    a=en, b=en, e=es, f=fr, h=hi, i=it, j=ja, p=pt, z=zh.
    """
    if voice_id:
        lang = _VOICE_LANG_MAP.get(voice_id[0])
        if lang:
            return lang
    return "en"


def _voice_id_to_gender(voice_id: str) -> str | None:
    """Extract gender from the voice_id prefix.

    Kokoro convention: second char = gender (f=female, m=male).
    """
    if len(voice_id) >= 2:
        if voice_id[1] == "f":
            return "female"
        if voice_id[1] == "m":
            return "male"
    return None
