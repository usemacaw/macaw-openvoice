"""TTS backend for Chatterbox Turbo.

Implements TTSBackend using Chatterbox Turbo as the inference library.
chatterbox-tts is an optional dependency — the import is guarded.

Chatterbox Turbo API (chatterbox):
  model = ChatterboxTurboTTS.from_pretrained(device="cuda")
  wav = model.generate(text, audio_prompt_path=None,
      temperature=0.8, top_p=0.95, top_k=50, min_p=0.05,
      repetition_penalty=1.0, norm_loudness=True)
  # wav: torch.Tensor at model.sr (24kHz)
"""

from __future__ import annotations

import asyncio
import tempfile
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
    from chatterbox.tts import ChatterboxTurboTTS as _ChatterboxTurboTTS
except ImportError:
    _ChatterboxTurboTTS = None

logger = get_logger("worker.tts.chatterbox")

# Chatterbox default sampling parameters
_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_TOP_P = 0.95
_DEFAULT_TOP_K = 50
_DEFAULT_MIN_P = 0.05
_DEFAULT_REPETITION_PENALTY = 1.0


class ChatterboxTurboBackend(TTSBackend):
    """TTS backend using Chatterbox Turbo.

    Generates complete audio in one shot (no native streaming).
    Supports voice cloning via audio_prompt_path reference audio.
    Output is chunked into CHUNK_SIZE_BYTES pieces for the gRPC stream.
    """

    def __init__(self) -> None:
        self._model: object | None = None
        self._model_path: str = ""
        self._sample_rate: int = TTS_DEFAULT_SAMPLE_RATE

    async def capabilities(self) -> TTSEngineCapabilities:
        return TTSEngineCapabilities(
            supports_streaming=False,
            supports_voice_cloning=True,
        )

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if _ChatterboxTurboTTS is None:
            msg = (
                "chatterbox-tts is not installed. "
                "Install with: pip install macaw-openvoice[chatterbox]"
            )
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "auto"))
        device = resolve_device(device_str)

        loop = asyncio.get_running_loop()
        try:
            model = await loop.run_in_executor(
                None,
                lambda: _ChatterboxTurboTTS.from_pretrained(device=device),
            )
        except Exception as exc:
            msg = str(exc)
            raise ModelLoadError(model_path, msg) from exc

        self._model = model
        self._model_path = model_path
        self._sample_rate = int(getattr(model, "sr", TTS_DEFAULT_SAMPLE_RATE))
        logger.info(
            "model_loaded",
            model_path=model_path,
            device=device,
            sample_rate=self._sample_rate,
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

        # Extract options
        ref_audio = options.get("ref_audio") if options else None
        temperature = (
            float(str(options.get("temperature", _DEFAULT_TEMPERATURE)))
            if options
            else _DEFAULT_TEMPERATURE
        )
        top_p = float(str(options.get("top_p", _DEFAULT_TOP_P))) if options else _DEFAULT_TOP_P
        top_k = int(str(options.get("top_k", _DEFAULT_TOP_K))) if options else _DEFAULT_TOP_K
        repetition_penalty = (
            float(str(options.get("repetition_penalty", _DEFAULT_REPETITION_PENALTY)))
            if options
            else _DEFAULT_REPETITION_PENALTY
        )

        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(
                None,
                lambda: _synthesize_with_model(
                    model=self._model,
                    text=text,
                    ref_audio=ref_audio,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                ),
            )
        except (TTSSynthesisError, TTSEngineError):
            raise
        except Exception as exc:
            raise TTSEngineError(self._model_path, str(exc)) from exc

        if len(audio_data) == 0:
            msg = "Synthesis returned empty audio"
            raise TTSEngineError(self._model_path, msg)

        # Apply speed adjustment via resampling if needed
        if speed != 1.0 and speed > 0:
            audio_data = _apply_speed(audio_data, speed, self._sample_rate)

        for i in range(0, len(audio_data), CHUNK_SIZE_BYTES):
            yield audio_data[i : i + CHUNK_SIZE_BYTES]

    async def voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                voice_id="default",
                name="default",
                language="en",
                gender="neutral",
            ),
        ]

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


def _synthesize_with_model(
    *,
    model: object,
    text: str,
    ref_audio: object | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> bytes:
    """Synthesize audio using Chatterbox Turbo model (blocking).

    Runs under torch.inference_mode() to eliminate autograd overhead.

    Returns:
        16-bit PCM audio as bytes.
    """
    from macaw.workers.torch_utils import get_inference_context

    # Handle reference audio for voice cloning
    audio_prompt_path: str | None = None
    tmp_path: str | None = None

    if ref_audio is not None and isinstance(ref_audio, bytes | bytearray | memoryview):
        # Write ref_audio bytes to a temp file for Chatterbox API
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(bytes(ref_audio))
            tmp_path = tmp_file.name
        audio_prompt_path = tmp_path
    elif ref_audio is not None and isinstance(ref_audio, str):
        audio_prompt_path = ref_audio

    try:
        with get_inference_context():
            wav = model.generate(  # type: ignore[attr-defined]
                text,
                audio_prompt_path=audio_prompt_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                norm_loudness=True,
            )
    finally:
        if tmp_path is not None:
            import os

            os.unlink(tmp_path)

    # Convert torch.Tensor to numpy float32
    audio_array: np.ndarray
    if hasattr(wav, "numpy"):
        audio_array = wav.cpu().numpy() if hasattr(wav, "cpu") else wav.numpy()
    else:
        audio_array = np.asarray(wav, dtype=np.float32)

    # Squeeze batch dimension if present
    audio_array = np.asarray(audio_array, dtype=np.float32).squeeze()

    return float32_to_pcm16_bytes(audio_array)


def _apply_speed(audio_data: bytes, speed: float, sample_rate: int) -> bytes:
    """Apply speed adjustment by resampling the audio.

    Args:
        audio_data: 16-bit PCM audio bytes.
        speed: Speed factor (>1.0 = faster, <1.0 = slower).
        sample_rate: Audio sample rate.

    Returns:
        Speed-adjusted 16-bit PCM audio bytes.
    """
    int16_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_float = int16_array.astype(np.float32)

    original_len = len(audio_float)
    new_len = int(original_len / speed)
    if new_len <= 0:
        return audio_data

    # Resample via linear interpolation
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, new_len)
    resampled = np.interp(x_new, x_old, audio_float)

    return resampled.astype(np.int16).tobytes()
