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

from macaw._types import VoiceInfo
from macaw.exceptions import ModelLoadError, TTSSynthesisError
from macaw.logging import get_logger
from macaw.workers.tts.interface import TTSBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

try:
    import kokoro as kokoro_lib
except ImportError:
    kokoro_lib = None

logger = get_logger("worker.tts.kokoro")

_DEFAULT_SAMPLE_RATE = 24000

# Size of audio chunks returned by synthesize (bytes).
# 4096 bytes = 2048 PCM 16-bit samples = ~85ms at 24kHz.
_CHUNK_SIZE_BYTES = 4096

# Voice prefix mapping -> (lang_code, language_name)
_VOICE_LANG_MAP: dict[str, tuple[str, str]] = {
    "a": ("a", "en"),  # American English
    "b": ("b", "en"),  # British English
    "e": ("e", "es"),  # Spanish
    "f": ("f", "fr"),  # French
    "h": ("h", "hi"),  # Hindi
    "i": ("i", "it"),  # Italian
    "j": ("j", "ja"),  # Japanese
    "p": ("p", "pt"),  # Portuguese
    "z": ("z", "zh"),  # Chinese
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

    async def load(self, model_path: str, config: dict[str, object]) -> None:
        if kokoro_lib is None:
            msg = "kokoro is not installed. Install with: pip install macaw-openvoice[kokoro]"
            raise ModelLoadError(model_path, msg)

        device_str = str(config.get("device", "cpu"))
        device = _resolve_device(device_str)
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

    async def synthesize(  # type: ignore[override, misc]
        self,
        text: str,
        voice: str = "default",
        *,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        """Synthesize text into audio, returning 16-bit PCM chunks.

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

        voice_path = _resolve_voice_path(
            voice,
            self._voices_dir,
            self._default_voice,
        )

        loop = asyncio.get_running_loop()
        try:
            audio_data = await loop.run_in_executor(
                None,
                lambda: _synthesize_with_pipeline(
                    self._pipeline,
                    text,
                    voice_path,
                    speed,
                ),
            )
        except TTSSynthesisError:
            raise
        except Exception as exc:
            raise TTSSynthesisError(self._model_path, str(exc)) from exc

        if len(audio_data) == 0:
            msg = "Synthesis returned empty audio"
            raise TTSSynthesisError(self._model_path, msg)

        # Yield in chunks for streaming
        for i in range(0, len(audio_data), _CHUNK_SIZE_BYTES):
            yield audio_data[i : i + _CHUNK_SIZE_BYTES]

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
        self._model = None
        self._pipeline = None
        self._model_path = ""
        self._voices_dir = ""
        logger.info("model_unloaded")

    async def health(self) -> dict[str, str]:
        if self._model is not None:
            return {"status": "ok"}
        return {"status": "not_loaded"}


# --- Pure helper functions ---


def _resolve_device(device_str: str) -> str:
    """Resolve device string to Kokoro format.

    Args:
        device_str: "auto", "cpu", "cuda", or "cuda:0".

    Returns:
        Device string in the format expected by Kokoro.
    """
    if device_str == "auto":
        return "cpu"
    return device_str


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


def _synthesize_with_pipeline(
    pipeline: object,
    text: str,
    voice_path: str,
    speed: float,
) -> bytes:
    """Synthesize text using Kokoro KPipeline (blocking).

    KPipeline returns a generator of tuples (graphemes, phonemes, audio).
    We concatenate all audio arrays and convert to 16-bit PCM.

    Args:
        pipeline: KPipeline instance.
        text: Text to synthesize.
        voice_path: Path to the .pt voice file.
        speed: Synthesis speed.

    Returns:
        16-bit PCM audio as bytes.

    Raises:
        TTSSynthesisError: If no audio is produced.
    """
    audio_arrays: list[np.ndarray] = []

    for _gs, _ps, audio in pipeline(text, voice=voice_path, speed=speed):  # type: ignore[operator]
        if audio is not None and len(audio) > 0:
            # Kokoro v0.9.4 returns torch.Tensor, convert to numpy
            arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
            audio_arrays.append(arr)

    if not audio_arrays:
        msg = "Synthesis returned empty audio"
        raise TTSSynthesisError("kokoro", msg)

    combined = np.concatenate(audio_arrays)
    return _float32_to_pcm16_bytes(combined)


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
        lang_info = _VOICE_LANG_MAP.get(voice_id[0])
        if lang_info:
            return lang_info[1]
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


def _float32_to_pcm16_bytes(audio_array: np.ndarray) -> bytes:
    """Convert normalized float32 array [-1, 1] to 16-bit PCM bytes.

    Args:
        audio_array: Normalized float32 audio.

    Returns:
        16-bit PCM little-endian bytes.
    """
    int16_data = (audio_array * 32768.0).clip(-32768, 32767).astype(np.int16)
    return int16_data.tobytes()
