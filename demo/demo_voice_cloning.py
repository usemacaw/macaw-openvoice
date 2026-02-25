"""Interactive Voice Cloning demo — Macaw OpenVoice.

Uses Qwen3-TTS-0.6B-Base to clone voice from reference audio.
Web interface via Gradio.

Usage:
    PYTHONPATH=. .venv/bin/python demo_voice_cloning.py

Opens in browser: http://localhost:7860
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import time
import wave

import numpy as np

MODEL_PATH = os.path.expanduser("~/.macaw/models/qwen3-tts-0.6b-base")
SAMPLE_RATE = 24000

LANGUAGES = [
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

# Global backend — loaded once
_backend = None
_load_time = 0.0


def _get_backend():
    """Lazy-load the backend (loads model on first call)."""
    global _backend, _load_time

    if _backend is not None:
        return _backend

    from macaw.workers.tts.qwen3 import Qwen3TTSBackend

    _backend = Qwen3TTSBackend()
    config = {
        "device": "cpu",
        "dtype": "float32",
        "variant": "base",
        "default_language": "English",
        "sample_rate": SAMPLE_RATE,
    }

    t0 = time.monotonic()
    asyncio.get_event_loop().run_until_complete(_backend.load(MODEL_PATH, config))
    _load_time = time.monotonic() - t0
    print(f"Model loaded in {_load_time:.1f}s")
    return _backend


def _audio_file_to_wav_bytes(filepath: str) -> bytes:
    """Convert any audio file to mono WAV bytes at 24kHz."""
    import soundfile as sf

    data, sr = sf.read(filepath, dtype="float32")
    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Resample to 24kHz if necessary
    if sr != SAMPLE_RATE:
        from scipy.signal import resample

        num_samples = int(len(data) * SAMPLE_RATE / sr)
        data = resample(data, num_samples).astype(np.float32)
        sr = SAMPLE_RATE

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _gradio_audio_to_wav_bytes(audio_input) -> bytes:
    """Convert Gradio input (filepath or tuple) to WAV bytes."""
    if isinstance(audio_input, str):
        return _audio_file_to_wav_bytes(audio_input)

    if isinstance(audio_input, tuple):
        sr, data = audio_input
        data = np.asarray(data, dtype=np.float32)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.max() > 1.0 or data.min() < -1.0:
            data = data / max(abs(data.max()), abs(data.min()))
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample
        if sr != SAMPLE_RATE:
            from scipy.signal import resample

            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = resample(data, num_samples).astype(np.float32)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            pcm16 = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    msg = f"Unsupported audio format: {type(audio_input)}"
    raise ValueError(msg)


async def clone_voice(
    ref_audio,
    ref_text: str,
    synth_text: str,
    language: str,
) -> tuple[str | None, str]:
    """Main function called by Gradio (async)."""
    if ref_audio is None:
        return None, "Error: Please upload or record a reference audio."

    if not synth_text.strip():
        return None, "Error: Please enter the text to synthesize."

    if not ref_text.strip():
        return None, "Error: Please enter the transcription of the reference audio."

    try:
        backend = _get_backend()
    except Exception as e:
        return None, f"Error loading model: {e}"

    # Convert reference audio
    try:
        ref_wav_bytes = _gradio_audio_to_wav_bytes(ref_audio)
    except Exception as e:
        return None, f"Error processing reference audio: {e}"

    # Synthesize
    t0 = time.monotonic()
    chunks: list[bytes] = []

    try:
        async for chunk in backend.synthesize(
            text=synth_text,
            voice="default",
            sample_rate=SAMPLE_RATE,
            options={
                "ref_audio": ref_wav_bytes,
                "ref_text": ref_text,
                "language": language,
            },
        ):
            chunks.append(chunk)
    except Exception as e:
        return None, f"Synthesis error: {e}"

    synth_time = time.monotonic() - t0
    total_pcm = b"".join(chunks)
    audio_duration = len(total_pcm) / (SAMPLE_RATE * 2)

    # Save temporary WAV
    out_path = os.path.join(tempfile.gettempdir(), "macaw_clone_output.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(total_pcm)

    status = (
        f"Synthesis complete!\n"
        f"  Time: {synth_time:.1f}s\n"
        f"  Generated audio: {audio_duration:.1f}s ({len(chunks)} chunks)\n"
        f"  RTF: {synth_time / audio_duration:.1f}x (CPU)"
    )

    return out_path, status


def build_app():
    """Build the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Macaw Voice Cloning") as app:
        gr.Markdown(
            """
            # Macaw OpenVoice — Voice Cloning Demo

            Clone any voice from ~3 seconds of reference audio.

            **Model:** Qwen3-TTS-0.6B-Base | **Engine:** qwen3-tts | **Device:** CPU

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference Audio")
                ref_audio = gr.Audio(
                    label="Record or upload audio (~3-10s)",
                    type="filepath",
                )
                ref_text = gr.Textbox(
                    label="Transcription of the reference audio",
                    placeholder="Type exactly what was said in the audio...",
                    lines=2,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 2. Text to Synthesize")
                synth_text = gr.Textbox(
                    label="Text the cloned voice will speak",
                    placeholder="Type the text you want to hear with the cloned voice...",
                    lines=3,
                )
                language = gr.Dropdown(
                    choices=LANGUAGES,
                    value="English",
                    label="Language",
                )

        gr.Markdown("---")

        with gr.Row():
            btn = gr.Button(
                "Clone Voice",
                variant="primary",
                size="lg",
            )

        with gr.Row():
            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                )
            with gr.Column(scale=1):
                status = gr.Textbox(
                    label="Status",
                    lines=4,
                    interactive=False,
                )

        btn.click(
            fn=clone_voice,
            inputs=[ref_audio, ref_text, synth_text, language],
            outputs=[output_audio, status],
        )

        gr.Markdown(
            """
            ---
            **Tips:**
            - Use clean audio of 3-10 seconds (no background noise)
            - The transcription must be exact — it helps the model understand the timbre
            - CPU inference is slow (~60-90s). With GPU, expect ~1-3s
            - Supported languages: zh, en, ja, ko, de, fr, ru, pt, es, it
            """
        )

    return app


if __name__ == "__main__":
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at: {MODEL_PATH}")
        print("Install with: macaw pull qwen3-tts-0.6b-base")
        raise SystemExit(1)

    print("Macaw Voice Cloning Demo")
    print(f"Model: {MODEL_PATH}")
    print()
    print("Loading model (first time may take ~10s)...")

    # Pre-load to avoid freezing on first request
    _get_backend()

    print()
    print("Starting web interface...")
    import gradio as gr

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
