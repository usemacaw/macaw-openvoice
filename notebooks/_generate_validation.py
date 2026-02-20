#!/usr/bin/env python3
"""Generate validation notebooks for Macaw OpenVoice.

Run this script to (re)generate the two validation notebooks:
  - validate_cli.ipynb   — CLI commands validation
  - validate_api.ipynb   — REST API + WebSocket validation

Usage:
    python notebooks/_generate_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def md(source: str) -> dict:
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [source],
    }


def notebook(cells: list[dict]) -> dict:
    """Wrap cells into a notebook document."""
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


# ===================================================================
# CLI Notebook
# ===================================================================


def build_cli_notebook() -> dict:
    cells = [
        md(
            "# Macaw OpenVoice — CLI Validation Notebook\n\n"
            "Validates **all CLI commands** of Macaw OpenVoice.\n"
            "Run all cells sequentially on Google Colab (GPU runtime recommended).\n\n"
            "**What this validates:**\n"
            "- Package installation (end-user experience)\n"
            "- Model management: `catalog`, `pull`, `list`, `inspect`, `remove`\n"
            "- Server management: `serve`, `ps`\n"
            "- Transcription: all output formats, language, ITN control\n"
            "- Translation: audio-to-English\n\n"
            "> **Pre-requisite:** Select a GPU runtime (Runtime > Change runtime type > T4 GPU)"
        ),
        # --- Configuration ---
        md("## 1. Configuration"),
        code(
            "# Edit these before running\n"
            "STT_MODEL = 'faster-whisper-tiny'\n"
            "TTS_MODEL = 'kokoro-v1'\n"
            "SERVER_PORT = 8000\n"
            "BASE_URL = f'http://localhost:{SERVER_PORT}'\n\n"
            "# Installation source:\n"
            "#   'pip'     — install released version from PyPI\n"
            "#   'develop' — install from git develop branch (pre-release validation)\n"
            "INSTALL_FROM = 'develop'"
        ),
        # --- Installation ---
        md(
            "## 2. Installation\n\n"
            "Installs from **PyPI** or **git develop branch** based on `INSTALL_FROM` above.\n\n"
            "> **Note (Colab):** You may see a pip resolver warning about `protobuf` conflicts "
            "with pre-installed `tensorflow`/`grpcio-status`. This is harmless — Macaw requires "
            "`protobuf>=6.31` and does not use TensorFlow. The warning can be safely ignored."
        ),
        code(
            "%%capture\n"
            "if INSTALL_FROM == 'develop':\n"
            "    !pip install 'macaw-openvoice[faster-whisper,kokoro,itn,codec] @ git+https://github.com/usemacaw/macaw-openvoice.git@develop'\n"
            "else:\n"
            "    !pip install macaw-openvoice[faster-whisper,kokoro,itn,codec]"
        ),
        code("!macaw --version"),
        # --- Model Management ---
        md("## 3. Model Management"),
        md("### 3.1 `macaw catalog`"),
        code("!macaw catalog"),
        md("### 3.2 `macaw pull`"),
        code("!macaw pull $STT_MODEL"),
        code("!macaw pull $TTS_MODEL"),
        md("### 3.3 `macaw list`"),
        code("!macaw list"),
        md("### 3.4 `macaw inspect`"),
        code("!macaw inspect $STT_MODEL"),
        code("!macaw inspect $TTS_MODEL"),
        # --- Server ---
        md("## 4. Server Management"),
        md("### 4.1 Start server in background"),
        code("import os\nos.environ['MACAW_WORKER_HEALTH_PROBE_TIMEOUT_S'] = '300'"),
        code("!nohup macaw serve --host 0.0.0.0 --port $SERVER_PORT > /tmp/macaw.log 2>&1 &"),
        code(
            "import time, httpx\n\n"
            "print('Waiting for server ...')\n"
            "for i in range(90):\n"
            "    try:\n"
            "        r = httpx.get(f'{BASE_URL}/health', timeout=5)\n"
            "        if r.json().get('status') == 'ok':\n"
            "            print(f'Server ready! (attempt {i+1})')\n"
            "            print(r.json())\n"
            "            break\n"
            "    except Exception:\n"
            "        pass\n"
            "    time.sleep(2)\n"
            "else:\n"
            "    print('Server did not start. Logs:')\n"
            "    !tail -50 /tmp/macaw.log\n"
            "    raise RuntimeError('Server not ready')"
        ),
        md("### 4.2 `macaw ps`"),
        code("!macaw ps --server $BASE_URL"),
        # --- Generate test audio via API ---
        md(
            "## 5. Generate Test Audio\n\n"
            "Uses TTS to generate speech, then uses that audio for STT testing (round-trip)."
        ),
        code(
            "import httpx\n\n"
            "TEST_TEXT = 'Hello world. This is a test of the Macaw voice system.'\n"
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': TEST_TEXT, 'voice': 'default'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200, f'TTS failed: {r.status_code} {r.text}'\n\n"
            "with open('/tmp/test_audio.wav', 'wb') as f:\n"
            "    f.write(r.content)\n"
            "print(f'Test audio saved: {len(r.content):,} bytes')"
        ),
        # --- Transcription ---
        md("## 6. Transcription Tests"),
        md("### 6.1 JSON format (default)"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --format json --server $BASE_URL"
        ),
        md("### 6.2 Text format"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --format text --server $BASE_URL"
        ),
        md("### 6.3 Verbose JSON format"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --format verbose_json --server $BASE_URL"
        ),
        md("### 6.4 SRT subtitle format"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --format srt --server $BASE_URL"
        ),
        md("### 6.5 VTT subtitle format"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --format vtt --server $BASE_URL"
        ),
        md("### 6.6 With explicit language"),
        code(
            "!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --language en --server $BASE_URL"
        ),
        md("### 6.7 With ITN disabled"),
        code("!macaw transcribe /tmp/test_audio.wav -m $STT_MODEL --no-itn --server $BASE_URL"),
        # --- Translation ---
        md("## 7. Translation"),
        code("!macaw translate /tmp/test_audio.wav -m $STT_MODEL --server $BASE_URL"),
        # --- Cleanup ---
        md("## 8. Cleanup"),
        code("!pkill -f 'macaw serve' || true\n!echo 'Server stopped.'"),
        code("!macaw remove $STT_MODEL --yes"),
        code("!macaw remove $TTS_MODEL --yes"),
        code("!macaw list"),
    ]
    return notebook(cells)


# ===================================================================
# API Notebook
# ===================================================================


def build_api_notebook() -> dict:
    cells = [
        md(
            "# Macaw OpenVoice — API Validation Notebook\n\n"
            "Validates **all REST API endpoints and WebSocket protocol** "
            "of Macaw OpenVoice.\n"
            "Run all cells sequentially on Google Colab (GPU runtime recommended).\n\n"
            "**What this validates:**\n"
            "- Health and system endpoints\n"
            "- Audio transcription (STT) — all formats, language, word timestamps\n"
            "- Audio translation\n"
            "- Speech synthesis (TTS) — WAV, PCM, effects, alignment, seed\n"
            "- Voice management — CRUD operations\n"
            "- WebSocket realtime — STT streaming, TTS full-duplex\n"
            "- Error handling — expected error responses\n\n"
            "> **Pre-requisite:** Select a GPU runtime (Runtime > Change runtime type > T4 GPU)"
        ),
        # --- Setup ---
        md("## 1. Setup"),
        code(
            "# Edit these before running\n"
            "STT_MODEL = 'faster-whisper-tiny'\n"
            "TTS_MODEL = 'kokoro-v1'\n"
            "SERVER_PORT = 8000\n"
            "BASE_URL = f'http://localhost:{SERVER_PORT}'\n\n"
            "# Installation source:\n"
            "#   'pip'     — install released version from PyPI\n"
            "#   'develop' — install from git develop branch (pre-release validation)\n"
            "INSTALL_FROM = 'develop'"
        ),
        md(
            "### Installation\n\n"
            "Installs from **PyPI** or **git develop branch** based on `INSTALL_FROM` above.\n\n"
            "> **Note (Colab):** You may see a pip resolver warning about `protobuf` conflicts "
            "with pre-installed `tensorflow`/`grpcio-status`. This is harmless — Macaw requires "
            "`protobuf>=6.31` and does not use TensorFlow. The warning can be safely ignored."
        ),
        code(
            "%%capture\n"
            "if INSTALL_FROM == 'develop':\n"
            "    !pip install 'macaw-openvoice[faster-whisper,kokoro,itn,codec] @ git+https://github.com/usemacaw/macaw-openvoice.git@develop'\n"
            "else:\n"
            "    !pip install macaw-openvoice[faster-whisper,kokoro,itn,codec]"
        ),
        code("!macaw pull $STT_MODEL\n!macaw pull $TTS_MODEL"),
        # Start server
        md("### Start server"),
        code("import os\nos.environ['MACAW_WORKER_HEALTH_PROBE_TIMEOUT_S'] = '300'"),
        code("!nohup macaw serve --host 0.0.0.0 --port $SERVER_PORT > /tmp/macaw.log 2>&1 &"),
        code(
            "import time, httpx\n\n"
            "print('Waiting for server ...')\n"
            "for i in range(90):\n"
            "    try:\n"
            "        r = httpx.get(f'{BASE_URL}/health', timeout=5)\n"
            "        if r.json().get('status') == 'ok':\n"
            "            print(f'Server ready! (attempt {i+1})')\n"
            "            print(r.json())\n"
            "            break\n"
            "    except Exception:\n"
            "        pass\n"
            "    time.sleep(2)\n"
            "else:\n"
            "    print('Server did not start. Logs:')\n"
            "    !tail -50 /tmp/macaw.log\n"
            "    raise RuntimeError('Server not ready')"
        ),
        # --- Health & System ---
        md("## 2. Health & System Endpoints"),
        md("### 2.1 `GET /health`"),
        code(
            "import httpx\n\n"
            "r = httpx.get(f'{BASE_URL}/health', timeout=10)\n"
            "data = r.json()\n"
            "print(data)\n"
            "assert r.status_code == 200\n"
            "assert data['status'] == 'ok'\n"
            "assert data['workers_ready'] > 0"
        ),
        md("### 2.2 `GET /v1/models`"),
        code(
            "r = httpx.get(f'{BASE_URL}/v1/models', timeout=10)\n"
            "data = r.json()\n"
            "print(data)\n"
            "assert data['object'] == 'list'\n"
            "model_ids = [m['id'] for m in data['data']]\n"
            "assert STT_MODEL in model_ids, f'{STT_MODEL} not loaded'\n"
            "assert TTS_MODEL in model_ids, f'{TTS_MODEL} not loaded'"
        ),
        # --- Generate test audio ---
        md(
            "## 3. Generate Test Audio\n\n"
            "Uses TTS to generate speech, then uses that audio for STT tests (round-trip)."
        ),
        code(
            "TEST_TEXT = 'Hello world. This is a test of the Macaw voice system.'\n\n"
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': TEST_TEXT, 'voice': 'default'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200, f'TTS failed: {r.status_code} {r.text}'\n"
            "assert len(r.content) > 1000\n\n"
            "TEST_AUDIO = '/tmp/test_audio.wav'\n"
            "with open(TEST_AUDIO, 'wb') as f:\n"
            "    f.write(r.content)\n"
            "print(f'Test audio: {len(r.content):,} bytes')"
        ),
        # --- Transcription ---
        md("## 4. Audio Transcription (`POST /v1/audio/transcriptions`)"),
        md("### 4.1 JSON format"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'json'},\n"
            "        timeout=120,\n"
            "    )\n"
            "data = r.json()\n"
            "print(data)\n"
            "assert r.status_code == 200\n"
            "assert len(data['text'].strip()) > 0"
        ),
        md("### 4.2 Verbose JSON format"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'verbose_json'},\n"
            "        timeout=120,\n"
            "    )\n"
            "data = r.json()\n"
            "print(f\"Language={data['language']}, Duration={data['duration']}s\")\n"
            "print(f\"Text: {data['text']}\")\n"
            "print(f\"Segments: {len(data['segments'])}\")\n"
            "assert 'segments' in data\n"
            "assert 'language' in data\n"
            "assert 'duration' in data"
        ),
        md("### 4.3 Text format"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'text'},\n"
            "        timeout=120,\n"
            "    )\n"
            "print(repr(r.text))\n"
            "assert r.status_code == 200\n"
            "assert len(r.text.strip()) > 0"
        ),
        md("### 4.4 SRT format"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'srt'},\n"
            "        timeout=120,\n"
            "    )\n"
            "print(r.text)\n"
            "assert '-->' in r.text, 'No SRT timestamps'"
        ),
        md("### 4.5 VTT format"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'vtt'},\n"
            "        timeout=120,\n"
            "    )\n"
            "print(r.text)\n"
            "assert 'WEBVTT' in r.text"
        ),
        md("### 4.6 With explicit language"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'language': 'en', 'response_format': 'verbose_json'},\n"
            "        timeout=120,\n"
            "    )\n"
            "data = r.json()\n"
            "print(f\"Language: {data['language']}, Text: {data['text']}\")\n"
            "assert data['language'] == 'en'"
        ),
        md("### 4.7 With word-level timestamps"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/transcriptions',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL, 'response_format': 'verbose_json',\n"
            "              'timestamp_granularities[]': 'word'},\n"
            "        timeout=120,\n"
            "    )\n"
            "data = r.json()\n"
            "words = data.get('words', [])\n"
            "print(f'Words: {words[:5]}')\n"
            "assert len(words) > 0, 'No word timestamps'\n"
            "assert 'start' in words[0]\n"
            "assert 'end' in words[0]"
        ),
        # --- Translation ---
        md("## 5. Audio Translation (`POST /v1/audio/translations`)"),
        code(
            "with open(TEST_AUDIO, 'rb') as f:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/translations',\n"
            "        files={'file': ('test.wav', f, 'audio/wav')},\n"
            "        data={'model': STT_MODEL},\n"
            "        timeout=120,\n"
            "    )\n"
            "data = r.json()\n"
            "print(data)\n"
            "assert r.status_code == 200\n"
            "assert len(data['text'].strip()) > 0"
        ),
        # --- Speech Synthesis ---
        md("## 6. Speech Synthesis (`POST /v1/audio/speech`)"),
        md("### 6.1 WAV format (default)"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Hello world', 'voice': 'default'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "assert r.content[:4] == b'RIFF', 'Not a WAV file'\n"
            "print(f'WAV: {len(r.content):,} bytes')"
        ),
        md("### 6.2 PCM format"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Hello world', 'response_format': 'pcm'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "assert r.content[:4] != b'RIFF', 'Should be raw PCM, not WAV'\n"
            "assert len(r.content) % 2 == 0, 'PCM 16-bit must be even bytes'\n"
            "print(f'PCM: {len(r.content):,} bytes')"
        ),
        md("### 6.3 Speed control"),
        code(
            "r_normal = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Testing speed control', 'speed': 1.0},\n"
            "    timeout=120,\n"
            ")\n"
            "r_fast = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Testing speed control', 'speed': 2.0},\n"
            "    timeout=120,\n"
            ")\n"
            "print(f'Normal: {len(r_normal.content):,} bytes')\n"
            "print(f'Fast:   {len(r_fast.content):,} bytes')\n"
            "assert len(r_fast.content) < len(r_normal.content), 'Fast should be shorter'"
        ),
        md("### 6.4 Audio effects — pitch shift"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Pitch shift test',\n"
            "          'effects': {'pitch_shift_semitones': 3.0}},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "print(f'Pitch shift: {len(r.content):,} bytes')"
        ),
        md("### 6.5 Audio effects — reverb"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Reverb test',\n"
            "          'effects': {'reverb_room_size': 0.7, 'reverb_damping': 0.5,\n"
            "                      'reverb_wet_dry_mix': 0.3}},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "print(f'Reverb: {len(r.content):,} bytes')"
        ),
        md("### 6.6 Audio effects — combined"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Combined effects',\n"
            "          'effects': {'pitch_shift_semitones': -2.0,\n"
            "                      'reverb_room_size': 0.5, 'reverb_wet_dry_mix': 0.2}},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "print(f'Combined: {len(r.content):,} bytes')"
        ),
        md("### 6.7 Word-level alignment (NDJSON)"),
        code(
            "import json as _json\n\n"
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Hello world',\n"
            "          'include_alignment': True, 'alignment_granularity': 'word'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "assert 'ndjson' in r.headers.get('content-type', '')\n\n"
            "lines = [_json.loads(l) for l in r.text.strip().split('\\n') if l.strip()]\n"
            "audio_lines = [l for l in lines if l['type'] == 'audio']\n"
            "done_lines = [l for l in lines if l['type'] == 'done']\n"
            "print(f'Audio chunks: {len(audio_lines)}, Done: {len(done_lines)}')\n\n"
            "aligned = [l for l in audio_lines if l.get('alignment')]\n"
            "if aligned:\n"
            "    print(f\"Alignment: {aligned[0]['alignment']}\")\n"
            "assert len(done_lines) == 1"
        ),
        md("### 6.8 Character-level alignment"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Hi',\n"
            "          'include_alignment': True, 'alignment_granularity': 'character'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "lines = [_json.loads(l) for l in r.text.strip().split('\\n') if l.strip()]\n"
            "aligned = [l for l in lines if l.get('type') == 'audio' and l.get('alignment')]\n"
            "if aligned:\n"
            "    a = aligned[0]['alignment']\n"
            "    print(f\"Granularity: {a.get('granularity')}, Items: {a['items']}\")\n"
            "    assert a['granularity'] == 'character'"
        ),
        md("### 6.9 Seed for reproducibility"),
        code(
            "payload = {'model': TTS_MODEL, 'input': 'Reproducibility test', 'seed': 42}\n"
            "r1 = httpx.post(f'{BASE_URL}/v1/audio/speech', json=payload, timeout=120)\n"
            "r2 = httpx.post(f'{BASE_URL}/v1/audio/speech', json=payload, timeout=120)\n"
            "assert r1.status_code == 200\n"
            "assert r2.status_code == 200\n"
            "print(f'Identical output: {r1.content == r2.content}')\n"
            "print(f'Size: {len(r1.content):,} bytes')"
        ),
        md("### 6.10 Text normalization"),
        code(
            "for mode in ['auto', 'on', 'off']:\n"
            "    r = httpx.post(\n"
            "        f'{BASE_URL}/v1/audio/speech',\n"
            "        json={'model': TTS_MODEL, 'input': 'I have 3 cats.',\n"
            "              'text_normalization': mode},\n"
            "        timeout=120,\n"
            "    )\n"
            "    assert r.status_code == 200\n"
            "    print(f'{mode}: {len(r.content):,} bytes')"
        ),
        # --- Voice Management ---
        md("## 7. Voice Management"),
        md("### 7.1 List preset voices"),
        code(
            "r = httpx.get(f'{BASE_URL}/v1/voices', timeout=30)\n"
            "data = r.json()\n"
            "assert data['object'] == 'list'\n"
            "assert len(data['data']) > 0\n"
            "print(f\"Total voices: {len(data['data'])}\")\n"
            "for v in data['data'][:5]:\n"
            "    print(f\"  {v['voice_id']}: {v['name']} ({v.get('language')})\")"
        ),
        md("### 7.2 Create designed voice"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/voices',\n"
            "    data={'name': 'Test Voice', 'voice_type': 'designed',\n"
            "          'instruction': 'A warm, friendly voice', 'language': 'en'},\n"
            "    timeout=30,\n"
            ")\n"
            "voice = r.json()\n"
            "print(voice)\n"
            "assert r.status_code == 201\n"
            "VOICE_ID = voice['voice_id']"
        ),
        md("### 7.3 Get saved voice"),
        code(
            "r = httpx.get(f'{BASE_URL}/v1/voices/{VOICE_ID}', timeout=10)\n"
            "data = r.json()\n"
            "print(data)\n"
            "assert r.status_code == 200\n"
            "assert data['voice_id'] == VOICE_ID"
        ),
        md("### 7.4 Use saved voice in synthesis"),
        code(
            "r = httpx.post(\n"
            "    f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'Saved voice test',\n"
            "          'voice': f'voice_{VOICE_ID}'},\n"
            "    timeout=120,\n"
            ")\n"
            "assert r.status_code == 200\n"
            "print(f'Audio with saved voice: {len(r.content):,} bytes')"
        ),
        md("### 7.5 Delete saved voice"),
        code(
            "r = httpx.delete(f'{BASE_URL}/v1/voices/{VOICE_ID}', timeout=10)\n"
            "assert r.status_code == 204\n\n"
            "# Verify it's gone\n"
            "r2 = httpx.get(f'{BASE_URL}/v1/voices/{VOICE_ID}', timeout=10)\n"
            "assert r2.status_code == 404\n"
            "print('Voice deleted and confirmed gone.')"
        ),
        # --- WebSocket ---
        md(
            "## 8. WebSocket Realtime (`/v1/realtime`)\n\n"
            "Tests bidirectional WebSocket protocol for STT streaming and TTS full-duplex."
        ),
        code("%%capture\n!pip install websockets"),
        md("### 8.1 STT streaming"),
        code(
            "import asyncio, json, websockets\n\n"
            "async def test_ws_stt():\n"
            "    ws_url = f'ws://localhost:{SERVER_PORT}/v1/realtime?model={STT_MODEL}'\n"
            "    async with websockets.connect(ws_url) as ws:\n"
            "        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))\n"
            "        assert msg['type'] == 'session.created'\n"
            "        print(f\"Session: {msg['session_id']}\")\n\n"
            "        # Send test audio PCM (skip 44-byte WAV header)\n"
            "        with open(TEST_AUDIO, 'rb') as f:\n"
            "            pcm = f.read()[44:]\n"
            "        for i in range(0, min(len(pcm), 48000), 3200):\n"
            "            await ws.send(pcm[i:i+3200])\n"
            "            await asyncio.sleep(0.05)\n\n"
            "        await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))\n\n"
            "        events = []\n"
            "        try:\n"
            "            while True:\n"
            "                raw = await asyncio.wait_for(ws.recv(), timeout=15)\n"
            "                if isinstance(raw, str):\n"
            "                    ev = json.loads(raw)\n"
            "                    events.append(ev)\n"
            "                    print(f\"  {ev['type']}\")\n"
            "                    if ev['type'] == 'transcript.final':\n"
            "                        print(f\"  Text: {ev['text']}\")\n"
            "                        break\n"
            "        except asyncio.TimeoutError:\n"
            "            pass\n\n"
            "        types = [e['type'] for e in events]\n"
            "        assert 'transcript.final' in types or 'transcript.partial' in types\n"
            "        await ws.send(json.dumps({'type': 'session.close'}))\n\n"
            "await test_ws_stt()"
        ),
        md("### 8.2 Session configuration"),
        code(
            "async def test_ws_configure():\n"
            "    ws_url = f'ws://localhost:{SERVER_PORT}/v1/realtime?model={STT_MODEL}'\n"
            "    async with websockets.connect(ws_url) as ws:\n"
            "        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))\n"
            "        assert msg['type'] == 'session.created'\n\n"
            "        await ws.send(json.dumps({\n"
            "            'type': 'session.configure',\n"
            "            'language': 'en',\n"
            "            'enable_partial_transcripts': True,\n"
            "            'vad_sensitivity': 'high',\n"
            "        }))\n"
            "        print('Configure sent OK')\n\n"
            "        await ws.send(json.dumps({'type': 'session.close'}))\n"
            "        try:\n"
            "            while True:\n"
            "                await asyncio.wait_for(ws.recv(), timeout=3)\n"
            "        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):\n"
            "            pass\n"
            "        print('Session closed OK')\n\n"
            "await test_ws_configure()"
        ),
        md("### 8.3 TTS via WebSocket"),
        code(
            "async def test_ws_tts():\n"
            "    ws_url = f'ws://localhost:{SERVER_PORT}/v1/realtime?model={STT_MODEL}'\n"
            "    async with websockets.connect(ws_url) as ws:\n"
            "        json.loads(await asyncio.wait_for(ws.recv(), timeout=10))\n\n"
            "        await ws.send(json.dumps({'type': 'session.configure', 'model_tts': TTS_MODEL}))\n"
            "        await ws.send(json.dumps({'type': 'tts.speak', 'text': 'Hello from WebSocket',\n"
            "                                  'request_id': 'test_tts_1'}))\n\n"
            "        events, audio_frames = [], []\n"
            "        try:\n"
            "            while True:\n"
            "                raw = await asyncio.wait_for(ws.recv(), timeout=30)\n"
            "                if isinstance(raw, bytes):\n"
            "                    audio_frames.append(raw)\n"
            "                else:\n"
            "                    ev = json.loads(raw)\n"
            "                    events.append(ev)\n"
            "                    print(f\"  {ev['type']}\")\n"
            "                    if ev['type'] == 'tts.speaking_end':\n"
            "                        break\n"
            "        except asyncio.TimeoutError:\n"
            "            pass\n\n"
            "        types = [e['type'] for e in events]\n"
            "        assert 'tts.speaking_start' in types\n"
            "        assert 'tts.speaking_end' in types\n"
            "        total = sum(len(f) for f in audio_frames)\n"
            "        print(f'Audio: {len(audio_frames)} frames, {total:,} bytes')\n"
            "        assert total > 500\n"
            "        await ws.send(json.dumps({'type': 'session.close'}))\n\n"
            "await test_ws_tts()"
        ),
        md("### 8.4 TTS with alignment"),
        code(
            "async def test_ws_alignment():\n"
            "    ws_url = f'ws://localhost:{SERVER_PORT}/v1/realtime?model={STT_MODEL}'\n"
            "    async with websockets.connect(ws_url) as ws:\n"
            "        json.loads(await asyncio.wait_for(ws.recv(), timeout=10))\n"
            "        await ws.send(json.dumps({'type': 'session.configure', 'model_tts': TTS_MODEL}))\n"
            "        await ws.send(json.dumps({'type': 'tts.speak', 'text': 'Alignment test',\n"
            "                                  'include_alignment': True, 'request_id': 'align'}))\n\n"
            "        events = []\n"
            "        try:\n"
            "            while True:\n"
            "                raw = await asyncio.wait_for(ws.recv(), timeout=30)\n"
            "                if isinstance(raw, str):\n"
            "                    ev = json.loads(raw)\n"
            "                    events.append(ev)\n"
            "                    if ev['type'] == 'tts.speaking_end':\n"
            "                        break\n"
            "        except asyncio.TimeoutError:\n"
            "            pass\n\n"
            "        types = [e['type'] for e in events]\n"
            "        assert 'tts.alignment' in types, f'No alignment event. Got: {types}'\n"
            "        ae = next(e for e in events if e['type'] == 'tts.alignment')\n"
            "        print(f\"Alignment items: {ae['items']}\")\n"
            "        await ws.send(json.dumps({'type': 'session.close'}))\n\n"
            "await test_ws_alignment()"
        ),
        md("### 8.5 TTS cancel"),
        code(
            "async def test_ws_cancel():\n"
            "    ws_url = f'ws://localhost:{SERVER_PORT}/v1/realtime?model={STT_MODEL}'\n"
            "    async with websockets.connect(ws_url) as ws:\n"
            "        json.loads(await asyncio.wait_for(ws.recv(), timeout=10))\n"
            "        await ws.send(json.dumps({'type': 'session.configure', 'model_tts': TTS_MODEL}))\n"
            "        await ws.send(json.dumps({'type': 'tts.speak', 'request_id': 'cancel_test',\n"
            "            'text': 'This long sentence should be cancelled before finishing.'}))\n\n"
            "        # Wait for speaking_start, then cancel\n"
            "        try:\n"
            "            while True:\n"
            "                raw = await asyncio.wait_for(ws.recv(), timeout=10)\n"
            "                if isinstance(raw, str):\n"
            "                    ev = json.loads(raw)\n"
            "                    if ev['type'] == 'tts.speaking_start':\n"
            "                        await ws.send(json.dumps({'type': 'tts.cancel'}))\n"
            "                        break\n"
            "        except asyncio.TimeoutError:\n"
            "            pass\n\n"
            "        # Wait for cancelled speaking_end\n"
            "        cancelled = False\n"
            "        try:\n"
            "            while True:\n"
            "                raw = await asyncio.wait_for(ws.recv(), timeout=10)\n"
            "                if isinstance(raw, str):\n"
            "                    ev = json.loads(raw)\n"
            "                    if ev['type'] == 'tts.speaking_end':\n"
            "                        cancelled = ev.get('cancelled', False)\n"
            "                        break\n"
            "        except asyncio.TimeoutError:\n"
            "            pass\n\n"
            "        print(f'Cancelled: {cancelled}')\n"
            "        assert cancelled\n"
            "        await ws.send(json.dumps({'type': 'session.close'}))\n\n"
            "await test_ws_cancel()"
        ),
        # --- Error Handling ---
        md("## 9. Error Handling"),
        code(
            "# Empty text\n"
            "r = httpx.post(f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': ''}, timeout=30)\n"
            "assert r.status_code >= 400\n"
            "print(f'Empty text: {r.status_code}')\n\n"
            "# Non-existent model\n"
            "r = httpx.post(f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': 'nonexistent', 'input': 'test'}, timeout=30)\n"
            "assert r.status_code >= 400\n"
            "print(f'Bad model: {r.status_code}')\n\n"
            "# Non-existent voice\n"
            "r = httpx.get(f'{BASE_URL}/v1/voices/nonexistent', timeout=10)\n"
            "assert r.status_code == 404\n"
            "print(f'Bad voice: {r.status_code}')\n\n"
            "# Invalid audio\n"
            "r = httpx.post(f'{BASE_URL}/v1/audio/transcriptions',\n"
            "    files={'file': ('bad.txt', b'not audio', 'text/plain')},\n"
            "    data={'model': STT_MODEL}, timeout=30)\n"
            "assert r.status_code == 400\n"
            "print(f'Bad audio: {r.status_code}')\n\n"
            "# Alignment + opus conflict\n"
            "r = httpx.post(f'{BASE_URL}/v1/audio/speech',\n"
            "    json={'model': TTS_MODEL, 'input': 'test',\n"
            "          'include_alignment': True, 'response_format': 'opus'}, timeout=30)\n"
            "assert r.status_code == 400\n"
            "print(f'Alignment+opus: {r.status_code}')"
        ),
        # --- Cleanup ---
        md("## 10. Cleanup"),
        code("!pkill -f 'macaw serve' || true\n!echo 'Server stopped.'"),
    ]
    return notebook(cells)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    out_dir = Path(__file__).parent

    cli_nb = build_cli_notebook()
    cli_path = out_dir / "validate_cli.ipynb"
    cli_path.write_text(json.dumps(cli_nb, indent=1, ensure_ascii=False))
    print(f"Generated: {cli_path} ({len(cli_nb['cells'])} cells)")

    api_nb = build_api_notebook()
    api_path = out_dir / "validate_api.ipynb"
    api_path.write_text(json.dumps(api_nb, indent=1, ensure_ascii=False))
    print(f"Generated: {api_path} ({len(api_nb['cells'])} cells)")


if __name__ == "__main__":
    main()
