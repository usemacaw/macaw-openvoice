"""`macaw transcribe` and `macaw translate` commands — thin HTTP clients."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from macaw._audio_constants import PCM_INT16_MAX, STT_SAMPLE_RATE
from macaw.cli.main import cli

DEFAULT_SERVER_URL = os.environ.get("MACAW_SERVER_URL", "http://localhost:8000")
_HTTP_TIMEOUT_S = float(os.environ.get("MACAW_HTTP_TIMEOUT_S", "120.0"))


def _post_audio(
    server_url: str,
    endpoint: str,
    file_path: Path,
    model: str,
    response_format: str,
    language: str | None,
    itn: bool = True,
    hot_words: str | None = None,
) -> None:
    """Send audio to the server via HTTP and print the result."""
    import httpx

    url = f"{server_url}{endpoint}"

    if not file_path.exists():
        click.echo(f"Error: file not found: {file_path}", err=True)
        sys.exit(1)

    data: dict[str, str] = {"model": model, "response_format": response_format}
    if language:
        data["language"] = language
    if not itn:
        data["itn"] = "false"
    if hot_words:
        data["hot_words"] = hot_words

    try:
        with file_path.open("rb") as f:
            response = httpx.post(
                url,
                files={"file": (file_path.name, f, "audio/wav")},
                data=data,
                timeout=_HTTP_TIMEOUT_S,
            )
    except httpx.ConnectError:
        click.echo(
            f"Error: server not available at {server_url}. Run 'macaw serve' first.",
            err=True,
        )
        sys.exit(1)

    if response.status_code != 200:
        try:
            error = response.json()
            msg = error.get("error", {}).get("message", response.text)
        except Exception:
            msg = response.text
        click.echo(f"Error ({response.status_code}): {msg}", err=True)
        sys.exit(1)

    # Output depends on format
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = response.json()
        if response_format == "json" and "text" in body:
            click.echo(body["text"])
        else:
            import json

            click.echo(json.dumps(body, indent=2, ensure_ascii=False))
    else:
        click.echo(response.text)


def _stream_microphone(
    server_url: str,
    model: str,
    language: str | None,
    hot_words: str | None,
    itn: bool,
) -> None:
    """Connect to the WebSocket and transcribe microphone audio in real time."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        click.echo(
            "Error: sounddevice is not installed. Install with: pip install macaw-openvoice[stream]",
            err=True,
        )
        sys.exit(1)

    try:
        import websockets  # noqa: F401
    except ImportError:
        click.echo(
            "Error: websockets is not installed. Install with: pip install macaw-openvoice[stream]",
            err=True,
        )
        sys.exit(1)

    import asyncio

    asyncio.run(
        _stream_microphone_async(
            server_url=server_url,
            model=model,
            language=language,
            hot_words=hot_words,
            itn=itn,
        )
    )


async def _stream_microphone_async(
    server_url: str,
    model: str,
    language: str | None,
    hot_words: str | None,
    itn: bool,
) -> None:
    """Async implementation of microphone streaming via WebSocket."""
    import asyncio
    import json
    import queue

    import sounddevice as sd
    from websockets.client import connect as ws_connect

    sample_rate = STT_SAMPLE_RATE
    frame_duration_ms = 40
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(
        indata: object,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        import numpy as np

        data = np.asarray(indata)
        pcm_bytes = (data * PCM_INT16_MAX).astype(np.int16).tobytes()
        audio_queue.put(pcm_bytes)

    # Build WebSocket URL
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/v1/realtime?model={model}"
    if language:
        ws_url += f"&language={language}"

    click.echo(f"Connecting to {ws_url} ...")
    click.echo("Press Ctrl+C to stop.\n")

    last_partial = ""

    try:
        async with ws_connect(ws_url) as ws:
            # Send session.configure if needed
            config: dict[str, object] = {}
            if hot_words:
                config["hot_words"] = hot_words.split(",")
            if not itn:
                config["enable_itn"] = False
            if config:
                config["type"] = "session.configure"
                await ws.send(json.dumps(config))

            # Wait for session.created
            msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
            event = json.loads(msg)
            if event.get("type") == "session.created":
                session_id = event.get("session_id", "?")
                click.echo(f"Session created: {session_id}")
            else:
                click.echo(f"Unexpected event: {event.get('type')}", err=True)

            stop_event = asyncio.Event()

            async def send_audio() -> None:
                """Send microphone audio frames via WebSocket."""
                while not stop_event.is_set():
                    try:
                        data = audio_queue.get_nowait()
                        await ws.send(data)
                    except queue.Empty:
                        await asyncio.sleep(0.01)

            async def receive_events() -> None:
                """Receive and display server events."""
                nonlocal last_partial
                async for raw_msg in ws:
                    if isinstance(raw_msg, bytes):
                        # TTS audio — ignore in CLI
                        continue
                    event = json.loads(raw_msg)
                    event_type = event.get("type", "")

                    if event_type == "transcript.partial":
                        text = event.get("text", "")
                        if text != last_partial:
                            # Clear line and show partial
                            click.echo(f"\r\033[K  ... {text}", nl=False)
                            last_partial = text

                    elif event_type == "transcript.final":
                        text = event.get("text", "")
                        # Clear partial line and show final
                        click.echo(f"\r\033[K> {text}")
                        last_partial = ""

                    elif event_type == "vad.speech_start":
                        pass  # Visual feedback could be added

                    elif event_type == "vad.speech_end":
                        pass

                    elif event_type == "error":
                        msg_text = event.get("message", "unknown error")
                        recoverable = event.get("recoverable", False)
                        if recoverable:
                            click.echo(f"\n[recovering] {msg_text}", err=True)
                        else:
                            click.echo(f"\n[error] {msg_text}", err=True)
                            stop_event.set()
                            return

                    elif event_type == "session.closed":
                        stop_event.set()
                        return

            # Start microphone capture
            stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=frame_size,
                callback=audio_callback,
            )
            stream.start()

            try:
                send_task = asyncio.create_task(send_audio())
                recv_task = asyncio.create_task(receive_events())

                # Wait for either to finish (error/close) or KeyboardInterrupt
                _done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

            finally:
                stream.stop()
                stream.close()

                # Send session.close
                import contextlib

                with contextlib.suppress(Exception):
                    await ws.send(json.dumps({"type": "session.close"}))

    except KeyboardInterrupt:
        click.echo("\n\nSession ended.")
    except ConnectionRefusedError:
        click.echo(
            f"Error: server not available at {server_url}. Run 'macaw serve' first.",
            err=True,
        )
        sys.exit(1)
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)

    click.echo("\nDone.")


@cli.command()
@click.argument("file", type=click.Path(exists=False), required=False, default=None)
@click.option("--model", "-m", required=True, help="STT model name.")
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["json", "verbose_json", "text", "srt", "vtt"]),
    default="json",
    show_default=True,
    help="Response format.",
)
@click.option("--language", "-l", default=None, help="ISO 639-1 language code.")
@click.option(
    "--no-itn",
    is_flag=True,
    default=False,
    help="Disable ITN (Inverse Text Normalization).",
)
@click.option(
    "--hot-words",
    default=None,
    help="Comma-separated list of hot words (e.g.: PIX,TED,Selic).",
)
@click.option(
    "--stream",
    is_flag=True,
    default=False,
    help="Real-time streaming via microphone/WebSocket.",
)
@click.option(
    "--server",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="Macaw server URL.",
)
def transcribe(
    file: str | None,
    model: str,
    response_format: str,
    language: str | None,
    no_itn: bool,
    hot_words: str | None,
    stream: bool,
    server: str,
) -> None:
    """Transcribes an audio file."""
    if stream:
        _stream_microphone(
            server_url=server,
            model=model,
            language=language,
            hot_words=hot_words,
            itn=not no_itn,
        )
        return

    if file is None:
        click.echo("Error: provide FILE or use --stream for microphone.", err=True)
        sys.exit(1)

    _post_audio(
        server_url=server,
        endpoint="/v1/audio/transcriptions",
        file_path=Path(file),
        model=model,
        response_format=response_format,
        language=language,
        itn=not no_itn,
        hot_words=hot_words,
    )


@cli.command()
@click.argument("file", type=click.Path(exists=False))
@click.option("--model", "-m", required=True, help="STT model name.")
@click.option(
    "--format",
    "response_format",
    type=click.Choice(["json", "verbose_json", "text", "srt", "vtt"]),
    default="json",
    show_default=True,
    help="Response format.",
)
@click.option(
    "--no-itn",
    is_flag=True,
    default=False,
    help="Disable ITN (Inverse Text Normalization).",
)
@click.option(
    "--hot-words",
    default=None,
    help="Comma-separated list of hot words (e.g.: PIX,TED,Selic).",
)
@click.option(
    "--server",
    default=DEFAULT_SERVER_URL,
    show_default=True,
    help="Macaw server URL.",
)
def translate(
    file: str,
    model: str,
    response_format: str,
    no_itn: bool,
    hot_words: str | None,
    server: str,
) -> None:
    """Translates an audio file to English."""
    _post_audio(
        server_url=server,
        endpoint="/v1/audio/translations",
        file_path=Path(file),
        model=model,
        response_format=response_format,
        language=None,
        itn=not no_itn,
        hot_words=hot_words,
    )
