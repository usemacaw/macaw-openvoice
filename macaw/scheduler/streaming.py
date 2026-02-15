"""gRPC streaming client para comunicacao runtime -> worker STT.

Gerencia streams bidirecionais gRPC para transcricao em tempo real.
Cada sessao abre um stream com o worker, envia AudioFrames e recebe
TranscriptEvents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
import grpc.aio

from macaw._types import TranscriptSegment, WordTimestamp
from macaw.exceptions import WorkerCrashError, WorkerTimeoutError
from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2 import AudioFrame, TranscriptEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub

logger = get_logger("scheduler.streaming")

# gRPC channel options para streaming — keepalive agressivo para detectar
# worker crash via stream break em <100ms (REGRA 1: P99, nao media)
_GRPC_STREAMING_CHANNEL_OPTIONS = [
    ("grpc.max_send_message_length", 10 * 1024 * 1024),
    ("grpc.max_receive_message_length", 10 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.min_recv_ping_interval_without_data_ms", 5_000),
    # Allow unlimited keepalive pings without data to prevent silent
    # connection death during mute-on-speak (no frames sent while TTS active).
    ("grpc.http2.max_pings_without_data", 0),
]


class StreamHandle:
    """Handle para um stream gRPC bidirecional com o worker.

    Gerencia o envio de AudioFrames e recepcao de TranscriptEvents
    para uma sessao de streaming.

    O lifecycle tipico e:
        1. open_stream() cria o StreamHandle
        2. send_frame() envia audio ao worker
        3. receive_events() consome transcript events
        4. close() encerra gracefully (is_last=True + done_writing)

    Se o worker crashar, send_frame() e receive_events() levantam
    WorkerCrashError. O runtime detecta via stream break e inicia recovery.
    """

    def __init__(
        self,
        session_id: str,
        call: grpc.aio.StreamStreamCall,  # type: ignore[type-arg]
    ) -> None:
        self._session_id = session_id
        self._call = call
        self._closed = False

    @property
    def session_id(self) -> str:
        """ID da sessao associada a este stream."""
        return self._session_id

    @property
    def is_closed(self) -> bool:
        """True se o stream foi fechado (graceful ou por erro)."""
        return self._closed

    async def send_frame(
        self,
        pcm_data: bytes,
        initial_prompt: str | None = None,
        hot_words: list[str] | None = None,
    ) -> None:
        """Envia um frame de audio ao worker.

        Args:
            pcm_data: Bytes PCM 16-bit 16kHz mono.
            initial_prompt: Contexto para conditioning (tipicamente no primeiro frame).
            hot_words: Palavras para keyword boosting.

        Raises:
            WorkerCrashError: Se o stream foi fechado pelo worker.
        """
        if self._closed:
            raise WorkerCrashError(self._session_id)

        frame = AudioFrame(
            session_id=self._session_id,
            data=pcm_data,
            is_last=False,
            initial_prompt=initial_prompt or "",
            hot_words=hot_words or [],
        )
        try:
            await self._call.write(frame)
        except grpc.aio.AioRpcError as e:
            self._closed = True
            logger.error(
                "stream_write_error",
                session_id=self._session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(self._session_id) from e

    async def receive_events(self) -> AsyncIterator[TranscriptSegment]:
        """Recebe TranscriptEvents do worker como TranscriptSegments.

        Yields:
            TranscriptSegment convertido do proto TranscriptEvent.

        Raises:
            WorkerCrashError: Se o stream quebrou inesperadamente.
            WorkerTimeoutError: Se timeout na recepcao de eventos.
        """
        try:
            async for event in self._call:
                yield _proto_event_to_transcript_segment(event)
        except grpc.aio.AioRpcError as e:
            self._closed = True
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                raise WorkerTimeoutError(self._session_id, 0.0) from e
            logger.error(
                "stream_receive_error",
                session_id=self._session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(self._session_id) from e

    async def close(self) -> None:
        """Fecha o stream gracefully, enviando is_last=True.

        Envia um AudioFrame vazio com is_last=True para sinalizar ao worker
        que nao havera mais frames, e chama done_writing() para fechar o
        lado de escrita do stream.

        Idempotente — chamadas em stream ja fechado sao no-op.
        """
        if self._closed:
            return

        try:
            frame = AudioFrame(
                session_id=self._session_id,
                data=b"",
                is_last=True,
            )
            await self._call.write(frame)
            await self._call.done_writing()
        except grpc.aio.AioRpcError:
            logger.debug(
                "stream_close_write_error",
                session_id=self._session_id,
            )
        finally:
            self._closed = True

    async def cancel(self) -> None:
        """Cancela o stream imediatamente.

        Nao espera flush de dados pendentes. Usado para cancelamento
        rapido (target: <=50ms conforme PRD).
        """
        self._closed = True
        self._call.cancel()


class StreamingGRPCClient:
    """Cliente gRPC para streaming bidirecional com workers STT.

    Gerencia o canal gRPC e a abertura de streams. Um canal e reutilizado
    para multiplos streams (uma sessao = um stream).

    Lifecycle tipico:
        client = StreamingGRPCClient("localhost:50051")
        await client.connect()
        handle = await client.open_stream("sess_123")
        # ... usar handle ...
        await client.close()
    """

    def __init__(self, worker_address: str) -> None:
        """Inicializa o cliente.

        Args:
            worker_address: Endereco do worker gRPC (ex: "localhost:50051").
        """
        self._worker_address = worker_address
        self._channel: grpc.aio.Channel | None = None
        self._stub: STTWorkerStub | None = None

    async def connect(self) -> None:
        """Abre canal gRPC com o worker.

        O canal e reutilizado para todos os streams abertos via open_stream().
        """
        self._channel = grpc.aio.insecure_channel(
            self._worker_address,
            options=_GRPC_STREAMING_CHANNEL_OPTIONS,
        )
        from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub

        self._stub = STTWorkerStub(self._channel)  # type: ignore[no-untyped-call]
        logger.info("grpc_streaming_connected", worker_address=self._worker_address)

    async def open_stream(self, session_id: str) -> StreamHandle:
        """Abre um stream bidirecional para uma sessao.

        Args:
            session_id: ID da sessao de streaming.

        Returns:
            StreamHandle para enviar/receber mensagens.

        Raises:
            WorkerCrashError: Se o canal nao esta conectado ou falhou ao abrir stream.
        """
        if self._stub is None:
            raise WorkerCrashError(session_id)

        try:
            call = self._stub.TranscribeStream()
            return StreamHandle(session_id=session_id, call=call)
        except grpc.aio.AioRpcError as e:
            logger.error(
                "stream_open_error",
                session_id=session_id,
                grpc_code=e.code().name if e.code() else "UNKNOWN",
            )
            raise WorkerCrashError(session_id) from e

    async def close(self) -> None:
        """Fecha o canal gRPC.

        Todos os streams abertos via este canal serao invalidados.
        """
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info(
                "grpc_streaming_disconnected",
                worker_address=self._worker_address,
            )


def _proto_event_to_transcript_segment(event: TranscriptEvent) -> TranscriptSegment:
    """Converte TranscriptEvent proto em TranscriptSegment Macaw.

    Funcao pura — sem side effects, sem IO.

    Args:
        event: TranscriptEvent recebido do worker via gRPC.

    Returns:
        TranscriptSegment imutavel com dados convertidos.
    """
    words: tuple[WordTimestamp, ...] | None = None
    if event.words:
        words = tuple(
            WordTimestamp(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability if w.probability != 0.0 else None,
            )
            for w in event.words
        )

    return TranscriptSegment(
        text=event.text,
        is_final=(event.event_type == "final"),
        segment_id=event.segment_id,
        start_ms=event.start_ms,
        end_ms=event.end_ms,
        language=event.language if event.language else None,
        confidence=event.confidence if event.confidence != 0.0 else None,
        words=words,
    )
