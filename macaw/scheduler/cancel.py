"""CancellationManager — rastreia e cancela requests batch (na fila e em execucao).

M8-03: Cancelamento de Requests na Fila.
M8-04: Cancelamento de Requests em Execucao (gRPC Cancel).

Responsabilidades:
- Registrar requests cancelaveis (cancel_event + future).
- Cancelar instantaneamente na fila (<1ms): seta cancel_event, resolve future.
- Cancelar requests em execucao via gRPC Cancel RPC ao worker.
- Idempotente: cancel de request inexistente/completada e no-op.
- Lifecycle: unregister apos conclusao (sucesso, erro ou cancel).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import grpc.aio

from macaw.logging import get_logger
from macaw.proto.stt_worker_pb2 import CancelRequest
from macaw.proto.stt_worker_pb2_grpc import STTWorkerStub
from macaw.scheduler.metrics import scheduler_cancel_latency_seconds

if TYPE_CHECKING:
    from macaw._types import BatchResult

logger = get_logger("scheduler.cancel")

# Timeout para propagacao de cancel via gRPC ao worker (segundos)
_CANCEL_PROPAGATION_TIMEOUT_S = 0.1


@dataclass(slots=True)
class _CancellableRequest:
    """Tracking info de uma request cancelavel."""

    request_id: str
    cancel_event: asyncio.Event
    result_future: asyncio.Future[BatchResult] | None
    worker_address: str | None = field(default=None)


class CancellationManager:
    """Gerencia cancelamento de requests batch.

    Thread-safe via event loop unico (asyncio single-threaded).

    Fluxo:
    1. ``register()`` quando request entra na fila.
    2. ``mark_in_flight()`` quando request e despachada ao worker.
    3. ``cancel()`` a qualquer momento — seta cancel_event e resolve future.
    4. ``unregister()`` apos conclusao (sucesso, erro ou cancel).
    """

    def __init__(self) -> None:
        self._requests: dict[str, _CancellableRequest] = {}

    def register(
        self,
        request_id: str,
        cancel_event: asyncio.Event,
        result_future: asyncio.Future[BatchResult] | None,
    ) -> None:
        """Registra request como cancelavel.

        Chamado quando request entra na fila (apos submit).

        Args:
            request_id: ID unico da request.
            cancel_event: Evento de cancelamento compartilhado com ScheduledRequest.
            result_future: Future do caller (resolvido com CancelledError no cancel).
        """
        self._requests[request_id] = _CancellableRequest(
            request_id=request_id,
            cancel_event=cancel_event,
            result_future=result_future,
        )

    def mark_in_flight(self, request_id: str, worker_address: str) -> None:
        """Marca request como em execucao (despachada ao worker).

        Usado por M8-04 para propagacao de cancel via gRPC.

        Args:
            request_id: ID da request.
            worker_address: Endereco do worker (ex: ``localhost:50051``).
        """
        entry = self._requests.get(request_id)
        if entry is not None:
            entry.worker_address = worker_address

    def cancel(self, request_id: str) -> bool:
        """Cancela request (na fila ou em execucao).

        Seta cancel_event e resolve future com CancelledError.
        Para requests em execucao, apenas seta flags locais — a propagacao
        gRPC ao worker deve ser feita via ``cancel_in_flight()`` pelo caller.

        Idempotente: cancel de request inexistente/completada retorna False.

        Args:
            request_id: ID da request a cancelar.

        Returns:
            True se request foi encontrada e cancelada, False caso contrario.
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return False

        # Seta cancel_event (dispatch loop verifica antes de gRPC call)
        entry.cancel_event.set()

        # Resolve future com CancelledError
        if entry.result_future is not None and not entry.result_future.done():
            entry.result_future.cancel()

        logger.info(
            "request_cancelled",
            request_id=request_id,
            in_flight=entry.worker_address is not None,
        )

        # Remove do tracking (cancel e terminal)
        self._requests.pop(request_id, None)

        return True

    async def cancel_in_flight(
        self,
        request_id: str,
        worker_address: str,
        channel: grpc.aio.Channel | None = None,
    ) -> bool:
        """Propaga cancelamento via gRPC Cancel RPC ao worker.

        Chamado pelo Scheduler quando a request esta em execucao (in-flight).
        Best-effort: se worker nao responde em 100ms, desiste silenciosamente.

        Args:
            request_id: ID da request a cancelar.
            worker_address: Endereco gRPC do worker (ex: ``localhost:50051``).
            channel: Canal gRPC existente (reusado do pool). Se None, cria temporario.

        Returns:
            True se worker confirmou cancel (acknowledged), False caso contrario.
        """
        start = time.monotonic()
        close_channel = False

        try:
            if channel is None:
                channel = grpc.aio.insecure_channel(worker_address)
                close_channel = True

            stub = STTWorkerStub(channel)  # type: ignore[no-untyped-call]
            response = await asyncio.wait_for(
                stub.Cancel(CancelRequest(request_id=request_id)),
                timeout=_CANCEL_PROPAGATION_TIMEOUT_S,
            )

            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            # Observa metrica de cancel latency
            if scheduler_cancel_latency_seconds is not None:
                scheduler_cancel_latency_seconds.observe(elapsed)

            logger.info(
                "cancel_propagated",
                request_id=request_id,
                worker_address=worker_address,
                acknowledged=response.acknowledged,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return bool(response.acknowledged)

        except (TimeoutError, grpc.aio.AioRpcError) as exc:
            elapsed = time.monotonic() - start
            elapsed_ms = elapsed * 1000

            # Observa metrica mesmo em falha
            if scheduler_cancel_latency_seconds is not None:
                scheduler_cancel_latency_seconds.observe(elapsed)

            logger.warning(
                "cancel_propagation_failed",
                request_id=request_id,
                worker_address=worker_address,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return False

        finally:
            if close_channel and channel is not None:
                with contextlib.suppress(Exception):
                    await channel.close()

    def unregister(self, request_id: str) -> None:
        """Remove request do tracking apos conclusao.

        Chamado quando request completa (sucesso ou erro).
        No-op se request ja foi removida (por cancel).

        Args:
            request_id: ID da request a remover.
        """
        self._requests.pop(request_id, None)

    def is_cancelled(self, request_id: str) -> bool:
        """Verifica se request foi cancelada.

        An unknown request (never registered or already unregistered) is
        considered **not cancelled** — it was never tracked, so it cannot
        have been cancelled. Only requests whose cancel_event is set
        return True.

        Args:
            request_id: ID da request.

        Returns:
            True if the request was explicitly cancelled, False otherwise
            (including unknown/unregistered requests).
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return False
        return entry.cancel_event.is_set()

    def get_worker_address(self, request_id: str) -> str | None:
        """Retorna worker address de request em execucao.

        Usado por M8-04 para propagacao de cancel via gRPC.

        Args:
            request_id: ID da request.

        Returns:
            Worker address se request esta em execucao, None caso contrario.
        """
        entry = self._requests.get(request_id)
        if entry is None:
            return None
        return entry.worker_address

    @property
    def pending_count(self) -> int:
        """Total de requests registradas (na fila + em execucao)."""
        return len(self._requests)
