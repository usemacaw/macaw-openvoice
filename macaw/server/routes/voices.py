"""Voices endpoints — list available TTS voices + saved voice CRUD."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import grpc.aio
from fastapi import APIRouter, Depends, Form, Request, UploadFile

from macaw._types import ModelType
from macaw.exceptions import InvalidRequestError, VoiceNotFoundError
from macaw.logging import get_logger
from macaw.proto.tts_worker_pb2 import ListVoicesRequest
from macaw.proto.tts_worker_pb2_grpc import TTSWorkerStub
from macaw.server.dependencies import get_registry, get_worker_manager, require_voice_store
from macaw.server.grpc_channels import get_or_create_tts_channel
from macaw.server.models.voices import (
    SavedVoiceResponse,
    VoiceListResponse,
    VoiceResponse,
)

if TYPE_CHECKING:
    from macaw.registry.registry import ModelRegistry
    from macaw.server.voice_store import VoiceStore
    from macaw.workers.manager import WorkerManager

router = APIRouter(tags=["Voices"])

logger = get_logger("server.routes.voices")

# Timeout for the ListVoices RPC (shorter than synthesis)
_GRPC_TIMEOUT = 10.0

# Allowed voice types
_ALLOWED_VOICE_TYPES = frozenset({"cloned", "designed"})


@router.get("/v1/voices", response_model=VoiceListResponse)
async def list_voices(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),  # noqa: B008
    worker_manager: WorkerManager = Depends(get_worker_manager),  # noqa: B008
) -> VoiceListResponse:
    """List all available TTS voices from loaded workers.

    Queries each loaded TTS model's worker via gRPC ListVoices RPC
    and aggregates the results into a single list.
    """
    all_voices: list[VoiceResponse] = []
    failed_models: list[str] = []

    # Get all TTS models from registry
    for manifest in registry.list_models():
        if manifest.model_type != ModelType.TTS:
            continue

        model_name = manifest.name

        # Check if worker is ready
        worker = worker_manager.get_ready_worker(model_name)
        if worker is None:
            logger.debug("voices_worker_not_ready", model=model_name)
            continue

        # Query voices from the worker via gRPC
        worker_address = f"localhost:{worker.port}"
        try:
            tts_channels: dict[str, grpc.aio.Channel] = request.app.state.tts_channels
            channel = get_or_create_tts_channel(tts_channels, worker_address)

            stub = TTSWorkerStub(channel)  # type: ignore[no-untyped-call]
            response = await stub.ListVoices(ListVoicesRequest(), timeout=_GRPC_TIMEOUT)

            for voice_proto in response.voices:
                all_voices.append(
                    VoiceResponse(
                        voice_id=voice_proto.voice_id,
                        name=voice_proto.name,
                        language=voice_proto.language,
                        gender=voice_proto.gender if voice_proto.gender else None,
                        model=model_name,
                    )
                )

        except grpc.aio.AioRpcError as exc:
            logger.warning(
                "voices_grpc_error",
                model=model_name,
                grpc_code=str(exc.code()),
            )
            failed_models.append(model_name)
            continue
        except Exception:
            logger.exception("voices_unexpected_error", model=model_name)
            failed_models.append(model_name)
            continue

    if failed_models:
        logger.warning(
            "voices_partial_response",
            failed_models=failed_models,
            returned_count=len(all_voices),
        )

    return VoiceListResponse(data=all_voices)


# ─── Saved Voice CRUD ───


@router.post("/v1/voices", response_model=SavedVoiceResponse, status_code=201)
async def create_voice(
    name: str = Form(),
    voice_type: str = Form(default="designed"),
    language: str | None = Form(default=None),
    ref_text: str | None = Form(default=None),
    instruction: str | None = Form(default=None),
    ref_audio: UploadFile | None = None,
    voice_store: VoiceStore = Depends(require_voice_store),  # noqa: B008
) -> SavedVoiceResponse:
    """Create a saved voice for reuse in speech requests.

    Use multipart/form-data. For cloned voices, include ``ref_audio`` file.
    For designed voices, include ``instruction`` text.

    The returned ``voice_id`` can be used as ``voice_{id}`` in POST /v1/audio/speech.
    """

    if voice_type not in _ALLOWED_VOICE_TYPES:
        allowed = ", ".join(sorted(_ALLOWED_VOICE_TYPES))
        raise InvalidRequestError(f"Invalid voice_type '{voice_type}'. Allowed: {allowed}")

    if voice_type == "cloned" and ref_audio is None:
        raise InvalidRequestError("voice_type 'cloned' requires ref_audio file upload.")

    if voice_type == "designed" and instruction is None:
        raise InvalidRequestError("voice_type 'designed' requires instruction text.")

    # Read ref_audio bytes if provided
    ref_audio_bytes: bytes | None = None
    if ref_audio is not None:
        ref_audio_bytes = await ref_audio.read()

    voice_id = str(uuid.uuid4())

    saved = await voice_store.save(
        voice_id=voice_id,
        name=name,
        voice_type=voice_type,
        ref_audio=ref_audio_bytes,
        language=language,
        ref_text=ref_text,
        instruction=instruction,
    )

    logger.info(
        "voice_created",
        voice_id=saved.voice_id,
        voice_type=voice_type,
        has_ref_audio=ref_audio_bytes is not None,
    )

    return SavedVoiceResponse(
        voice_id=saved.voice_id,
        name=saved.name,
        voice_type=saved.voice_type,
        language=saved.language,
        ref_text=saved.ref_text,
        instruction=saved.instruction,
        has_ref_audio=saved.ref_audio_path is not None,
        created_at=saved.created_at,
    )


@router.get("/v1/voices/{voice_id}", response_model=SavedVoiceResponse)
async def get_voice(
    voice_id: str,
    voice_store: VoiceStore = Depends(require_voice_store),  # noqa: B008
) -> SavedVoiceResponse:
    """Get a saved voice by ID."""

    saved = await voice_store.get(voice_id)
    if saved is None:
        raise VoiceNotFoundError(voice_id)

    return SavedVoiceResponse(
        voice_id=saved.voice_id,
        name=saved.name,
        voice_type=saved.voice_type,
        language=saved.language,
        ref_text=saved.ref_text,
        instruction=saved.instruction,
        has_ref_audio=saved.ref_audio_path is not None,
        created_at=saved.created_at,
    )


@router.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(
    voice_id: str,
    voice_store: VoiceStore = Depends(require_voice_store),  # noqa: B008
) -> None:
    """Delete a saved voice."""

    deleted = await voice_store.delete(voice_id)
    if not deleted:
        raise VoiceNotFoundError(voice_id)

    logger.info("voice_deleted", voice_id=voice_id)
