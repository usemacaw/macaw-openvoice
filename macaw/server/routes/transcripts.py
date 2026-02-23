"""GET/DELETE /v1/speech-to-text/{transcription_id} -- transcript retrieval and deletion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from macaw.exceptions import TranscriptNotFoundError
from macaw.logging import get_logger
from macaw.server.dependencies import require_transcript_store

if TYPE_CHECKING:
    from macaw.server.transcript_store.interface import TranscriptStore

router = APIRouter(tags=["Speech-To-Text"])

logger = get_logger("server.routes.transcripts")


class StoredTranscriptResponse(BaseModel):
    """JSON response for a stored transcript."""

    transcription_id: str
    text: str
    language: str | None = None
    duration: float | None = None
    model: str | None = None
    created_at: str = ""
    metadata: dict[str, object] = {}


@router.get("/v1/speech-to-text/{transcription_id}", response_model=StoredTranscriptResponse)
async def get_transcript(
    transcription_id: str,
    transcript_store: TranscriptStore = Depends(require_transcript_store),  # noqa: B008
) -> StoredTranscriptResponse:
    """Get a stored transcript by ID."""
    transcript = await transcript_store.get(transcription_id)
    if transcript is None:
        raise TranscriptNotFoundError(transcription_id)

    return StoredTranscriptResponse(
        transcription_id=transcript.transcription_id,
        text=transcript.text,
        language=transcript.language,
        duration=transcript.duration,
        model=transcript.model,
        created_at=transcript.created_at,
        metadata=transcript.metadata,
    )


@router.delete("/v1/speech-to-text/{transcription_id}", status_code=204)
async def delete_transcript(
    transcription_id: str,
    transcript_store: TranscriptStore = Depends(require_transcript_store),  # noqa: B008
) -> None:
    """Delete a stored transcript."""
    deleted = await transcript_store.delete(transcription_id)
    if not deleted:
        raise TranscriptNotFoundError(transcription_id)

    logger.info("transcript_deleted", transcription_id=transcription_id)
