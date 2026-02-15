"""Logica compartilhada entre rotas de audio (transcriptions e translations)."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from fastapi import UploadFile  # noqa: TC002

from macaw._types import ResponseFormat
from macaw.exceptions import AudioFormatError, AudioTooLargeError, InvalidRequestError
from macaw.logging import get_logger
from macaw.server.constants import ALLOWED_AUDIO_CONTENT_TYPES, MAX_FILE_SIZE_BYTES
from macaw.server.formatters import format_response
from macaw.server.models.requests import TranscribeRequest

if TYPE_CHECKING:
    from macaw.postprocessing.pipeline import PostProcessingPipeline
    from macaw.preprocessing.pipeline import AudioPreprocessingPipeline
    from macaw.scheduler.scheduler import Scheduler

logger = get_logger("server.routes")


async def handle_audio_request(
    *,
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float,
    task: str,
    scheduler: Scheduler,
    preprocessing_pipeline: AudioPreprocessingPipeline | None = None,
    postprocessing_pipeline: PostProcessingPipeline | None = None,
    itn: bool = True,
) -> Any:
    """Processa request de audio (transcricao ou traducao).

    Valida input, le audio, aplica preprocessing, envia ao scheduler,
    aplica post-processing, formata resposta.

    Args:
        file: Arquivo de audio enviado pelo cliente.
        model: Nome do modelo no registry.
        language: Codigo ISO 639-1 ou None (auto-detect).
        prompt: Contexto para guiar transcricao.
        response_format: Formato de resposta desejado.
        temperature: Temperatura de sampling.
        task: "transcribe" ou "translate".
        scheduler: Scheduler para rotear ao worker.
        preprocessing_pipeline: Pipeline de preprocessamento de audio (opcional).
        postprocessing_pipeline: Pipeline de pos-processamento de texto (opcional).
        itn: Se True (default), aplica post-processing ao resultado.

    Returns:
        Response formatada conforme response_format.
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "audio_request",
        task=task,
        request_id=request_id,
        model=model,
        language=language,
        response_format=response_format,
    )

    # Validar response_format
    try:
        fmt = ResponseFormat(response_format)
    except ValueError:
        valid = ", ".join(e.value for e in ResponseFormat)
        raise InvalidRequestError(
            f"Invalid response_format '{response_format}'. Accepted values: {valid}"
        ) from None

    # Validar content-type do arquivo
    if file.content_type and file.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise AudioFormatError(
            f"Unsupported content-type '{file.content_type}'. "
            "Accepted formats: WAV, MP3, FLAC, OGG, WebM"
        )

    # Validar tamanho (pre-leitura se disponivel)
    if file.size is not None and file.size > MAX_FILE_SIZE_BYTES:
        raise AudioTooLargeError(file.size, MAX_FILE_SIZE_BYTES)

    # Ler audio com limite para prevenir OOM em uploads sem Content-Length
    audio_data = await file.read(MAX_FILE_SIZE_BYTES + 1)

    if len(audio_data) > MAX_FILE_SIZE_BYTES:
        raise AudioTooLargeError(len(audio_data), MAX_FILE_SIZE_BYTES)

    # Aplicar preprocessing se pipeline configurado
    if preprocessing_pipeline is not None:
        audio_data = preprocessing_pipeline.process(audio_data)

    request = TranscribeRequest(
        request_id=request_id,
        model_name=model,
        audio_data=audio_data,
        language=language,
        response_format=fmt,
        temperature=temperature,
        initial_prompt=prompt,
        task=task,
    )

    result = await scheduler.transcribe(request)

    # Aplicar post-processing se pipeline configurado e ITN habilitado
    if postprocessing_pipeline is not None and itn:
        result = postprocessing_pipeline.process_result(result)

    return format_response(result, fmt, task=task)
