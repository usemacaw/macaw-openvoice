"""Protobuf gRPC definitions para comunicacao runtime <-> worker."""

from macaw.proto.stt_worker_pb2 import (
    AudioFrame,
    CancelRequest,
    CancelResponse,
    HealthRequest,
    HealthResponse,
    Segment,
    TranscribeFileRequest,
    TranscribeFileResponse,
    TranscriptEvent,
    Word,
)
from macaw.proto.stt_worker_pb2_grpc import (
    STTWorkerServicer,
    STTWorkerStub,
    add_STTWorkerServicer_to_server,
)
from macaw.proto.tts_worker_pb2 import (
    HealthRequest as TTSHealthRequest,
)
from macaw.proto.tts_worker_pb2 import (
    HealthResponse as TTSHealthResponse,
)
from macaw.proto.tts_worker_pb2 import (
    SynthesizeChunk,
    SynthesizeRequest,
)
from macaw.proto.tts_worker_pb2_grpc import (
    TTSWorkerServicer,
    TTSWorkerStub,
    add_TTSWorkerServicer_to_server,
)

__all__ = [
    "AudioFrame",
    "CancelRequest",
    "CancelResponse",
    "HealthRequest",
    "HealthResponse",
    "STTWorkerServicer",
    "STTWorkerStub",
    "Segment",
    "SynthesizeChunk",
    "SynthesizeRequest",
    "TTSHealthRequest",
    "TTSHealthResponse",
    "TTSWorkerServicer",
    "TTSWorkerStub",
    "TranscribeFileRequest",
    "TranscribeFileResponse",
    "TranscriptEvent",
    "Word",
    "add_STTWorkerServicer_to_server",
    "add_TTSWorkerServicer_to_server",
]
