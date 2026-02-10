"""Protobuf gRPC definitions para comunicacao runtime <-> worker.

Use relative imports so the package works correctly when installed or
imported from a development checkout (src/ layout). Absolute imports
(`from macaw.proto...`) can fail during import time depending on
sys.path ordering; relative imports are more robust here.
"""

from .stt_worker_pb2 import (
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
from .stt_worker_pb2_grpc import (
    STTWorkerServicer,
    STTWorkerStub,
    add_STTWorkerServicer_to_server,
)
from .tts_worker_pb2 import (
    HealthRequest as TTSHealthRequest,
)
from .tts_worker_pb2 import (
    HealthResponse as TTSHealthResponse,
)
from .tts_worker_pb2 import (
    SynthesizeChunk,
    SynthesizeRequest,
)
from .tts_worker_pb2_grpc import (
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
