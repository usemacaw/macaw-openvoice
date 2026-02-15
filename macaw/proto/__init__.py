"""Protobuf gRPC definitions for runtime <-> worker communication.

Re-exports all generated protobuf types (STT + TTS) so callers import
from ``macaw.proto`` instead of the individual ``_pb2`` modules.
Uses relative imports for robustness across install and development layouts.
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
    ListVoicesRequest,
    ListVoicesResponse,
    SynthesizeChunk,
    SynthesizeRequest,
    VoiceInfoProto,
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
    "ListVoicesRequest",
    "ListVoicesResponse",
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
    "VoiceInfoProto",
    "Word",
    "add_STTWorkerServicer_to_server",
    "add_TTSWorkerServicer_to_server",
]
