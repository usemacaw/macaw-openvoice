from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ("request_id", "text", "voice", "sample_rate", "speed")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    voice: str
    sample_rate: int
    speed: float
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., voice: _Optional[str] = ..., sample_rate: _Optional[int] = ..., speed: _Optional[float] = ...) -> None: ...

class SynthesizeChunk(_message.Message):
    __slots__ = ("audio_data", "is_last", "duration")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    is_last: bool
    duration: float
    def __init__(self, audio_data: _Optional[bytes] = ..., is_last: bool = ..., duration: _Optional[float] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "model_name", "engine", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_name: str
    engine: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, status: _Optional[str] = ..., model_name: _Optional[str] = ..., engine: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
