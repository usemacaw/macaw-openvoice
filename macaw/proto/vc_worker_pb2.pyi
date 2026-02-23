from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VoiceChangeRequest(_message.Message):
    __slots__ = ("request_id", "source_audio", "voice", "sample_rate", "ref_audio", "ref_text", "voice_settings_json")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_AUDIO_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    REF_AUDIO_FIELD_NUMBER: _ClassVar[int]
    REF_TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_SETTINGS_JSON_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    source_audio: bytes
    voice: str
    sample_rate: int
    ref_audio: bytes
    ref_text: str
    voice_settings_json: str
    def __init__(self, request_id: _Optional[str] = ..., source_audio: _Optional[bytes] = ..., voice: _Optional[str] = ..., sample_rate: _Optional[int] = ..., ref_audio: _Optional[bytes] = ..., ref_text: _Optional[str] = ..., voice_settings_json: _Optional[str] = ...) -> None: ...

class VoiceChangeChunk(_message.Message):
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
    __slots__ = ("status", "model_name", "engine")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_name: str
    engine: str
    def __init__(self, status: _Optional[str] = ..., model_name: _Optional[str] = ..., engine: _Optional[str] = ...) -> None: ...
