from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ("request_id", "text", "voice", "sample_rate", "speed", "language", "ref_audio", "ref_text", "instruction", "codec", "include_alignment", "alignment_granularity", "seed", "text_normalization", "temperature", "top_k", "top_p")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    REF_AUDIO_FIELD_NUMBER: _ClassVar[int]
    REF_TEXT_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_ALIGNMENT_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_GRANULARITY_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    TEXT_NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    TOP_P_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    voice: str
    sample_rate: int
    speed: float
    language: str
    ref_audio: bytes
    ref_text: str
    instruction: str
    codec: str
    include_alignment: bool
    alignment_granularity: str
    seed: int
    text_normalization: str
    temperature: float
    top_k: int
    top_p: float
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., voice: _Optional[str] = ..., sample_rate: _Optional[int] = ..., speed: _Optional[float] = ..., language: _Optional[str] = ..., ref_audio: _Optional[bytes] = ..., ref_text: _Optional[str] = ..., instruction: _Optional[str] = ..., codec: _Optional[str] = ..., include_alignment: bool = ..., alignment_granularity: _Optional[str] = ..., seed: _Optional[int] = ..., text_normalization: _Optional[str] = ..., temperature: _Optional[float] = ..., top_k: _Optional[int] = ..., top_p: _Optional[float] = ...) -> None: ...

class AlignmentItem(_message.Message):
    __slots__ = ("text", "start_ms", "duration_ms")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    text: str
    start_ms: int
    duration_ms: int
    def __init__(self, text: _Optional[str] = ..., start_ms: _Optional[int] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class ChunkAlignment(_message.Message):
    __slots__ = ("items", "granularity")
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    GRANULARITY_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[AlignmentItem]
    granularity: str
    def __init__(self, items: _Optional[_Iterable[_Union[AlignmentItem, _Mapping]]] = ..., granularity: _Optional[str] = ...) -> None: ...

class SynthesizeChunk(_message.Message):
    __slots__ = ("audio_data", "is_last", "duration", "codec", "alignment", "normalized_alignment")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_ALIGNMENT_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    is_last: bool
    duration: float
    codec: str
    alignment: ChunkAlignment
    normalized_alignment: ChunkAlignment
    def __init__(self, audio_data: _Optional[bytes] = ..., is_last: bool = ..., duration: _Optional[float] = ..., codec: _Optional[str] = ..., alignment: _Optional[_Union[ChunkAlignment, _Mapping]] = ..., normalized_alignment: _Optional[_Union[ChunkAlignment, _Mapping]] = ...) -> None: ...

class ListVoicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class VoiceInfoProto(_message.Message):
    __slots__ = ("voice_id", "name", "language", "gender")
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    voice_id: str
    name: str
    language: str
    gender: str
    def __init__(self, voice_id: _Optional[str] = ..., name: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ...) -> None: ...

class ListVoicesResponse(_message.Message):
    __slots__ = ("voices",)
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedCompositeFieldContainer[VoiceInfoProto]
    def __init__(self, voices: _Optional[_Iterable[_Union[VoiceInfoProto, _Mapping]]] = ...) -> None: ...

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
