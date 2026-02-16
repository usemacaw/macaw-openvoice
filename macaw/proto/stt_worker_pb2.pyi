from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranscribeFileRequest(_message.Message):
    __slots__ = ("request_id", "audio_data", "language", "response_format", "temperature", "timestamp_granularities", "initial_prompt", "hot_words", "task")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_GRANULARITIES_FIELD_NUMBER: _ClassVar[int]
    INITIAL_PROMPT_FIELD_NUMBER: _ClassVar[int]
    HOT_WORDS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    audio_data: bytes
    language: str
    response_format: str
    temperature: float
    timestamp_granularities: _containers.RepeatedScalarFieldContainer[str]
    initial_prompt: str
    hot_words: _containers.RepeatedScalarFieldContainer[str]
    task: str
    def __init__(self, request_id: _Optional[str] = ..., audio_data: _Optional[bytes] = ..., language: _Optional[str] = ..., response_format: _Optional[str] = ..., temperature: _Optional[float] = ..., timestamp_granularities: _Optional[_Iterable[str]] = ..., initial_prompt: _Optional[str] = ..., hot_words: _Optional[_Iterable[str]] = ..., task: _Optional[str] = ...) -> None: ...

class TranscribeFileResponse(_message.Message):
    __slots__ = ("text", "language", "duration", "segments", "words")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    text: str
    language: str
    duration: float
    segments: _containers.RepeatedCompositeFieldContainer[Segment]
    words: _containers.RepeatedCompositeFieldContainer[Word]
    def __init__(self, text: _Optional[str] = ..., language: _Optional[str] = ..., duration: _Optional[float] = ..., segments: _Optional[_Iterable[_Union[Segment, _Mapping]]] = ..., words: _Optional[_Iterable[_Union[Word, _Mapping]]] = ...) -> None: ...

class AudioFrame(_message.Message):
    __slots__ = ("session_id", "data", "is_last", "initial_prompt", "hot_words", "language")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    INITIAL_PROMPT_FIELD_NUMBER: _ClassVar[int]
    HOT_WORDS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    data: bytes
    is_last: bool
    initial_prompt: str
    hot_words: _containers.RepeatedScalarFieldContainer[str]
    language: str
    def __init__(self, session_id: _Optional[str] = ..., data: _Optional[bytes] = ..., is_last: bool = ..., initial_prompt: _Optional[str] = ..., hot_words: _Optional[_Iterable[str]] = ..., language: _Optional[str] = ...) -> None: ...

class TranscriptEvent(_message.Message):
    __slots__ = ("session_id", "event_type", "text", "segment_id", "start_ms", "end_ms", "language", "confidence", "words")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    event_type: str
    text: str
    segment_id: int
    start_ms: int
    end_ms: int
    language: str
    confidence: float
    words: _containers.RepeatedCompositeFieldContainer[Word]
    def __init__(self, session_id: _Optional[str] = ..., event_type: _Optional[str] = ..., text: _Optional[str] = ..., segment_id: _Optional[int] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., language: _Optional[str] = ..., confidence: _Optional[float] = ..., words: _Optional[_Iterable[_Union[Word, _Mapping]]] = ...) -> None: ...

class Segment(_message.Message):
    __slots__ = ("id", "start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio")
    ID_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    AVG_LOGPROB_FIELD_NUMBER: _ClassVar[int]
    NO_SPEECH_PROB_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_RATIO_FIELD_NUMBER: _ClassVar[int]
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    compression_ratio: float
    def __init__(self, id: _Optional[int] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., text: _Optional[str] = ..., avg_logprob: _Optional[float] = ..., no_speech_prob: _Optional[float] = ..., compression_ratio: _Optional[float] = ...) -> None: ...

class Word(_message.Message):
    __slots__ = ("word", "start", "end", "probability")
    WORD_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    word: str
    start: float
    end: float
    probability: float
    def __init__(self, word: _Optional[str] = ..., start: _Optional[float] = ..., end: _Optional[float] = ..., probability: _Optional[float] = ...) -> None: ...

class CancelRequest(_message.Message):
    __slots__ = ("request_id", "session_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    session_id: str
    def __init__(self, request_id: _Optional[str] = ..., session_id: _Optional[str] = ...) -> None: ...

class CancelResponse(_message.Message):
    __slots__ = ("acknowledged",)
    ACKNOWLEDGED_FIELD_NUMBER: _ClassVar[int]
    acknowledged: bool
    def __init__(self, acknowledged: bool = ...) -> None: ...

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
