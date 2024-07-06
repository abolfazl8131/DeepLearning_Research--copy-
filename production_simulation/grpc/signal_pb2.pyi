from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SignalREQ(_message.Message):
    __slots__ = ("signal",)
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    signal: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, signal: _Optional[_Iterable[float]] = ...) -> None: ...

class Prediction(_message.Message):
    __slots__ = ("predicton",)
    PREDICTON_FIELD_NUMBER: _ClassVar[int]
    predicton: str
    def __init__(self, predicton: _Optional[str] = ...) -> None: ...
