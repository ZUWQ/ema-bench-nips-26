from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectId(_message.Message):
    __slots__ = ("guid",)
    GUID_FIELD_NUMBER: _ClassVar[int]
    guid: bytes
    def __init__(self, guid: _Optional[bytes] = ...) -> None: ...

class ObjectInfo(_message.Message):
    __slots__ = ("id", "class_path", "name")
    ID_FIELD_NUMBER: _ClassVar[int]
    CLASS_PATH_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: ObjectId
    class_path: str
    name: str
    def __init__(self, id: _Optional[_Union[ObjectId, _Mapping]] = ..., class_path: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...
