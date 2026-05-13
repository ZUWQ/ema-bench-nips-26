from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Vector3f(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Rotatorf(_message.Message):
    __slots__ = ("roll_deg", "pitch_deg", "yaw_deg")
    ROLL_DEG_FIELD_NUMBER: _ClassVar[int]
    PITCH_DEG_FIELD_NUMBER: _ClassVar[int]
    YAW_DEG_FIELD_NUMBER: _ClassVar[int]
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    def __init__(self, roll_deg: _Optional[float] = ..., pitch_deg: _Optional[float] = ..., yaw_deg: _Optional[float] = ...) -> None: ...

class Transform(_message.Message):
    __slots__ = ("location", "rotation", "scale")
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    location: Vector3f
    rotation: Rotatorf
    scale: Vector3f
    def __init__(self, location: _Optional[_Union[Vector3f, _Mapping]] = ..., rotation: _Optional[_Union[Rotatorf, _Mapping]] = ..., scale: _Optional[_Union[Vector3f, _Mapping]] = ...) -> None: ...

class Box(_message.Message):
    __slots__ = ("min_vertex", "max_vertex")
    MIN_VERTEX_FIELD_NUMBER: _ClassVar[int]
    MAX_VERTEX_FIELD_NUMBER: _ClassVar[int]
    min_vertex: Vector3f
    max_vertex: Vector3f
    def __init__(self, min_vertex: _Optional[_Union[Vector3f, _Mapping]] = ..., max_vertex: _Optional[_Union[Vector3f, _Mapping]] = ...) -> None: ...
