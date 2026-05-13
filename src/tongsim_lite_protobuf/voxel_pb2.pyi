from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QueryVoxelRequest(_message.Message):
    __slots__ = ("transform", "voxel_num_x", "voxel_num_y", "voxel_num_z", "extent", "ActorsToIgnore")
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    VOXEL_NUM_X_FIELD_NUMBER: _ClassVar[int]
    VOXEL_NUM_Y_FIELD_NUMBER: _ClassVar[int]
    VOXEL_NUM_Z_FIELD_NUMBER: _ClassVar[int]
    EXTENT_FIELD_NUMBER: _ClassVar[int]
    ACTORSTOIGNORE_FIELD_NUMBER: _ClassVar[int]
    transform: _common_pb2.Transform
    voxel_num_x: int
    voxel_num_y: int
    voxel_num_z: int
    extent: _common_pb2.Vector3f
    ActorsToIgnore: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    def __init__(self, transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., voxel_num_x: _Optional[int] = ..., voxel_num_y: _Optional[int] = ..., voxel_num_z: _Optional[int] = ..., extent: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., ActorsToIgnore: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ...) -> None: ...

class Voxel(_message.Message):
    __slots__ = ("voxel_buffer",)
    VOXEL_BUFFER_FIELD_NUMBER: _ClassVar[int]
    voxel_buffer: bytes
    def __init__(self, voxel_buffer: _Optional[bytes] = ...) -> None: ...
