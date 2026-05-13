from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LoadArenaRequest(_message.Message):
    __slots__ = ("level_asset_path", "anchor", "make_visible")
    LEVEL_ASSET_PATH_FIELD_NUMBER: _ClassVar[int]
    ANCHOR_FIELD_NUMBER: _ClassVar[int]
    MAKE_VISIBLE_FIELD_NUMBER: _ClassVar[int]
    level_asset_path: str
    anchor: _common_pb2.Transform
    make_visible: bool
    def __init__(self, level_asset_path: _Optional[str] = ..., anchor: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., make_visible: bool = ...) -> None: ...

class LoadArenaResponse(_message.Message):
    __slots__ = ("arena_id",)
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class DestroyArenaRequest(_message.Message):
    __slots__ = ("arena_id",)
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class ResetArenaRequest(_message.Message):
    __slots__ = ("arena_id",)
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class SetArenaVisibleRequest(_message.Message):
    __slots__ = ("arena_id", "visible")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    VISIBLE_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    visible: bool
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., visible: bool = ...) -> None: ...

class ArenaDescriptor(_message.Message):
    __slots__ = ("arena_id", "asset_path", "anchor", "is_loaded", "is_visible", "num_actors")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    ASSET_PATH_FIELD_NUMBER: _ClassVar[int]
    ANCHOR_FIELD_NUMBER: _ClassVar[int]
    IS_LOADED_FIELD_NUMBER: _ClassVar[int]
    IS_VISIBLE_FIELD_NUMBER: _ClassVar[int]
    NUM_ACTORS_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    asset_path: str
    anchor: _common_pb2.Transform
    is_loaded: bool
    is_visible: bool
    num_actors: int
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., asset_path: _Optional[str] = ..., anchor: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., is_loaded: bool = ..., is_visible: bool = ..., num_actors: _Optional[int] = ...) -> None: ...

class ListArenasRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListArenasResponse(_message.Message):
    __slots__ = ("arenas",)
    ARENAS_FIELD_NUMBER: _ClassVar[int]
    arenas: _containers.RepeatedCompositeFieldContainer[ArenaDescriptor]
    def __init__(self, arenas: _Optional[_Iterable[_Union[ArenaDescriptor, _Mapping]]] = ...) -> None: ...

class SpawnActorInArenaRequest(_message.Message):
    __slots__ = ("arena_id", "class_path", "local_transform")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    CLASS_PATH_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    class_path: str
    local_transform: _common_pb2.Transform
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., class_path: _Optional[str] = ..., local_transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class SpawnActorInArenaResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: _object_pb2.ObjectInfo
    def __init__(self, actor: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ...) -> None: ...

class SetActorPoseLocalRequest(_message.Message):
    __slots__ = ("arena_id", "actor_id", "local_transform", "reset_physics")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    RESET_PHYSICS_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    actor_id: _object_pb2.ObjectId
    local_transform: _common_pb2.Transform
    reset_physics: bool
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., local_transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., reset_physics: bool = ...) -> None: ...

class GetActorPoseLocalRequest(_message.Message):
    __slots__ = ("arena_id", "actor_id")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    actor_id: _object_pb2.ObjectId
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class GetActorPoseLocalResponse(_message.Message):
    __slots__ = ("local_transform",)
    LOCAL_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    local_transform: _common_pb2.Transform
    def __init__(self, local_transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class LocalToWorldRequest(_message.Message):
    __slots__ = ("arena_id", "local")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    local: _common_pb2.Transform
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., local: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class LocalToWorldResponse(_message.Message):
    __slots__ = ("world",)
    WORLD_FIELD_NUMBER: _ClassVar[int]
    world: _common_pb2.Transform
    def __init__(self, world: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class WorldToLocalRequest(_message.Message):
    __slots__ = ("arena_id", "world")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    WORLD_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    world: _common_pb2.Transform
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., world: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class WorldToLocalResponse(_message.Message):
    __slots__ = ("local",)
    LOCAL_FIELD_NUMBER: _ClassVar[int]
    local: _common_pb2.Transform
    def __init__(self, local: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class DestroyActorInArenaRequest(_message.Message):
    __slots__ = ("arena_id", "actor_id")
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    actor_id: _object_pb2.ObjectId
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class SimpleMoveTowardsInArenaRequest(_message.Message):
    __slots__ = ("arena_id", "target_local_location", "orientation_mode", "given_forward")
    class OrientationMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ORIENTATION_KEEP_CURRENT: _ClassVar[SimpleMoveTowardsInArenaRequest.OrientationMode]
        ORIENTATION_FACE_MOVEMENT: _ClassVar[SimpleMoveTowardsInArenaRequest.OrientationMode]
        ORIENTATION_GIVEN: _ClassVar[SimpleMoveTowardsInArenaRequest.OrientationMode]
    ORIENTATION_KEEP_CURRENT: SimpleMoveTowardsInArenaRequest.OrientationMode
    ORIENTATION_FACE_MOVEMENT: SimpleMoveTowardsInArenaRequest.OrientationMode
    ORIENTATION_GIVEN: SimpleMoveTowardsInArenaRequest.OrientationMode
    ARENA_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_LOCAL_LOCATION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_MODE_FIELD_NUMBER: _ClassVar[int]
    GIVEN_FORWARD_FIELD_NUMBER: _ClassVar[int]
    arena_id: _object_pb2.ObjectId
    target_local_location: _common_pb2.Vector3f
    orientation_mode: SimpleMoveTowardsInArenaRequest.OrientationMode
    given_forward: _common_pb2.Vector3f
    def __init__(self, arena_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., target_local_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., orientation_mode: _Optional[_Union[SimpleMoveTowardsInArenaRequest.OrientationMode, str]] = ..., given_forward: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ...) -> None: ...

class SimpleMoveTowardsInArenaResponse(_message.Message):
    __slots__ = ("current_location", "hit_result")
    class HitResult(_message.Message):
        __slots__ = ("hit_actor",)
        HIT_ACTOR_FIELD_NUMBER: _ClassVar[int]
        hit_actor: str
        def __init__(self, hit_actor: _Optional[str] = ...) -> None: ...
    CURRENT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    HIT_RESULT_FIELD_NUMBER: _ClassVar[int]
    current_location: _common_pb2.Vector3f
    hit_result: SimpleMoveTowardsInArenaResponse.HitResult
    def __init__(self, current_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., hit_result: _Optional[_Union[SimpleMoveTowardsInArenaResponse.HitResult, _Mapping]] = ...) -> None: ...
