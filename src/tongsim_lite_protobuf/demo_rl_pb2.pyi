from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OrientationMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ORIENTATION_KEEP_CURRENT: _ClassVar[OrientationMode]
    ORIENTATION_FACE_MOVEMENT: _ClassVar[OrientationMode]
    ORIENTATION_GIVEN: _ClassVar[OrientationMode]

class CollisionObjectType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECT_WORLD_STATIC: _ClassVar[CollisionObjectType]
    OBJECT_WORLD_DYNAMIC: _ClassVar[CollisionObjectType]
    OBJECT_PAWN: _ClassVar[CollisionObjectType]
    OBJECT_PHYSICS_BODY: _ClassVar[CollisionObjectType]
    OBJECT_VEHICLE: _ClassVar[CollisionObjectType]
    OBJECT_DESTRUCTIBLE: _ClassVar[CollisionObjectType]
ORIENTATION_KEEP_CURRENT: OrientationMode
ORIENTATION_FACE_MOVEMENT: OrientationMode
ORIENTATION_GIVEN: OrientationMode
OBJECT_WORLD_STATIC: CollisionObjectType
OBJECT_WORLD_DYNAMIC: CollisionObjectType
OBJECT_PAWN: CollisionObjectType
OBJECT_PHYSICS_BODY: CollisionObjectType
OBJECT_VEHICLE: CollisionObjectType
OBJECT_DESTRUCTIBLE: CollisionObjectType

class ActorState(_message.Message):
    __slots__ = ("object_info", "location", "unit_forward_vector", "unit_right_vector", "bounding_box", "tag", "current_speed", "destroyed")
    OBJECT_INFO_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    UNIT_FORWARD_VECTOR_FIELD_NUMBER: _ClassVar[int]
    UNIT_RIGHT_VECTOR_FIELD_NUMBER: _ClassVar[int]
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    CURRENT_SPEED_FIELD_NUMBER: _ClassVar[int]
    DESTROYED_FIELD_NUMBER: _ClassVar[int]
    object_info: _object_pb2.ObjectInfo
    location: _common_pb2.Vector3f
    unit_forward_vector: _common_pb2.Vector3f
    unit_right_vector: _common_pb2.Vector3f
    bounding_box: _common_pb2.Box
    tag: str
    current_speed: float
    destroyed: bool
    def __init__(self, object_info: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ..., location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., unit_forward_vector: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., unit_right_vector: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., bounding_box: _Optional[_Union[_common_pb2.Box, _Mapping]] = ..., tag: _Optional[str] = ..., current_speed: _Optional[float] = ..., destroyed: bool = ...) -> None: ...

class DemoRLState(_message.Message):
    __slots__ = ("actor_states",)
    ACTOR_STATES_FIELD_NUMBER: _ClassVar[int]
    actor_states: _containers.RepeatedCompositeFieldContainer[ActorState]
    def __init__(self, actor_states: _Optional[_Iterable[_Union[ActorState, _Mapping]]] = ...) -> None: ...

class SimpleMoveTowardsRequest(_message.Message):
    __slots__ = ("target_location", "orientation_mode", "given_orientation", "actor_id", "speed_uu_per_sec", "tolerance_uu")
    TARGET_LOCATION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_MODE_FIELD_NUMBER: _ClassVar[int]
    GIVEN_ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    SPEED_UU_PER_SEC_FIELD_NUMBER: _ClassVar[int]
    TOLERANCE_UU_FIELD_NUMBER: _ClassVar[int]
    target_location: _common_pb2.Vector3f
    orientation_mode: OrientationMode
    given_orientation: _common_pb2.Vector3f
    actor_id: _object_pb2.ObjectId
    speed_uu_per_sec: float
    tolerance_uu: float
    def __init__(self, target_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., orientation_mode: _Optional[_Union[OrientationMode, str]] = ..., given_orientation: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., speed_uu_per_sec: _Optional[float] = ..., tolerance_uu: _Optional[float] = ...) -> None: ...

class HitResult(_message.Message):
    __slots__ = ("hit_actor",)
    HIT_ACTOR_FIELD_NUMBER: _ClassVar[int]
    hit_actor: ActorState
    def __init__(self, hit_actor: _Optional[_Union[ActorState, _Mapping]] = ...) -> None: ...

class SimpleMoveTowardsResponse(_message.Message):
    __slots__ = ("current_location", "hit_result")
    CURRENT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    HIT_RESULT_FIELD_NUMBER: _ClassVar[int]
    current_location: _common_pb2.Vector3f
    hit_result: HitResult
    def __init__(self, current_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., hit_result: _Optional[_Union[HitResult, _Mapping]] = ...) -> None: ...

class GetActorStateRequest(_message.Message):
    __slots__ = ("actor_id",)
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    actor_id: _object_pb2.ObjectId
    def __init__(self, actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class GetActorStateResponse(_message.Message):
    __slots__ = ("actor_state",)
    ACTOR_STATE_FIELD_NUMBER: _ClassVar[int]
    actor_state: ActorState
    def __init__(self, actor_state: _Optional[_Union[ActorState, _Mapping]] = ...) -> None: ...

class SetActorTransformRequest(_message.Message):
    __slots__ = ("actor_id", "transform")
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    actor_id: _object_pb2.ObjectId
    transform: _common_pb2.Transform
    def __init__(self, actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class GetActorTransformRequest(_message.Message):
    __slots__ = ("actor_id",)
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    actor_id: _object_pb2.ObjectId
    def __init__(self, actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class GetActorTransformResponse(_message.Message):
    __slots__ = ("transform",)
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    transform: _common_pb2.Transform
    def __init__(self, transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class SpawnActorRequest(_message.Message):
    __slots__ = ("blueprint", "transform", "name", "tags")
    BLUEPRINT_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    blueprint: str
    transform: _common_pb2.Transform
    name: str
    tags: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, blueprint: _Optional[str] = ..., transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., name: _Optional[str] = ..., tags: _Optional[_Iterable[str]] = ...) -> None: ...

class SpawnActorResponse(_message.Message):
    __slots__ = ("actor",)
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    actor: _object_pb2.ObjectInfo
    def __init__(self, actor: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ...) -> None: ...

class ExecConsoleCommandRequest(_message.Message):
    __slots__ = ("command", "write_to_log")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    WRITE_TO_LOG_FIELD_NUMBER: _ClassVar[int]
    command: str
    write_to_log: bool
    def __init__(self, command: _Optional[str] = ..., write_to_log: bool = ...) -> None: ...

class ExecConsoleCommandResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class QueryNavigationPathRequest(_message.Message):
    __slots__ = ("start", "end", "allow_partial", "require_navigable_end_location", "cost_limit")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    ALLOW_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    REQUIRE_NAVIGABLE_END_LOCATION_FIELD_NUMBER: _ClassVar[int]
    COST_LIMIT_FIELD_NUMBER: _ClassVar[int]
    start: _common_pb2.Vector3f
    end: _common_pb2.Vector3f
    allow_partial: bool
    require_navigable_end_location: bool
    cost_limit: float
    def __init__(self, start: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., end: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., allow_partial: bool = ..., require_navigable_end_location: bool = ..., cost_limit: _Optional[float] = ...) -> None: ...

class QueryNavigationPathResponse(_message.Message):
    __slots__ = ("path_points", "is_partial", "path_cost", "path_length")
    PATH_POINTS_FIELD_NUMBER: _ClassVar[int]
    IS_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    PATH_COST_FIELD_NUMBER: _ClassVar[int]
    PATH_LENGTH_FIELD_NUMBER: _ClassVar[int]
    path_points: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector3f]
    is_partial: bool
    path_cost: float
    path_length: float
    def __init__(self, path_points: _Optional[_Iterable[_Union[_common_pb2.Vector3f, _Mapping]]] = ..., is_partial: bool = ..., path_cost: _Optional[float] = ..., path_length: _Optional[float] = ...) -> None: ...

class DestroyActorRequest(_message.Message):
    __slots__ = ("actor_id", "force")
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    actor_id: _object_pb2.ObjectId
    force: bool
    def __init__(self, actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., force: bool = ...) -> None: ...

class LineTraceByObjectJob(_message.Message):
    __slots__ = ("start", "end", "object_types", "trace_complex", "actors_to_ignore")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    OBJECT_TYPES_FIELD_NUMBER: _ClassVar[int]
    TRACE_COMPLEX_FIELD_NUMBER: _ClassVar[int]
    ACTORS_TO_IGNORE_FIELD_NUMBER: _ClassVar[int]
    start: _common_pb2.Vector3f
    end: _common_pb2.Vector3f
    object_types: _containers.RepeatedScalarFieldContainer[CollisionObjectType]
    trace_complex: bool
    actors_to_ignore: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    def __init__(self, start: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., end: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., object_types: _Optional[_Iterable[_Union[CollisionObjectType, str]]] = ..., trace_complex: bool = ..., actors_to_ignore: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ...) -> None: ...

class BatchSingleLineTraceByObjectRequest(_message.Message):
    __slots__ = ("jobs",)
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[LineTraceByObjectJob]
    def __init__(self, jobs: _Optional[_Iterable[_Union[LineTraceByObjectJob, _Mapping]]] = ...) -> None: ...

class SingleLineTraceByObjectResult(_message.Message):
    __slots__ = ("job_index", "blocking_hit", "distance", "impact_point", "actor_state")
    JOB_INDEX_FIELD_NUMBER: _ClassVar[int]
    BLOCKING_HIT_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    IMPACT_POINT_FIELD_NUMBER: _ClassVar[int]
    ACTOR_STATE_FIELD_NUMBER: _ClassVar[int]
    job_index: int
    blocking_hit: bool
    distance: float
    impact_point: _common_pb2.Vector3f
    actor_state: ActorState
    def __init__(self, job_index: _Optional[int] = ..., blocking_hit: bool = ..., distance: _Optional[float] = ..., impact_point: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., actor_state: _Optional[_Union[ActorState, _Mapping]] = ...) -> None: ...

class BatchSingleLineTraceByObjectResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SingleLineTraceByObjectResult]
    def __init__(self, results: _Optional[_Iterable[_Union[SingleLineTraceByObjectResult, _Mapping]]] = ...) -> None: ...

class BatchMultiLineTraceByObjectRequest(_message.Message):
    __slots__ = ("jobs", "enable_debug_draw")
    JOBS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DEBUG_DRAW_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[LineTraceByObjectJob]
    enable_debug_draw: bool
    def __init__(self, jobs: _Optional[_Iterable[_Union[LineTraceByObjectJob, _Mapping]]] = ..., enable_debug_draw: bool = ...) -> None: ...

class MultiLineTraceHit(_message.Message):
    __slots__ = ("distance", "impact_point", "actor_state", "impact_normal")
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    IMPACT_POINT_FIELD_NUMBER: _ClassVar[int]
    ACTOR_STATE_FIELD_NUMBER: _ClassVar[int]
    IMPACT_NORMAL_FIELD_NUMBER: _ClassVar[int]
    distance: float
    impact_point: _common_pb2.Vector3f
    actor_state: ActorState
    impact_normal: _common_pb2.Vector3f
    def __init__(self, distance: _Optional[float] = ..., impact_point: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., actor_state: _Optional[_Union[ActorState, _Mapping]] = ..., impact_normal: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ...) -> None: ...

class MultiLineTraceResult(_message.Message):
    __slots__ = ("job_index", "hits")
    JOB_INDEX_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    job_index: int
    hits: _containers.RepeatedCompositeFieldContainer[MultiLineTraceHit]
    def __init__(self, job_index: _Optional[int] = ..., hits: _Optional[_Iterable[_Union[MultiLineTraceHit, _Mapping]]] = ...) -> None: ...

class BatchMultiLineTraceByObjectResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[MultiLineTraceResult]
    def __init__(self, results: _Optional[_Iterable[_Union[MultiLineTraceResult, _Mapping]]] = ...) -> None: ...
