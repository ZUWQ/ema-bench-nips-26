from tongsim_lite_protobuf import demo_rl_pb2 as _demo_rl_pb2
from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NPCInstructionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Follow: _ClassVar[NPCInstructionType]
    Stop: _ClassVar[NPCInstructionType]
    CarryMask: _ClassVar[NPCInstructionType]

class ExtinguishFireResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Success: _ClassVar[ExtinguishFireResult]
    Fail_CannotExtinguish: _ClassVar[ExtinguishFireResult]
    Fail_NeedSupply: _ClassVar[ExtinguishFireResult]

class health_state_type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    Safe: _ClassVar[health_state_type]
    IsBurning: _ClassVar[health_state_type]
    WithoutMask: _ClassVar[health_state_type]
Follow: NPCInstructionType
Stop: NPCInstructionType
CarryMask: NPCInstructionType
Success: ExtinguishFireResult
Fail_CannotExtinguish: ExtinguishFireResult
Fail_NeedSupply: ExtinguishFireResult
Safe: health_state_type
IsBurning: health_state_type
WithoutMask: health_state_type

class PerceptionInfoRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class ActorInfo(_message.Message):
    __slots__ = ("actor", "mass", "movable", "burning_state", "burned_percentage", "burning_speed")
    ACTOR_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    MOVABLE_FIELD_NUMBER: _ClassVar[int]
    BURNING_STATE_FIELD_NUMBER: _ClassVar[int]
    BURNED_PERCENTAGE_FIELD_NUMBER: _ClassVar[int]
    BURNING_SPEED_FIELD_NUMBER: _ClassVar[int]
    actor: _demo_rl_pb2.ActorState
    mass: float
    movable: bool
    burning_state: bool
    burned_percentage: float
    burning_speed: float
    def __init__(self, actor: _Optional[_Union[_demo_rl_pb2.ActorState, _Mapping]] = ..., mass: _Optional[float] = ..., movable: bool = ..., burning_state: bool = ..., burned_percentage: _Optional[float] = ..., burning_speed: _Optional[float] = ...) -> None: ...

class NPCInfo(_message.Message):
    __slots__ = ("object_info", "position", "health", "burning_state", "mass")
    OBJECT_INFO_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    BURNING_STATE_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    object_info: _object_pb2.ObjectInfo
    position: _common_pb2.Vector3f
    health: float
    burning_state: bool
    mass: float
    def __init__(self, object_info: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ..., position: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., health: _Optional[float] = ..., burning_state: bool = ..., mass: _Optional[float] = ...) -> None: ...

class PerceptionInfoResponse(_message.Message):
    __slots__ = ("actor_info", "npc_info")
    ACTOR_INFO_FIELD_NUMBER: _ClassVar[int]
    NPC_INFO_FIELD_NUMBER: _ClassVar[int]
    actor_info: _containers.RepeatedCompositeFieldContainer[ActorInfo]
    npc_info: _containers.RepeatedCompositeFieldContainer[NPCInfo]
    def __init__(self, actor_info: _Optional[_Iterable[_Union[ActorInfo, _Mapping]]] = ..., npc_info: _Optional[_Iterable[_Union[NPCInfo, _Mapping]]] = ...) -> None: ...

class NPCSOSRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    def __init__(self, agent_id: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ...) -> None: ...

class NPCSOSInfo(_message.Message):
    __slots__ = ("agent_id", "orientation", "distance")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    orientation: _containers.RepeatedCompositeFieldContainer[_common_pb2.Rotatorf]
    distance: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., orientation: _Optional[_Iterable[_Union[_common_pb2.Rotatorf, _Mapping]]] = ..., distance: _Optional[_Iterable[float]] = ...) -> None: ...

class NPCSOSResponse(_message.Message):
    __slots__ = ("npc_sos_info",)
    NPC_SOS_INFO_FIELD_NUMBER: _ClassVar[int]
    npc_sos_info: _containers.RepeatedCompositeFieldContainer[NPCSOSInfo]
    def __init__(self, npc_sos_info: _Optional[_Iterable[_Union[NPCSOSInfo, _Mapping]]] = ...) -> None: ...

class NPCInstructionRequest(_message.Message):
    __slots__ = ("instruction", "agent_id")
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    instruction: NPCInstructionType
    agent_id: _object_pb2.ObjectId
    def __init__(self, instruction: _Optional[_Union[NPCInstructionType, str]] = ..., agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class NPCInstructionResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class ExtinguishFireRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class ExtinguishFireResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: ExtinguishFireResult
    def __init__(self, result: _Optional[_Union[ExtinguishFireResult, str]] = ...) -> None: ...

class ReleaseMaskRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class ReleaseMaskResponse(_message.Message):
    __slots__ = ("result", "mask_id", "mask_position")
    class ReleaseMaskResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Success: _ClassVar[ReleaseMaskResponse.ReleaseMaskResult]
        Fail_NothingtoRelease: _ClassVar[ReleaseMaskResponse.ReleaseMaskResult]
        Fail_NeedSupply: _ClassVar[ReleaseMaskResponse.ReleaseMaskResult]
    Success: ReleaseMaskResponse.ReleaseMaskResult
    Fail_NothingtoRelease: ReleaseMaskResponse.ReleaseMaskResult
    Fail_NeedSupply: ReleaseMaskResponse.ReleaseMaskResult
    RESULT_FIELD_NUMBER: _ClassVar[int]
    MASK_ID_FIELD_NUMBER: _ClassVar[int]
    MASK_POSITION_FIELD_NUMBER: _ClassVar[int]
    result: ReleaseMaskResponse.ReleaseMaskResult
    mask_id: _object_pb2.ObjectId
    mask_position: _common_pb2.Vector3f
    def __init__(self, result: _Optional[_Union[ReleaseMaskResponse.ReleaseMaskResult, str]] = ..., mask_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., mask_position: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ...) -> None: ...

class SendSupplyCmdRequest(_message.Message):
    __slots__ = ("supplier_id", "receiver_id", "type", "amount")
    class SupplyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        HEAL: _ClassVar[SendSupplyCmdRequest.SupplyType]
        WATER: _ClassVar[SendSupplyCmdRequest.SupplyType]
        MASK: _ClassVar[SendSupplyCmdRequest.SupplyType]
    HEAL: SendSupplyCmdRequest.SupplyType
    WATER: SendSupplyCmdRequest.SupplyType
    MASK: SendSupplyCmdRequest.SupplyType
    SUPPLIER_ID_FIELD_NUMBER: _ClassVar[int]
    RECEIVER_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_FIELD_NUMBER: _ClassVar[int]
    supplier_id: _object_pb2.ObjectId
    receiver_id: _object_pb2.ObjectId
    type: SendSupplyCmdRequest.SupplyType
    amount: float
    def __init__(self, supplier_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., receiver_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., type: _Optional[_Union[SendSupplyCmdRequest.SupplyType, str]] = ..., amount: _Optional[float] = ...) -> None: ...

class SendSupplyCmdResponse(_message.Message):
    __slots__ = ("result",)
    class SendSupplyCmdResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Success: _ClassVar[SendSupplyCmdResponse.SendSupplyCmdResult]
        Fail_CannotCarry: _ClassVar[SendSupplyCmdResponse.SendSupplyCmdResult]
        Fail_NothingtoCarry: _ClassVar[SendSupplyCmdResponse.SendSupplyCmdResult]
        Fail_Overload: _ClassVar[SendSupplyCmdResponse.SendSupplyCmdResult]
    Success: SendSupplyCmdResponse.SendSupplyCmdResult
    Fail_CannotCarry: SendSupplyCmdResponse.SendSupplyCmdResult
    Fail_NothingtoCarry: SendSupplyCmdResponse.SendSupplyCmdResult
    Fail_Overload: SendSupplyCmdResponse.SendSupplyCmdResult
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: SendSupplyCmdResponse.SendSupplyCmdResult
    def __init__(self, result: _Optional[_Union[SendSupplyCmdResponse.SendSupplyCmdResult, str]] = ...) -> None: ...

class LoadSomethingRequest(_message.Message):
    __slots__ = ("agent_id", "target_actor_id")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    target_actor_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., target_actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class LoadSomethingResponse(_message.Message):
    __slots__ = ("result",)
    class LoadSomethingResult(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        Success: _ClassVar[LoadSomethingResponse.LoadSomethingResult]
        Fail_UnableToLoad: _ClassVar[LoadSomethingResponse.LoadSomethingResult]
        Fail_Overload: _ClassVar[LoadSomethingResponse.LoadSomethingResult]
    Success: LoadSomethingResponse.LoadSomethingResult
    Fail_UnableToLoad: LoadSomethingResponse.LoadSomethingResult
    Fail_Overload: LoadSomethingResponse.LoadSomethingResult
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: LoadSomethingResponse.LoadSomethingResult
    def __init__(self, result: _Optional[_Union[LoadSomethingResponse.LoadSomethingResult, str]] = ...) -> None: ...

class UnLoadSomethingRequest(_message.Message):
    __slots__ = ("agent_id", "target_actor_id")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    target_actor_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., target_actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class UnLoadSomethingResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class SelfStateRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class SelfStateResponse(_message.Message):
    __slots__ = ("actor_info", "location", "unit_forward_vector", "unit_right_vector", "head_pitch_angle", "tag", "maxload", "currentload", "rest_usage", "used_usage", "carrying_obj_ids", "following_NPC_ids", "health_state", "health_value")
    ACTOR_INFO_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    UNIT_FORWARD_VECTOR_FIELD_NUMBER: _ClassVar[int]
    UNIT_RIGHT_VECTOR_FIELD_NUMBER: _ClassVar[int]
    HEAD_PITCH_ANGLE_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    MAXLOAD_FIELD_NUMBER: _ClassVar[int]
    CURRENTLOAD_FIELD_NUMBER: _ClassVar[int]
    REST_USAGE_FIELD_NUMBER: _ClassVar[int]
    USED_USAGE_FIELD_NUMBER: _ClassVar[int]
    CARRYING_OBJ_IDS_FIELD_NUMBER: _ClassVar[int]
    FOLLOWING_NPC_IDS_FIELD_NUMBER: _ClassVar[int]
    HEALTH_STATE_FIELD_NUMBER: _ClassVar[int]
    HEALTH_VALUE_FIELD_NUMBER: _ClassVar[int]
    actor_info: _object_pb2.ObjectInfo
    location: _common_pb2.Vector3f
    unit_forward_vector: _common_pb2.Vector3f
    unit_right_vector: _common_pb2.Vector3f
    head_pitch_angle: float
    tag: str
    maxload: float
    currentload: float
    rest_usage: int
    used_usage: int
    carrying_obj_ids: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    following_NPC_ids: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    health_state: _containers.RepeatedScalarFieldContainer[health_state_type]
    health_value: float
    def __init__(self, actor_info: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ..., location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., unit_forward_vector: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., unit_right_vector: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., head_pitch_angle: _Optional[float] = ..., tag: _Optional[str] = ..., maxload: _Optional[float] = ..., currentload: _Optional[float] = ..., rest_usage: _Optional[int] = ..., used_usage: _Optional[int] = ..., carrying_obj_ids: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ..., following_NPC_ids: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ..., health_state: _Optional[_Iterable[_Union[health_state_type, str]]] = ..., health_value: _Optional[float] = ...) -> None: ...

class SendDoorToggleCmdRequest(_message.Message):
    __slots__ = ("agent_id", "door_id", "open")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    DOOR_ID_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    door_id: _object_pb2.ObjectId
    open: bool
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., door_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., open: bool = ...) -> None: ...

class SendDoorToggleCmdResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class SendDialogueCmdRequest(_message.Message):
    __slots__ = ("agent_id", "content", "target_name", "speaker_name")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TARGET_NAME_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_NAME_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    content: str
    target_name: _containers.RepeatedScalarFieldContainer[str]
    speaker_name: str
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., content: _Optional[str] = ..., target_name: _Optional[_Iterable[str]] = ..., speaker_name: _Optional[str] = ...) -> None: ...

class SendDialogueCmdResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class ExtinguisherCfg(_message.Message):
    __slots__ = ("agent_id", "water_capacity", "recover_time")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    WATER_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    RECOVER_TIME_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    water_capacity: int
    recover_time: int
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., water_capacity: _Optional[int] = ..., recover_time: _Optional[int] = ...) -> None: ...

class SetExtinguisherResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class ExtinguisherRotation(_message.Message):
    __slots__ = ("agent_id", "yaw", "pitch")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    YAW_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    yaw: float
    pitch: float
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., yaw: _Optional[float] = ..., pitch: _Optional[float] = ...) -> None: ...

class PauseSceneRequest(_message.Message):
    __slots__ = ("bPause",)
    BPAUSE_FIELD_NUMBER: _ClassVar[int]
    bPause: bool
    def __init__(self, bPause: bool = ...) -> None: ...

class MoveTowardsByNavRequest(_message.Message):
    __slots__ = ("actor_id", "target_location", "accept_radius", "allow_partial", "speed_uu_per_sec")
    ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_LOCATION_FIELD_NUMBER: _ClassVar[int]
    ACCEPT_RADIUS_FIELD_NUMBER: _ClassVar[int]
    ALLOW_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    SPEED_UU_PER_SEC_FIELD_NUMBER: _ClassVar[int]
    actor_id: _object_pb2.ObjectId
    target_location: _common_pb2.Vector3f
    accept_radius: float
    allow_partial: bool
    speed_uu_per_sec: float
    def __init__(self, actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., target_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., accept_radius: _Optional[float] = ..., allow_partial: bool = ..., speed_uu_per_sec: _Optional[float] = ...) -> None: ...

class MoveTowardsByNavResponse(_message.Message):
    __slots__ = ("success", "message", "current_location", "is_partial")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LOCATION_FIELD_NUMBER: _ClassVar[int]
    IS_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    current_location: _common_pb2.Vector3f
    is_partial: bool
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., current_location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., is_partial: bool = ...) -> None: ...

class QueryNavDistanceRequest(_message.Message):
    __slots__ = ("agent_id", "target_id", "allow_partial")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TARGET_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOW_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    target_id: _object_pb2.ObjectId
    allow_partial: bool
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., target_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., allow_partial: bool = ...) -> None: ...

class QueryNavDistanceResponse(_message.Message):
    __slots__ = ("distance",)
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    distance: float
    def __init__(self, distance: _Optional[float] = ...) -> None: ...

class BurnedStateResponse(_message.Message):
    __slots__ = ("unburned_obj_num", "burning_obj_num", "extinguished_obj_num", "be_watered_obj_num", "obj_total_num")
    UNBURNED_OBJ_NUM_FIELD_NUMBER: _ClassVar[int]
    BURNING_OBJ_NUM_FIELD_NUMBER: _ClassVar[int]
    EXTINGUISHED_OBJ_NUM_FIELD_NUMBER: _ClassVar[int]
    BE_WATERED_OBJ_NUM_FIELD_NUMBER: _ClassVar[int]
    OBJ_TOTAL_NUM_FIELD_NUMBER: _ClassVar[int]
    unburned_obj_num: int
    burning_obj_num: int
    extinguished_obj_num: int
    be_watered_obj_num: int
    obj_total_num: int
    def __init__(self, unburned_obj_num: _Optional[int] = ..., burning_obj_num: _Optional[int] = ..., extinguished_obj_num: _Optional[int] = ..., be_watered_obj_num: _Optional[int] = ..., obj_total_num: _Optional[int] = ...) -> None: ...

class ObjHealthResponse(_message.Message):
    __slots__ = ("obj_sum_hp",)
    OBJ_SUM_HP_FIELD_NUMBER: _ClassVar[int]
    obj_sum_hp: float
    def __init__(self, obj_sum_hp: _Optional[float] = ...) -> None: ...

class NPCHealthResponse(_message.Message):
    __slots__ = ("npc_id", "npc_hp")
    NPC_ID_FIELD_NUMBER: _ClassVar[int]
    NPC_HP_FIELD_NUMBER: _ClassVar[int]
    npc_id: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    npc_hp: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, npc_id: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ..., npc_hp: _Optional[_Iterable[float]] = ...) -> None: ...

class OutFireResultResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class ExtinguishedObjectInfoList(_message.Message):
    __slots__ = ("objects",)
    OBJECTS_FIELD_NUMBER: _ClassVar[int]
    objects: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectInfo]
    def __init__(self, objects: _Optional[_Iterable[_Union[_object_pb2.ObjectInfo, _Mapping]]] = ...) -> None: ...

class DestroyedObjects(_message.Message):
    __slots__ = ("objects_name",)
    OBJECTS_NAME_FIELD_NUMBER: _ClassVar[int]
    objects_name: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, objects_name: _Optional[_Iterable[str]] = ...) -> None: ...

class GetAgentExtinguishedObjectsResponse(_message.Message):
    __slots__ = ("agent_ids", "agent_extinguished_objects")
    class AgentExtinguishedObjectsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: ExtinguishedObjectInfoList
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[ExtinguishedObjectInfoList, _Mapping]] = ...) -> None: ...
    AGENT_IDS_FIELD_NUMBER: _ClassVar[int]
    AGENT_EXTINGUISHED_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    agent_ids: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    agent_extinguished_objects: _containers.MessageMap[int, ExtinguishedObjectInfoList]
    def __init__(self, agent_ids: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ..., agent_extinguished_objects: _Optional[_Mapping[int, ExtinguishedObjectInfoList]] = ...) -> None: ...

class NPCPosResponse(_message.Message):
    __slots__ = ("npc_id", "position")
    NPC_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    npc_id: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    position: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vector3f]
    def __init__(self, npc_id: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ..., position: _Optional[_Iterable[_Union[_common_pb2.Vector3f, _Mapping]]] = ...) -> None: ...
