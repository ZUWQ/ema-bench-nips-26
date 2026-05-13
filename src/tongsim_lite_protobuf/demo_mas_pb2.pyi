from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetAgentViewImageRequest(_message.Message):
    __slots__ = ("agent_id", "width", "height", "compress_jpg")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    COMPRESS_JPG_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    width: int
    height: int
    compress_jpg: bool
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., compress_jpg: bool = ...) -> None: ...

class GetAgentViewImageResponse(_message.Message):
    __slots__ = ("image_data", "width", "height")
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    width: int
    height: int
    def __init__(self, image_data: _Optional[bytes] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...

class GetPerceptionInfoRequest(_message.Message):
    __slots__ = ("agent_id", "radius_uu")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    RADIUS_UU_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    radius_uu: float
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., radius_uu: _Optional[float] = ...) -> None: ...

class PerceivedEntity(_message.Message):
    __slots__ = ("object_info", "distance_uu", "relative_pos")
    OBJECT_INFO_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_UU_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_POS_FIELD_NUMBER: _ClassVar[int]
    object_info: _object_pb2.ObjectInfo
    distance_uu: float
    relative_pos: _common_pb2.Vector3f
    def __init__(self, object_info: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ..., distance_uu: _Optional[float] = ..., relative_pos: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ...) -> None: ...

class GetPerceptionInfoResponse(_message.Message):
    __slots__ = ("entities",)
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[PerceivedEntity]
    def __init__(self, entities: _Optional[_Iterable[_Union[PerceivedEntity, _Mapping]]] = ...) -> None: ...

class DistressSignal(_message.Message):
    __slots__ = ("location", "intensity", "type")
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TYPE_FIRE: _ClassVar[DistressSignal.Type]
        TYPE_INJURED: _ClassVar[DistressSignal.Type]
        TYPE_COLLAPSED: _ClassVar[DistressSignal.Type]
    TYPE_FIRE: DistressSignal.Type
    TYPE_INJURED: DistressSignal.Type
    TYPE_COLLAPSED: DistressSignal.Type
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    INTENSITY_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    location: _common_pb2.Vector3f
    intensity: float
    type: DistressSignal.Type
    def __init__(self, location: _Optional[_Union[_common_pb2.Vector3f, _Mapping]] = ..., intensity: _Optional[float] = ..., type: _Optional[_Union[DistressSignal.Type, str]] = ...) -> None: ...

class GetDistressSignalsResponse(_message.Message):
    __slots__ = ("signals",)
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    signals: _containers.RepeatedCompositeFieldContainer[DistressSignal]
    def __init__(self, signals: _Optional[_Iterable[_Union[DistressSignal, _Mapping]]] = ...) -> None: ...

class SendDoorToggleCmdRequest(_message.Message):
    __slots__ = ("door_id", "open")
    DOOR_ID_FIELD_NUMBER: _ClassVar[int]
    OPEN_FIELD_NUMBER: _ClassVar[int]
    door_id: _object_pb2.ObjectId
    open: bool
    def __init__(self, door_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., open: bool = ...) -> None: ...

class DialogueUnit(_message.Message):
    __slots__ = ("speaker_name", "content", "timestamp_sec")
    SPEAKER_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_SEC_FIELD_NUMBER: _ClassVar[int]
    speaker_name: str
    content: str
    timestamp_sec: float
    def __init__(self, speaker_name: _Optional[str] = ..., content: _Optional[str] = ..., timestamp_sec: _Optional[float] = ...) -> None: ...

class SendDialogueCmdRequest(_message.Message):
    __slots__ = ("type", "dialogue", "target_ids")
    class DialogueType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        BROADCAST: _ClassVar[SendDialogueCmdRequest.DialogueType]
        WHISPER: _ClassVar[SendDialogueCmdRequest.DialogueType]
        TEAM: _ClassVar[SendDialogueCmdRequest.DialogueType]
    BROADCAST: SendDialogueCmdRequest.DialogueType
    WHISPER: SendDialogueCmdRequest.DialogueType
    TEAM: SendDialogueCmdRequest.DialogueType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DIALOGUE_FIELD_NUMBER: _ClassVar[int]
    TARGET_IDS_FIELD_NUMBER: _ClassVar[int]
    type: SendDialogueCmdRequest.DialogueType
    dialogue: DialogueUnit
    target_ids: _containers.RepeatedCompositeFieldContainer[_object_pb2.ObjectId]
    def __init__(self, type: _Optional[_Union[SendDialogueCmdRequest.DialogueType, str]] = ..., dialogue: _Optional[_Union[DialogueUnit, _Mapping]] = ..., target_ids: _Optional[_Iterable[_Union[_object_pb2.ObjectId, _Mapping]]] = ...) -> None: ...

class SendDialogueCmdResponse(_message.Message):
    __slots__ = ("delivered",)
    DELIVERED_FIELD_NUMBER: _ClassVar[int]
    delivered: bool
    def __init__(self, delivered: bool = ...) -> None: ...

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

class SendExtinguishFireCmdRequest(_message.Message):
    __slots__ = ("agent_id",)
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    agent_id: _object_pb2.ObjectId
    def __init__(self, agent_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...
