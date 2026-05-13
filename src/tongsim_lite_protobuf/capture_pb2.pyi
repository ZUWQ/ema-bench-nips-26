from tongsim_lite_protobuf import common_pb2 as _common_pb2
from tongsim_lite_protobuf import object_pb2 as _object_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CaptureColorSource(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COLOR_SOURCE_SCENE_COLOR_HDR: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_SCENE_COLOR_HDR_NO_ALPHA: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_FINAL_COLOR_LDR: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_SCENE_COLOR_SCENE_DEPTH: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_SCENE_DEPTH: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_DEVICE_DEPTH: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_NORMAL: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_BASE_COLOR: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_FINAL_COLOR_HDR: _ClassVar[CaptureColorSource]
    COLOR_SOURCE_FINAL_TONE_CURVE_HDR: _ClassVar[CaptureColorSource]

class CaptureRenderTargetFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COLOR_FORMAT_R8: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RG8: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RGBA8: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RGBA8_SRGB: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_R16F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RG16F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RGBA16F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_R32F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RG32F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RGBA32F: _ClassVar[CaptureRenderTargetFormat]
    COLOR_FORMAT_RGB10A2: _ClassVar[CaptureRenderTargetFormat]

class CaptureDepthMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAPTURE_DEPTH_NONE: _ClassVar[CaptureDepthMode]
    CAPTURE_DEPTH_LINEAR: _ClassVar[CaptureDepthMode]
    CAPTURE_DEPTH_DEVICE_Z: _ClassVar[CaptureDepthMode]
    CAPTURE_DEPTH_VIEW_SPACE_Z: _ClassVar[CaptureDepthMode]
    CAPTURE_DEPTH_NORMALIZED_01: _ClassVar[CaptureDepthMode]

class CaptureRgbCodec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAPTURE_RGB_CODEC_NONE: _ClassVar[CaptureRgbCodec]
    CAPTURE_RGB_CODEC_JPEG: _ClassVar[CaptureRgbCodec]

class CaptureDepthCodec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CAPTURE_DEPTH_CODEC_NONE: _ClassVar[CaptureDepthCodec]
    CAPTURE_DEPTH_CODEC_EXR: _ClassVar[CaptureDepthCodec]
COLOR_SOURCE_SCENE_COLOR_HDR: CaptureColorSource
COLOR_SOURCE_SCENE_COLOR_HDR_NO_ALPHA: CaptureColorSource
COLOR_SOURCE_FINAL_COLOR_LDR: CaptureColorSource
COLOR_SOURCE_SCENE_COLOR_SCENE_DEPTH: CaptureColorSource
COLOR_SOURCE_SCENE_DEPTH: CaptureColorSource
COLOR_SOURCE_DEVICE_DEPTH: CaptureColorSource
COLOR_SOURCE_NORMAL: CaptureColorSource
COLOR_SOURCE_BASE_COLOR: CaptureColorSource
COLOR_SOURCE_FINAL_COLOR_HDR: CaptureColorSource
COLOR_SOURCE_FINAL_TONE_CURVE_HDR: CaptureColorSource
COLOR_FORMAT_R8: CaptureRenderTargetFormat
COLOR_FORMAT_RG8: CaptureRenderTargetFormat
COLOR_FORMAT_RGBA8: CaptureRenderTargetFormat
COLOR_FORMAT_RGBA8_SRGB: CaptureRenderTargetFormat
COLOR_FORMAT_R16F: CaptureRenderTargetFormat
COLOR_FORMAT_RG16F: CaptureRenderTargetFormat
COLOR_FORMAT_RGBA16F: CaptureRenderTargetFormat
COLOR_FORMAT_R32F: CaptureRenderTargetFormat
COLOR_FORMAT_RG32F: CaptureRenderTargetFormat
COLOR_FORMAT_RGBA32F: CaptureRenderTargetFormat
COLOR_FORMAT_RGB10A2: CaptureRenderTargetFormat
CAPTURE_DEPTH_NONE: CaptureDepthMode
CAPTURE_DEPTH_LINEAR: CaptureDepthMode
CAPTURE_DEPTH_DEVICE_Z: CaptureDepthMode
CAPTURE_DEPTH_VIEW_SPACE_Z: CaptureDepthMode
CAPTURE_DEPTH_NORMALIZED_01: CaptureDepthMode
CAPTURE_RGB_CODEC_NONE: CaptureRgbCodec
CAPTURE_RGB_CODEC_JPEG: CaptureRgbCodec
CAPTURE_DEPTH_CODEC_NONE: CaptureDepthCodec
CAPTURE_DEPTH_CODEC_EXR: CaptureDepthCodec

class CaptureCameraParams(_message.Message):
    __slots__ = ("width", "height", "fov_degrees", "qps", "enable_depth", "color_source", "color_format", "enable_post_process", "enable_temporal_aa", "depth_near", "depth_far", "depth_mode", "rgb_codec", "depth_codec", "jpeg_quality")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FOV_DEGREES_FIELD_NUMBER: _ClassVar[int]
    QPS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    COLOR_SOURCE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FORMAT_FIELD_NUMBER: _ClassVar[int]
    ENABLE_POST_PROCESS_FIELD_NUMBER: _ClassVar[int]
    ENABLE_TEMPORAL_AA_FIELD_NUMBER: _ClassVar[int]
    DEPTH_NEAR_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FAR_FIELD_NUMBER: _ClassVar[int]
    DEPTH_MODE_FIELD_NUMBER: _ClassVar[int]
    RGB_CODEC_FIELD_NUMBER: _ClassVar[int]
    DEPTH_CODEC_FIELD_NUMBER: _ClassVar[int]
    JPEG_QUALITY_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    fov_degrees: float
    qps: float
    enable_depth: bool
    color_source: CaptureColorSource
    color_format: CaptureRenderTargetFormat
    enable_post_process: bool
    enable_temporal_aa: bool
    depth_near: float
    depth_far: float
    depth_mode: CaptureDepthMode
    rgb_codec: CaptureRgbCodec
    depth_codec: CaptureDepthCodec
    jpeg_quality: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., fov_degrees: _Optional[float] = ..., qps: _Optional[float] = ..., enable_depth: bool = ..., color_source: _Optional[_Union[CaptureColorSource, str]] = ..., color_format: _Optional[_Union[CaptureRenderTargetFormat, str]] = ..., enable_post_process: bool = ..., enable_temporal_aa: bool = ..., depth_near: _Optional[float] = ..., depth_far: _Optional[float] = ..., depth_mode: _Optional[_Union[CaptureDepthMode, str]] = ..., rgb_codec: _Optional[_Union[CaptureRgbCodec, str]] = ..., depth_codec: _Optional[_Union[CaptureDepthCodec, str]] = ..., jpeg_quality: _Optional[int] = ...) -> None: ...

class CaptureCameraStatus(_message.Message):
    __slots__ = ("capturing", "queue_count", "compressed_queue_count", "width", "height", "fov_degrees", "depth_mode")
    CAPTURING_FIELD_NUMBER: _ClassVar[int]
    QUEUE_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPRESSED_QUEUE_COUNT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FOV_DEGREES_FIELD_NUMBER: _ClassVar[int]
    DEPTH_MODE_FIELD_NUMBER: _ClassVar[int]
    capturing: bool
    queue_count: int
    compressed_queue_count: int
    width: int
    height: int
    fov_degrees: float
    depth_mode: CaptureDepthMode
    def __init__(self, capturing: bool = ..., queue_count: _Optional[int] = ..., compressed_queue_count: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., fov_degrees: _Optional[float] = ..., depth_mode: _Optional[_Union[CaptureDepthMode, str]] = ...) -> None: ...

class CameraIntrinsics(_message.Message):
    __slots__ = ("fx", "fy", "cx", "cy")
    FX_FIELD_NUMBER: _ClassVar[int]
    FY_FIELD_NUMBER: _ClassVar[int]
    CX_FIELD_NUMBER: _ClassVar[int]
    CY_FIELD_NUMBER: _ClassVar[int]
    fx: float
    fy: float
    cx: float
    cy: float
    def __init__(self, fx: _Optional[float] = ..., fy: _Optional[float] = ..., cx: _Optional[float] = ..., cy: _Optional[float] = ...) -> None: ...

class CaptureFrame(_message.Message):
    __slots__ = ("camera_id", "frame_id", "game_time_seconds", "gpu_ready_timestamp", "width", "height", "world_pose", "intrinsics", "rgba8", "depth_r32", "depth_near", "depth_far", "depth_mode", "has_color", "has_depth")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    GAME_TIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    GPU_READY_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WORLD_POSE_FIELD_NUMBER: _ClassVar[int]
    INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    RGBA8_FIELD_NUMBER: _ClassVar[int]
    DEPTH_R32_FIELD_NUMBER: _ClassVar[int]
    DEPTH_NEAR_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FAR_FIELD_NUMBER: _ClassVar[int]
    DEPTH_MODE_FIELD_NUMBER: _ClassVar[int]
    HAS_COLOR_FIELD_NUMBER: _ClassVar[int]
    HAS_DEPTH_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    frame_id: int
    game_time_seconds: float
    gpu_ready_timestamp: float
    width: int
    height: int
    world_pose: _common_pb2.Transform
    intrinsics: CameraIntrinsics
    rgba8: bytes
    depth_r32: bytes
    depth_near: float
    depth_far: float
    depth_mode: CaptureDepthMode
    has_color: bool
    has_depth: bool
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., frame_id: _Optional[int] = ..., game_time_seconds: _Optional[float] = ..., gpu_ready_timestamp: _Optional[float] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., world_pose: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., intrinsics: _Optional[_Union[CameraIntrinsics, _Mapping]] = ..., rgba8: _Optional[bytes] = ..., depth_r32: _Optional[bytes] = ..., depth_near: _Optional[float] = ..., depth_far: _Optional[float] = ..., depth_mode: _Optional[_Union[CaptureDepthMode, str]] = ..., has_color: bool = ..., has_depth: bool = ...) -> None: ...

class CaptureCameraDescriptor(_message.Message):
    __slots__ = ("camera", "params", "status")
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    camera: _object_pb2.ObjectInfo
    params: CaptureCameraParams
    status: CaptureCameraStatus
    def __init__(self, camera: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ..., params: _Optional[_Union[CaptureCameraParams, _Mapping]] = ..., status: _Optional[_Union[CaptureCameraStatus, _Mapping]] = ...) -> None: ...

class ListCaptureCamerasRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListCaptureCamerasResponse(_message.Message):
    __slots__ = ("cameras",)
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    cameras: _containers.RepeatedCompositeFieldContainer[CaptureCameraDescriptor]
    def __init__(self, cameras: _Optional[_Iterable[_Union[CaptureCameraDescriptor, _Mapping]]] = ...) -> None: ...

class CreateCaptureCameraRequest(_message.Message):
    __slots__ = ("capture_name", "world_transform", "params", "attach_parent", "attach_socket", "keep_world")
    CAPTURE_NAME_FIELD_NUMBER: _ClassVar[int]
    WORLD_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    ATTACH_PARENT_FIELD_NUMBER: _ClassVar[int]
    ATTACH_SOCKET_FIELD_NUMBER: _ClassVar[int]
    KEEP_WORLD_FIELD_NUMBER: _ClassVar[int]
    capture_name: str
    world_transform: _common_pb2.Transform
    params: CaptureCameraParams
    attach_parent: _object_pb2.ObjectId
    attach_socket: str
    keep_world: bool
    def __init__(self, capture_name: _Optional[str] = ..., world_transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ..., params: _Optional[_Union[CaptureCameraParams, _Mapping]] = ..., attach_parent: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., attach_socket: _Optional[str] = ..., keep_world: bool = ...) -> None: ...

class CreateCaptureCameraResponse(_message.Message):
    __slots__ = ("camera",)
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    camera: _object_pb2.ObjectInfo
    def __init__(self, camera: _Optional[_Union[_object_pb2.ObjectInfo, _Mapping]] = ...) -> None: ...

class DestroyCaptureCameraRequest(_message.Message):
    __slots__ = ("camera_id", "force_stop_capture")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_STOP_CAPTURE_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    force_stop_capture: bool
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., force_stop_capture: bool = ...) -> None: ...

class SetCaptureCameraPoseRequest(_message.Message):
    __slots__ = ("camera_id", "world_transform")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    WORLD_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    world_transform: _common_pb2.Transform
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., world_transform: _Optional[_Union[_common_pb2.Transform, _Mapping]] = ...) -> None: ...

class UpdateCaptureCameraParamsRequest(_message.Message):
    __slots__ = ("camera_id", "params")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    params: CaptureCameraParams
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., params: _Optional[_Union[CaptureCameraParams, _Mapping]] = ...) -> None: ...

class UpdateCaptureCameraParamsResponse(_message.Message):
    __slots__ = ("applied_params",)
    APPLIED_PARAMS_FIELD_NUMBER: _ClassVar[int]
    applied_params: CaptureCameraParams
    def __init__(self, applied_params: _Optional[_Union[CaptureCameraParams, _Mapping]] = ...) -> None: ...

class AttachCaptureCameraRequest(_message.Message):
    __slots__ = ("camera_id", "parent_actor_id", "socket_name", "keep_world")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ACTOR_ID_FIELD_NUMBER: _ClassVar[int]
    SOCKET_NAME_FIELD_NUMBER: _ClassVar[int]
    KEEP_WORLD_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    parent_actor_id: _object_pb2.ObjectId
    socket_name: str
    keep_world: bool
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., parent_actor_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., socket_name: _Optional[str] = ..., keep_world: bool = ...) -> None: ...

class CaptureSnapshotRequest(_message.Message):
    __slots__ = ("camera_id", "timeout_seconds", "include_color", "include_depth")
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_COLOR_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    timeout_seconds: float
    include_color: bool
    include_depth: bool
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ..., timeout_seconds: _Optional[float] = ..., include_color: bool = ..., include_depth: bool = ...) -> None: ...

class GetCaptureStatusRequest(_message.Message):
    __slots__ = ("camera_id",)
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    camera_id: _object_pb2.ObjectId
    def __init__(self, camera_id: _Optional[_Union[_object_pb2.ObjectId, _Mapping]] = ...) -> None: ...

class GetCaptureStatusResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: CaptureCameraStatus
    def __init__(self, status: _Optional[_Union[CaptureCameraStatus, _Mapping]] = ...) -> None: ...
