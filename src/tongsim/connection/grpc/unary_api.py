from tongsim.math import Transform, Vector3
from tongsim.type.rl_demo import RLDemoOrientationMode
from tongsim_lite_protobuf.arena_pb2 import (
    DestroyArenaRequest,
    GetActorPoseLocalRequest,
    GetActorPoseLocalResponse,
    ListArenasRequest,
    ListArenasResponse,
    LoadArenaRequest,
    LoadArenaResponse,
    LocalToWorldRequest,
    LocalToWorldResponse,
    ResetArenaRequest,
    SetActorPoseLocalRequest,
    SetArenaVisibleRequest,
    SpawnActorInArenaRequest,
    SpawnActorInArenaResponse,
    WorldToLocalRequest,
    WorldToLocalResponse,
    DestroyActorInArenaRequest,
    SimpleMoveTowardsInArenaRequest,
    SimpleMoveTowardsInArenaResponse,
)
from tongsim_lite_protobuf.arena_pb2_grpc import ArenaServiceStub
from tongsim_lite_protobuf.common_pb2 import Empty
from tongsim_lite_protobuf.demo_rl_pb2 import (
    ActorState,
    DemoRLState,
    ExecConsoleCommandRequest,
    ExecConsoleCommandResponse,
    GetActorStateRequest,
    GetActorStateResponse,
    GetActorTransformRequest,
    GetActorTransformResponse,
    QueryNavigationPathRequest,
    QueryNavigationPathResponse,
    SetActorTransformRequest,
    SimpleMoveTowardsRequest,
    SimpleMoveTowardsResponse,
    SpawnActorRequest,
    SpawnActorResponse,
    DestroyActorRequest,
    BatchSingleLineTraceByObjectRequest,
    LineTraceByObjectJob,                   # 若你的生成名是 SingleLineTraceByObjectJob，请按生成结果替换
    CollisionObjectType,
)
from tongsim_lite_protobuf.demo_rl_pb2_grpc import DemoRLServiceStub
from tongsim_lite_protobuf.embodied_mas_pb2 import (
    DestroyedObjects,
    PerceptionInfoRequest,
    PerceptionInfoResponse,
    ExtinguishFireRequest,
    ExtinguishFireResponse,
    ExtinguishFireResult,
    NPCSOSInfo,
    NPCSOSRequest,
    NPCSOSResponse,
    NPCInstructionRequest,
    NPCInstructionType,
    NPCInstructionResponse,
    ExtinguisherCfg,
    SetExtinguisherResponse,
    ExtinguisherRotation,
    PauseSceneRequest,
    BurnedStateResponse,
    ObjHealthResponse,
    NPCHealthResponse,
    OutFireResultResponse,
    ExtinguishedObjectInfoList,
    GetAgentExtinguishedObjectsResponse,
    NPCPosResponse,
    SelfStateRequest,
    SelfStateResponse,
    health_state_type,
    MoveTowardsByNavRequest,
    MoveTowardsByNavResponse,
    QueryNavDistanceRequest,
    QueryNavDistanceResponse,
)
from tongsim_lite_protobuf.embodied_mas_pb2_grpc import EmbodiedMASStub
from tongsim_lite_protobuf.object_pb2 import ObjectId
from tongsim_lite_protobuf.voxel_pb2 import QueryVoxelRequest, Voxel
from tongsim_lite_protobuf.voxel_pb2_grpc import VoxelServiceStub

from .core import GrpcConnection
from .utils import proto_to_sdk, safe_async_rpc, sdk_to_proto

# --------------------------
# GUID helpers (UE FGuid LE)
# --------------------------


def _fguid_bytes_to_str(guid_bytes: bytes) -> str:
    """
    Convert Unreal FGuid (16 bytes; first 3 fields little-endian) to canonical
    GUID string: 8-4-4-4-12 uppercase hex.

    Layout (Windows/MS GUID):
      Data1[4] LE, Data2[2] LE, Data3[2] LE, Data4[8] BE(as-is)
    """
    if not guid_bytes:
        return ""
    if len(guid_bytes) != 16:
        return guid_bytes.hex().upper()

    d1 = guid_bytes[0:4][::-1]  # LE -> BE
    d2 = guid_bytes[4:6][::-1]
    d3 = guid_bytes[6:8][::-1]
    d4 = guid_bytes[8:10]  # as-is
    d5 = guid_bytes[10:16]  # as-is
    return f"{d1.hex()}-{d2.hex()}-{d3.hex()}-{d4.hex()}-{d5.hex()}".upper()


def _guid_str_to_fguid_bytes(guid_str: str) -> bytes:
    """
    Convert canonical GUID string (8-4-4-4-12) to Unreal FGuid bytes
    with first 3 fields little-endian.
    """
    if not guid_str:
        return b""
    s = guid_str.replace("-", "").strip()
    if len(s) != 32:
        # Try raw hex fallback
        try:
            b = bytes.fromhex(s)
            return b if len(b) == 16 else b""
        except Exception:
            return b""

    try:
        raw = bytes.fromhex(s)
        # raw = [Data1(4) | Data2(2) | Data3(2) | Data4(8)] all BE
        d1 = raw[0:4][::-1]  # -> LE
        d2 = raw[4:6][::-1]
        d3 = raw[6:8][::-1]
        d4 = raw[8:16]  # as-is
        return d1 + d2 + d3 + d4
    except Exception:
        return b""


def _to_object_id(actor_id: bytes | str | dict) -> ObjectId:
    """
    Build ObjectId from:
      - bytes (len==16, UE FGuid layout), or
      - canonical GUID str "XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX", or
      - dict {"guid": <bytes|str>}
    """
    if isinstance(actor_id, dict):
        actor_id = actor_id.get("guid", b"")

    oid = ObjectId()
    if isinstance(actor_id, bytes | bytearray):
        guid_bytes = bytes(actor_id)
    elif isinstance(actor_id, str):
        guid_bytes = _guid_str_to_fguid_bytes(actor_id)
    else:
        guid_bytes = b""

    if not guid_bytes or len(guid_bytes) != 16:
        raise ValueError("actor_id must be 16-byte FGuid or canonical GUID string.")

    oid.guid = guid_bytes
    return oid


def _actor_state_to_dict(actor: ActorState) -> dict:
    """将 proto 的 ActorState 转为 SDK 字典结构。"""
    return {
        "id": _fguid_bytes_to_str(actor.object_info.id.guid),
        "name": actor.object_info.name,
        "class_path": actor.object_info.class_path,
        "location": proto_to_sdk(actor.location),
        "unit_forward_vector": proto_to_sdk(actor.unit_forward_vector),
        "unit_right_vector": proto_to_sdk(actor.unit_right_vector),
        "bounding_box": {
            "min": proto_to_sdk(actor.bounding_box.min_vertex),
            "max": proto_to_sdk(actor.bounding_box.max_vertex),
        },
        "tag": actor.tag,
        "destroyed": bool(getattr(actor, "destroyed", False)),
        "current_speed": float(getattr(actor, "current_speed", 0.0)),
    }


# --------------------------
# Public gRPC unary wrappers
# --------------------------


class UnaryAPI:
    """
    封装 DemoRLService 的 unary_unary 接口为 Python 方法。
    """

    @staticmethod
    @safe_async_rpc(default=[])
    async def query_info(conn: GrpcConnection) -> list[dict]:
        """
        获取当前 Demo RL 场景中所有 Actor 状态信息。

        Returns:
            list[dict]: 每个元素包含：
                {
                  "id": <GUID str>,
                  "name": str,
                  "class_path": str,
                  "location": {...},
                  "unit_forward_vector": {...},
                  "unit_right_vector": {...},
                  "bounding_box": {"min": {...}, "max": {...}},
                  "tag": str,
                }
        """
        stub = conn.get_stub(DemoRLServiceStub)
        resp: DemoRLState = await stub.QueryState(Empty(), timeout=10.0)

        result: list[dict] = []
        for actor in resp.actor_states:
            result.append(_actor_state_to_dict(actor))
        return result
    
    @staticmethod
    @safe_async_rpc(default=None)
    async def refresh_actors_map(conn: GrpcConnection):
        stub = conn.get_stub(EmbodiedMASStub)
        await stub.RefreshActorMappings(Empty(), timeout=2.0)
    
    @staticmethod
    @safe_async_rpc(default=None)
    async def start_to_burn(conn: GrpcConnection) -> bool:
        stub = conn.get_stub(EmbodiedMASStub)
        await stub.SetToBurn(Empty(), timeout=2.0)
        return True
     
    @staticmethod
    @safe_async_rpc(default=None)
    async def pause_scene(conn: GrpcConnection, pause: bool):
        stub = conn.get_stub(EmbodiedMASStub)
        req = PauseSceneRequest(bPause=pause)
        await stub.PauseScene(req, timeout=2.0)

    @staticmethod
    @safe_async_rpc(default=[])
    async def sendfollow(conn: GrpcConnection, actor_id: str) -> list[str]:
        
        stub = conn.get_stub(EmbodiedMASStub)
        req = NPCInstructionRequest(agent_id=_to_object_id(actor_id), instruction=NPCInstructionType.Follow)
        resp: NPCInstructionResponse = await stub.SendNPCInstrucions(req, timeout=2.0)
        return resp.result
    
    @staticmethod
    @safe_async_rpc(default=[])
    async def sendstopfollow(conn: GrpcConnection, actor_id: str) -> list[str]:
        
        stub = conn.get_stub(EmbodiedMASStub)
        req = NPCInstructionRequest(agent_id=_to_object_id(actor_id), instruction=NPCInstructionType.Stop)
        resp: NPCInstructionResponse = await stub.SendNPCInstrucions(req, timeout=2.0)
        return resp.result
    
    @staticmethod
    @safe_async_rpc(default="Fail_CanotExtinguish")
    async def extinguish_fire(
        conn: GrpcConnection,
        agent_id: bytes | str | dict,
        *,
        timeout: float = 5.0,
    ) -> str:
        """
        调用 EmbodiedMAS.ExtinguishFire，让指定 Agent 执行灭火动作。

        返回:
            str: 结果字符串，便于上层直接比较
                "Success" | "Fail_CanotExtinguish" | "Fail_NeedSupply"
        """

        stub = conn.get_stub(EmbodiedMASStub)
        req = ExtinguishFireRequest(agent_id=_to_object_id(agent_id))

        resp :ExtinguishFireResponse= await stub.ExtinguishFire(req, timeout=timeout)

        # 把枚举值转字符串，方便 Python 端直接比较
        return ExtinguishFireResult.Name(resp.result)
    
    @staticmethod
    @safe_async_rpc(default=None)
    async def set_extinguisher(conn: GrpcConnection,agent_id: bytes | str | dict,water_capacity:int,recover_time:int,timeout:float=5.0) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        req = ExtinguisherCfg(agent_id=_to_object_id(agent_id), water_capacity=water_capacity, recover_time=recover_time)
        resp: SetExtinguisherResponse = await stub.SetExtinguisher(req, timeout=timeout)
        return bool(resp.result)

    @staticmethod
    @safe_async_rpc(default=False)
    async def set_extinguisher_rotation(
        conn: GrpcConnection,
        agent_id: bytes | str | dict,
        yaw: float,
        pitch: float,
        timeout: float = 2.0,
    ) -> bool:
        """
        调用 EmbodiedMAS.SetExtinguisherRotation，设置灭火器的俯仰/偏航角。
        """
        stub = conn.get_stub(EmbodiedMASStub)
        req = ExtinguisherRotation(
            agent_id=_to_object_id(agent_id),
            yaw=float(yaw),
            pitch=float(pitch),
        )
        await stub.SetExtinguisherRotation(req, timeout=timeout)
        return True
 
    @staticmethod
    @safe_async_rpc(default=None)
    async def navigate_to_location(
        conn: GrpcConnection,
        actor_id: bytes | str | dict,
        target_location: Vector3,
        accept_radius: float,
        allow_partial: bool = True,
        speed_uu_per_sec: float | None = None,
        timeout: float = 3600.0,
    ) -> dict | None:
        """
        Navigate a Character to a location using UE NavMesh (server-side async Reactor).

        Returns:
            dict: ``success``, ``message``, ``final_location`` and ``is_partial``.
        """
        stub = conn.get_stub(EmbodiedMASStub)
        req = MoveTowardsByNavRequest(
            actor_id=_to_object_id(actor_id),
            target_location=sdk_to_proto(target_location),
            accept_radius=float(accept_radius),
            allow_partial=bool(allow_partial),
        )
        if speed_uu_per_sec is not None:
            req.speed_uu_per_sec = float(speed_uu_per_sec)
            resp: MoveTowardsByNavResponse = await stub.MoveTowardsByNav(
            req, timeout=timeout
        )
        return {
            "success": bool(resp.success),
            "message": str(resp.message),
            "final_location": proto_to_sdk(resp.current_location),
            "is_partial": bool(resp.is_partial),
        }

    @staticmethod
    @safe_async_rpc(default=0.0)
    async def query_nav_distance(
        conn: GrpcConnection,
        agent_id: bytes | str | dict,
        target_id: bytes | str | dict,
        allow_partial: bool = True,
        timeout: float = 2.0,
    ) -> float:
        """
        调用 EmbodiedMAS.QueryNavDistance，返回两个对象间的可导航距离（UU）。
        """
        stub = conn.get_stub(EmbodiedMASStub)
        req = QueryNavDistanceRequest(
            agent_id=_to_object_id(agent_id),
            target_id=_to_object_id(target_id),
            allow_partial=bool(allow_partial),
        )
        resp: QueryNavDistanceResponse = await stub.QueryNavDistance(req, timeout=timeout)
        return float(resp.distance)

    @staticmethod
    @safe_async_rpc(default={"actor_info": [], "npc_info": []})
    async def get_embodied_perception(
        conn: GrpcConnection,
        agent_id: bytes | str | dict,
        timeout: float = 5.0,
    ) -> dict:
        """
        调用 EmbodiedMAS.GetPerceptionInfo，返回结构化 Python 数据。
        """
        stub = conn.get_stub(EmbodiedMASStub)
        req = PerceptionInfoRequest(agent_id=_to_object_id(agent_id))
        resp: PerceptionInfoResponse = await stub.GetPerceptionInfo(req, timeout=timeout)

        # 1. ActorInfo 列表
        actor_list = [
            {
                "actor": _actor_state_to_dict(a.actor),          
                "burning_state": a.burning_state,
                "burned_percentage": a.burned_percentage,
                "mass": a.mass,
                "movable": a.movable,
                "burning_speed": a.burning_speed,
            }
            for a in resp.actor_info
        ]

        # 2. NPCInfo 列表
        npc_list = [
            {
                "object_info": {
                    "id": _fguid_bytes_to_str(n.object_info.id.guid),
                    "name": n.object_info.name,
                    "class_path": n.object_info.class_path,
                },
                "position": proto_to_sdk(n.position),
                "health": n.health,
                "burning_state": n.burning_state,
                "mass": n.mass,
            }
            for n in resp.npc_info
        ]

        return {"actor_info": actor_list, "npc_info": npc_list}

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_selfstate(conn: GrpcConnection,
        actor_id: bytes | str | dict,
         *,
        timeout: float = 5.0,
    ) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        req = SelfStateRequest(agent_id=_to_object_id(actor_id))
        resp: SelfStateResponse = await stub.GetSelfState(req, timeout=timeout)
        hp_list = {}
        for i in range(len(resp.following_NPC_ids)):
            hp_list[_fguid_bytes_to_str(resp.following_NPC_ids[i].guid)] = health_state_type.Name(resp.health_state[i])
        return {
            "actor_info": {
                    "id": _fguid_bytes_to_str(resp.actor_info.id.guid),
                    "name": resp.actor_info.name,
                    "class_path": resp.actor_info.class_path,
                },
            "location": proto_to_sdk(resp.location),
            "unit_forward_vector": proto_to_sdk(resp.unit_forward_vector),
            "unit_right_vector": proto_to_sdk(resp.unit_right_vector),
            "tags": resp.tag,
            "rest_usage": resp.rest_usage,
            "used_usage": resp.used_usage,
            "following_NPC": hp_list,
            "health":resp.health_value,
        }

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_burned_area(conn: GrpcConnection) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        resp: BurnedStateResponse = await stub.GetBurnedArea(Empty(), timeout=2.0)
        return {"unburned_num":resp.unburned_obj_num, 
                "burning_num":resp.burning_obj_num, 
                "extinguished_num":resp.extinguished_obj_num, 
                "burned_obj_num":resp.be_watered_obj_num, 
                "total_num":resp.obj_total_num
            }

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_obj_residual(conn: GrpcConnection) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        resp: ObjHealthResponse = await stub.GetObjectHealth(Empty(), timeout=2.0)
        return resp.obj_sum_hp

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_npc_health(conn: GrpcConnection) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        resp: NPCHealthResponse = await stub.GetNPCHealth(Empty(), timeout=2.0)
        hp_list = {}
        for i in range(len(resp.npc_hp)):
            hp_list[_fguid_bytes_to_str(resp.npc_id[i].guid)] = resp.npc_hp[i]
        return hp_list

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_outfire_state(conn: GrpcConnection) -> bool:

        stub = conn.get_stub(EmbodiedMASStub)
        resp: OutFireResultResponse = await stub.GetOutFireResult(Empty(), timeout=2.0)
        return bool(resp.result)

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_npc_postions(conn: GrpcConnection) -> dict:
        stub = conn.get_stub(EmbodiedMASStub)
        resp: NPCPosResponse = await stub.GetNPCPos(Empty(), timeout=2.0)
        pos_list = {}
        for i in range(len(resp.position)):
            pos_list[_fguid_bytes_to_str(resp.npc_id[i].guid)] = resp.position[i]
        return pos_list
 
    @staticmethod
    @safe_async_rpc(default=None)
    async def get_destroyed_objects(conn: GrpcConnection) -> list:
        stub = conn.get_stub(EmbodiedMASStub)
        resp: DestroyedObjects = await stub.GetDestroyedObjects(Empty(), timeout=2.0)
        destroyed_objects = []
        for i in range(len(resp.objects_name)):
            destroyed_objects.append(resp.objects_name[i])
        return destroyed_objects

    @staticmethod   
    @safe_async_rpc(default={})
    async def get_agent_extinguished_objects(
        conn: GrpcConnection,
        timeout: float = 2.0,
    ) -> dict:
        """
        调用 EmbodiedMAS.GetAgentExtinguishedObjects，返回每个 Agent 已扑灭目标列表。

        Returns:
            dict[str, list[dict]]: {agent_guid: [{"id","name","class_path"}, ...], ...}
        """
        stub = conn.get_stub(EmbodiedMASStub)
        resp: GetAgentExtinguishedObjectsResponse = await stub.GetAgentExtinguishedObjects(
            Empty(), timeout=timeout
        )

        result: dict[str, list[dict]] = {}

        # proto3 map<int32, ExtinguishedObjectInfoList> 映射下标 -> 对应对象列表
        for idx, agent_oid in enumerate(resp.agent_ids):
            guid = _fguid_bytes_to_str(agent_oid.guid)
            info_list: ExtinguishedObjectInfoList | None = resp.agent_extinguished_objects.get(
                idx
            )
            if info_list is None:
                result[guid] = []
                continue

            objs: list[dict] = []
            for oi in info_list.objects:
                objs.append(
                    {
                        "id": _fguid_bytes_to_str(oi.id.guid),
                        "name": oi.name,
                        "class_path": oi.class_path,
                    }
                )
            result[guid] = objs

        return result

    @staticmethod
    async def receive_npc_sos(conn: GrpcConnection, actor_ids: list[str]) :
        stub = conn.get_stub(EmbodiedMASStub)

        req = NPCSOSRequest(
            agent_id=[_to_object_id(aid) for aid in actor_ids]
        )

        call = stub.NPCSOSInfo(req, timeout=100.0)

        async for resp in call:
            for npc_sos_info in resp.npc_sos_info:
                yield {
                    "agent_id": npc_sos_info.agent_id.guid,
                    "orientations": [
                        {
                            "roll": ori.roll_deg,
                            "pitch": ori.pitch_deg,
                            "yaw": ori.yaw_deg,
                        }
                        for ori in npc_sos_info.orientation
                    ],
                    "distances": list(npc_sos_info.distance),
                }
   
    @staticmethod
    @safe_async_rpc(default=False)
    async def reset_level(conn: GrpcConnection, timeout: float = 60.0) -> bool:
        """
        重置当前关卡。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        await stub.ResetLevel(Empty(), timeout=timeout)
        return True

    @staticmethod
    @safe_async_rpc(default=(None, None))
    async def simple_move_towards(
        conn: GrpcConnection,
        target_location: Vector3,
        actor_id: bytes | str | dict,                      
        orientation_mode: RLDemoOrientationMode = RLDemoOrientationMode.ORIENTATION_KEEP_CURRENT,
        given_forward: Vector3 | None = None,
        timeout: float = 3600.0,
        speed_uu_per_sec: float = 12000.0,
        tolerance_uu: float = 5.0,
    ) -> tuple[dict | None, dict | None]:
        """
        """
        req = SimpleMoveTowardsRequest(
            actor_id=_to_object_id(actor_id), 
            target_location=sdk_to_proto(target_location),
            orientation_mode=orientation_mode,
            speed_uu_per_sec = float(speed_uu_per_sec),
            tolerance_uu = float(tolerance_uu)
        )

        if (
            orientation_mode == RLDemoOrientationMode.ORIENTATION_GIVEN
            and given_forward is not None
        ):
            req.given_orientation.CopyFrom(sdk_to_proto(given_forward))

        stub = conn.get_stub(DemoRLServiceStub)
        resp: SimpleMoveTowardsResponse = await stub.SimpleMoveTowards(
            req, timeout=timeout
        )

        current_location = proto_to_sdk(resp.current_location)

        hit_result = None
        if resp.HasField("hit_result"):
            hit_result = {"hit_actor": (resp.hit_result.hit_actor)}

        return current_location, hit_result

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_actor_state(conn: GrpcConnection, actor_id: str) -> dict | None:
        """
        通过 ObjectId 获取 ActorState。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = GetActorStateRequest(actor_id=_to_object_id(actor_id))
        resp: GetActorStateResponse = await stub.GetActorState(req, timeout=2.0)
        return _actor_state_to_dict(resp.actor_state)

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_actor_transform(conn: GrpcConnection, actor_id: str) -> Transform:
        """
        通过 ObjectId 获取 Transform。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = GetActorTransformRequest(actor_id=_to_object_id(actor_id))
        resp: GetActorTransformResponse = await stub.GetActorTransform(req, timeout=2.0)
        return proto_to_sdk(resp.transform)

    @staticmethod
    @safe_async_rpc(default=False)
    async def set_actor_transform(
        conn: GrpcConnection, actor_id: bytes | str | dict, transform: Transform
    ) -> bool:
        """
        设置 Actor 的 Transform（TeleportPhysics）。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = SetActorTransformRequest(
            actor_id=_to_object_id(actor_id),
            transform=sdk_to_proto(transform),
        )
        await stub.SetActorTransform(req, timeout=2.0)
        return True

    @staticmethod
    @safe_async_rpc(default=None)
    async def spawn_actor(
        conn: GrpcConnection,
        blueprint: str,
        transform: Transform,
        name: str | None = None,
        tags: list[str] | None = None,
        timeout: float = 5.0,
    ) -> dict | None:
        """
        生成 Actor（若无 RL_Agent 标签，会在服务器端默认附加）。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = SpawnActorRequest(
            blueprint=blueprint,
            transform=sdk_to_proto(transform),
        )
        if name:
            req.name = name
        if tags:
            req.tags.extend(tags)

        resp: SpawnActorResponse = await stub.SpawnActor(req, timeout=timeout)
        ai = resp.actor
        return {
            "id": _fguid_bytes_to_str(ai.id.guid),
            "name": ai.name,
            "class_path": ai.class_path,
        }

    @staticmethod
    @safe_async_rpc(default=None)
    async def query_voxel(
        conn: GrpcConnection,
        transform: Transform,
        voxel_num_x: int,
        voxel_num_y: int,
        voxel_num_z: int,
        box_extent: Vector3,
        actors_to_ignore: list[str] | None = None,
        timeout: float = 5.0,
    ) -> bytes:
        """
        TODO:
        """
        if actors_to_ignore is None:
            actors_to_ignore = []

        stub = conn.get_stub(VoxelServiceStub)
        req = QueryVoxelRequest(
            transform=sdk_to_proto(transform),
            voxel_num_x=voxel_num_x,
            voxel_num_y=voxel_num_y,
            voxel_num_z=voxel_num_z,
            extent=sdk_to_proto(box_extent),
            ActorsToIgnore=[_to_object_id(actor_id) for actor_id in actors_to_ignore],
        )
        resp: Voxel = await stub.QueryVoxel(req, timeout=2.0)
        return resp.voxel_buffer


    @staticmethod
    @safe_async_rpc(default=False)
    async def exec_console_command(
        conn: GrpcConnection,
        command: str,
        write_to_log: bool = True,
        timeout: float = 2.0,
    ) -> bool:
        """
        执行 UE 控制台命令（如 'stat fps'、'r.Streaming.PoolSize 4000' 等）。

        Returns:
            bool: 是否提交成功（无法返回控制台输出文本）。
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = ExecConsoleCommandRequest(command=command, write_to_log=write_to_log)
        resp: ExecConsoleCommandResponse = await stub.ExecConsoleCommand(req, timeout=timeout)
        return bool(resp.success)


    @staticmethod
    @safe_async_rpc(default=None)
    async def query_navigation_path(
        conn: GrpcConnection,
        start: Vector3,
        end: Vector3,
        allow_partial: bool = True,
        require_navigable_end_location: bool = False,
        cost_limit: float | None = None,
        timeout: float = 2.0,
    ) -> dict | None:
        """
        查询从 start 到 end 的导航路径。

        TODO:

        Returns:
            dict: {
              "points": [Vector3, ...],  # 路径点
              "is_partial": bool,
              "path_cost": float,
              "path_length": float,
            }
        """
        stub = conn.get_stub(DemoRLServiceStub)
        req = QueryNavigationPathRequest(
            start=sdk_to_proto(start),
            end=sdk_to_proto(end),
            allow_partial=allow_partial,
            require_navigable_end_location=require_navigable_end_location,
        )
        if cost_limit is not None and cost_limit > 0:
            req.cost_limit = float(cost_limit)

        resp: QueryNavigationPathResponse = await stub.QueryNavigationPath(req, timeout=timeout)
        return {
            "points": [proto_to_sdk(p) for p in resp.path_points],
            "is_partial": bool(resp.is_partial),
            "path_cost": float(resp.path_cost) if hasattr(resp, "path_cost") else 0.0,
            "path_length": float(resp.path_length) if hasattr(resp, "path_length") else 0.0,
        }

    # =========================
    # ArenaService (multi-level)
    # =========================

    @staticmethod
    @safe_async_rpc(default="")
    async def load_arena(
        conn: GrpcConnection,
        level_asset_path: str,
        anchor: Transform,
        make_visible: bool = True,
    ) -> str:
        """
        动态加载一个 Level（Arena 实例），并返回 arena_id（GUID 字符串）。
        """
        stub = conn.get_stub(ArenaServiceStub)
        req = LoadArenaRequest(
            level_asset_path=level_asset_path,
            anchor=sdk_to_proto(anchor),
            make_visible=make_visible,
        )
        resp: LoadArenaResponse = await stub.LoadArena(req, timeout=10.0)
        # arena_id.id.guid: bytes(16, UE FGuid LE)
        return _fguid_bytes_to_str(resp.arena_id.guid)

    @staticmethod
    @safe_async_rpc(default=False)
    async def destroy_arena(conn: GrpcConnection, arena_id: str) -> bool:
        """
        销毁指定 Arena 实例。
        """
        stub = conn.get_stub(ArenaServiceStub)
        await stub.DestroyArena(DestroyArenaRequest(arena_id=_to_object_id(arena_id)), timeout=5.0)
        return True

    @staticmethod
    @safe_async_rpc(default=False)
    async def reset_arena(conn: GrpcConnection, arena_id: str) -> bool:
        """
        重置指定 Arena
        """
        stub = conn.get_stub(ArenaServiceStub)
        await stub.ResetArena(ResetArenaRequest(arena_id=_to_object_id(arena_id)), timeout=30.0)
        return True

    @staticmethod
    @safe_async_rpc(default=False)
    async def set_arena_visible(conn: GrpcConnection, arena_id: str, visible: bool) -> bool:
        """
        设置 Arena 可见性。
        """
        stub = conn.get_stub(ArenaServiceStub)
        await stub.SetArenaVisible(SetArenaVisibleRequest(arena_id=_to_object_id(arena_id), visible=visible), timeout=2.0)
        return True

    @staticmethod
    @safe_async_rpc(default=[])
    async def list_arenas(conn: GrpcConnection) -> list[dict]:
        """
        列出当前所有 Arena 实例（包含资源、锚点、可见/加载状态、Actor 数等）。
        """
        stub = conn.get_stub(ArenaServiceStub)
        resp: ListArenasResponse = await stub.ListArenas(ListArenasRequest(), timeout=2.0)
        out: list[dict] = []
        for a in resp.arenas:
            out.append({
                "id": _fguid_bytes_to_str(a.arena_id.guid),
                "asset_path": a.asset_path,
                "anchor": proto_to_sdk(a.anchor),
                "is_loaded": bool(a.is_loaded),
                "is_visible": bool(a.is_visible),
                "num_actors": int(a.num_actors),
            })
        return out

    @staticmethod
    @safe_async_rpc(default=None)
    async def spawn_actor_in_arena(
        conn: GrpcConnection,
        arena_id: str,
        class_path: str,
        local_transform: Transform,
        timeout: float = 5.0,
    ) -> dict | None:
        """
        在指定 Arena 的局部坐标系下生成一个 Actor（蓝图/类路径）。
        返回 {"id","name","class_path"}。
        """
        stub = conn.get_stub(ArenaServiceStub)
        req = SpawnActorInArenaRequest(
            arena_id=_to_object_id(arena_id),
            class_path=class_path,
            local_transform=sdk_to_proto(local_transform),
        )
        resp: SpawnActorInArenaResponse = await stub.SpawnActorInArena(req, timeout=timeout)
        ai = resp.actor
        return {
            "id": _fguid_bytes_to_str(ai.id.guid),
            "name": ai.name,
            "class_path": ai.class_path,
        }

    @staticmethod
    @safe_async_rpc(default=False)
    async def set_actor_pose_local(
        conn: GrpcConnection,
        arena_id: str,
        actor_id: str,
        local_transform: Transform,
        reset_physics: bool = True,
    ) -> bool:
        """
        将 `actor_id` 放到该 Arena 的局部 Transform（内部转换为世界坐标并 Teleport）。
        """
        stub = conn.get_stub(ArenaServiceStub)
        await stub.SetActorPoseLocal(SetActorPoseLocalRequest(
            arena_id=_to_object_id(arena_id),
            actor_id=_to_object_id(actor_id),
            local_transform=sdk_to_proto(local_transform),
            reset_physics=reset_physics,
        ), timeout=2.0)
        return True

    @staticmethod
    @safe_async_rpc(default=None)
    async def get_actor_pose_local(conn: GrpcConnection, arena_id: str, actor_id: str) -> Transform | None:
        """
        读取 `actor_id` 在该 Arena 局部坐标系下的 Transform。
        """
        stub = conn.get_stub(ArenaServiceStub)
        resp: GetActorPoseLocalResponse = await stub.GetActorPoseLocal(GetActorPoseLocalRequest(
            arena_id=_to_object_id(arena_id),
            actor_id=_to_object_id(actor_id),
        ), timeout=2.0)
        return proto_to_sdk(resp.local_transform)

    @staticmethod
    @safe_async_rpc(default=None)
    async def local_to_world(conn: GrpcConnection, arena_id: str, local_transform: Transform) -> Transform | None:
        """
        将 Arena 局部 Transform 转为世界 Transform。
        """
        stub = conn.get_stub(ArenaServiceStub)
        resp: LocalToWorldResponse = await stub.LocalToWorld(LocalToWorldRequest(
            arena_id=_to_object_id(arena_id),
            local=sdk_to_proto(local_transform),
        ), timeout=2.0)
        return proto_to_sdk(resp.world)

    @staticmethod
    @safe_async_rpc(default=None)
    async def world_to_local(conn: GrpcConnection, arena_id: str, world_transform: Transform) -> Transform | None:
        """
        将世界 Transform 转为 Arena 局部 Transform。
        """
        stub = conn.get_stub(ArenaServiceStub)
        resp: WorldToLocalResponse = await stub.WorldToLocal(WorldToLocalRequest(
            arena_id=_to_object_id(arena_id),
            world=sdk_to_proto(world_transform),
        ), timeout=2.0)
        return proto_to_sdk(resp.local)


    @staticmethod
    @safe_async_rpc(default=False)
    async def destroy_actor(conn: GrpcConnection, actor_id: bytes | str | dict) -> bool:
        stub = conn.get_stub(DemoRLServiceStub)
        req = DestroyActorRequest(actor_id=_to_object_id(actor_id))
        await stub.DestroyActor(req, timeout=2.0)
        return True

    # Arena —— Load/Reset/Destroy 现在为“延迟返回”，但对 SDK 来说依然是 await 即可
    # （因服务端改 Reactor 实现，接口签名与返回值未变，ID 也保持不变，无需修改）

    @staticmethod
    @safe_async_rpc(default=False)
    async def arena_destroy_actor(conn: GrpcConnection, arena_id: str, actor_id: str) -> bool:
        stub = conn.get_stub(ArenaServiceStub)
        req = DestroyActorInArenaRequest(arena_id=_to_object_id(arena_id),
                                         actor_id=_to_object_id(actor_id))
        await stub.DestroyActorInArena(req, timeout=2.0)
        return True

    @staticmethod
    @safe_async_rpc(default=(None, None))
    async def arena_simple_move_towards(
        conn: GrpcConnection,
        arena_id: str,
        target_local_location: Vector3,
        orientation_mode: int = 0,  # 0 KEEP_CURRENT, 1 FACE_MOVEMENT, 2 GIVEN
        given_forward: Vector3 | None = None,
        timeout: float = 3600.0,
    ) -> tuple[dict | None, dict | None]:
        stub = conn.get_stub(ArenaServiceStub)
        req = SimpleMoveTowardsInArenaRequest(
            arena_id=_to_object_id(arena_id),
            target_local_location=sdk_to_proto(target_local_location),
            orientation_mode=orientation_mode,
        )
        if orientation_mode == 2 and given_forward is not None:
            req.given_forward.CopyFrom(sdk_to_proto(given_forward))

        resp: SimpleMoveTowardsInArenaResponse = await stub.SimpleMoveTowardsInArena(req, timeout=timeout)
        current_location = proto_to_sdk(resp.current_location)
        hit_result = {"hit_actor": resp.hit_result.hit_actor} if resp.HasField("hit_result") else None
        return current_location, hit_result

    @staticmethod
    @safe_async_rpc(default=[])
    async def single_line_trace_by_object(
        conn: GrpcConnection,
        jobs: list[dict],
        global_actors_to_ignore: list[bytes | str | dict] | None = None,
        timeout: float = 20.0,
    ) -> list[dict]:
        """
        批量 SingleLineTraceByObject（每条线返回 0/1 次命中）。
        jobs: [
        {
            "start": Vector3, "end": Vector3,
            "object_types": [CollisionObjectType.OBJECT_WORLD_STATIC, ...],
            "trace_complex": bool | None,
            "actors_to_ignore": [actor_id, ...] | None,
        },
        ...
        ]
        """
        req = BatchSingleLineTraceByObjectRequest()
        for j in jobs:
            job = req.jobs.add()
            job.start.CopyFrom(sdk_to_proto(j["start"]))
            job.end.CopyFrom(sdk_to_proto(j["end"]))
            for ot in j.get("object_types", []):
                job.object_types.append(int(ot))
            if "trace_complex" in j and j["trace_complex"] is not None:
                job.trace_complex = bool(j["trace_complex"])
            for ig in j.get("actors_to_ignore", []) or []:
                job.actors_to_ignore.add().CopyFrom(_to_object_id(ig))

        for ig in global_actors_to_ignore or []:
            req.global_actors_to_ignore.add().CopyFrom(_to_object_id(ig))

        stub = conn.get_stub(DemoRLServiceStub)
        resp = await stub.BatchSingleLineTraceByObject(req, timeout=timeout)

        out: list[dict] = []
        for r in resp.results:
            item = {
                "job_index": int(r.job_index),
                "blocking_hit": bool(r.blocking_hit),
                "distance": float(r.distance),
                "impact_point": proto_to_sdk(r.impact_point),
            }
            if r.HasField("actor_state"):
                item["actor_state"] = _actor_state_to_dict(r.actor_state)
            out.append(item)
        return out