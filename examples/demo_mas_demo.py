"""
RL API 综合演示脚本
------------------
覆盖并验证以下接口（均为 DemoRLService 的 unary RPC）：
1) ResetLevel                    —— 重置关卡
2) QueryState                    —— 查询全局 Actor 状态
3) SpawnActor                    —— 生成一个 Agent（带初始 Transform、可选 name/tags）
4) GetActorTransform / SetActorTransform
5) GetActorState
6) SimpleMoveTowards             —— 三种朝向模式：
   - ORIENTATION_KEEP_CURRENT
   - ORIENTATION_FACE_MOVEMENT
   - ORIENTATION_GIVEN（given_forward 仅使用 XY 分量）
"""

import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode, CollisionObjectType
import asyncio
import time
import uuid
GRPC_ENDPOINT = "127.0.0.1:5726"

FireDog_BP= "/Game/Blueprint/BP_Firedog.BP_Firedog_C"
SaveDog_BP= "/Game/Blueprint/BP_Savedog.BP_Savedog_C"
Firefighter_BP= "/Game/Blueprint/BP_Firefighter.BP_Firefighter_C"

# SimpleMoveTowards 的三个目标点
MOVE_TARGET_KEEP = ts.Vector3(1200, -2000, 0)
MOVE_TARGET_FACE = ts.Vector3(400, -1500, 0)
MOVE_TARGET_GIVEN = ts.Vector3(400, -2500, 0)

Hall_Spawn = ts.Vector3(-200, 100, 1000)
Hall_Target = ts.Vector3(200, 100, 1000)
WashRoom_Spawn = ts.Vector3(945, 40, 1000)
WashRoom_Target = ts.Vector3(1100, 190, 1000)
WashRoom_Target2 = ts.Vector3(1045, 1000, 1000)
pos1 = ts.Vector3(200, 0, 980)
door_pos = ts.Vector3(675.6, 123.1, 970)
door_pos2 = ts.Vector3(1025.79, 461.157, 980)
tv_pos = ts.Vector3(1596, 200, 980)

ACCEPT_RADIUS = 50.0
ALLOW_PARTIAL = False
SPEED_UU_PER_SEC = 50.0
MOVE_TIMEOUT = 60.0

# 给定朝向的 forward 向量（仅用 XY 分量；Z 会被服务端忽略）
GIVEN_FORWARD = ts.Vector3(0, 1, 0)  # 面向 +Y 方向
# =================================
async def listen_sos(context, ids):
    async for sos in ts.UnaryAPI.receive_npc_sos(context.conn, ids):
        print("[SOS]", sos)

async def mas_api_full_demo(context: WorldContext):
    print("\n[EMAS:S1] refresh_map_actors")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    time.sleep(3)


    # print("\n[RL:2] query_info")
    # state_list = await ts.UnaryAPI.query_info(context.conn)
    # print(f"  - actor count: {len(state_list)}")
    # for actor in state_list:
    #     if "Door" in actor["name"]:
    #         print(actor["name"], actor["location"])
    # time.sleep(3)


    print("\n[EMAS:S2] start_to_burn")
    await ts.UnaryAPI.start_to_burn(context.conn)
    # time.sleep(10)


    print("\n[RL:1] Generate FireDog ...")
    # 关卡里往往已有名为 FireDog 的放置 Actor；SpawnActor 的 name 必须在 PersistentLevel 内唯一，否则会崩溃。
    spawn_name = f"FireDog_Grpc_{uuid.uuid4().hex[:12]}"
    actor = await ts.UnaryAPI.spawn_actor(
    context.conn,
    blueprint=FireDog_BP,
    transform=ts.Transform(
    location= ts.Vector3(-1180, -140, 1000),
    rotation=ts.math.euler_to_quaternion(ts.Vector3(0,0,-140),is_degree=True),
    ),
    name=spawn_name,
    tags=["FireDog"],
    timeout=5.0,
    )
    time.sleep(3)


    spawn_name = f"FireDog_Grpc_{uuid.uuid4().hex[:12]}"
    actor1 = await ts.UnaryAPI.spawn_actor(
    context.conn,
    blueprint=FireDog_BP,
    transform=ts.Transform(
    location= ts.Vector3(-1452, -73, 1000),
    rotation=ts.math.euler_to_quaternion(ts.Vector3(0,0,0),is_degree=True),
    ),
    name=spawn_name,
    tags=["FireDog"],
    timeout=5.0,
    )
    spawn_name = f"FireDog_Grpc_{uuid.uuid4().hex[:12]}"
    actor2 = await ts.UnaryAPI.spawn_actor(
    context.conn,
    blueprint=SaveDog_BP,
    transform=ts.Transform(
    location= ts.Vector3(-1400, 394, 1000),
    rotation=ts.math.euler_to_quaternion(ts.Vector3(0,0,180),is_degree=True),
    ),
    name=spawn_name,
    tags=["SaveDog"],
    timeout=5.0,
    )
    time.sleep(3)
    spawn_name = f"SaveDog_Grpc_{uuid.uuid4().hex[:12]}"
    actor3 = await ts.UnaryAPI.spawn_actor(
    context.conn,
    blueprint=FireDog_BP,
    transform=ts.Transform(
    location= ts.Vector3(-2113, -136, 1000),
    rotation=ts.math.euler_to_quaternion(ts.Vector3(0,0,156),is_degree=True),
    ),
    name=spawn_name,
    tags=["SaveDog"],
    timeout=5.0,
    )
    time.sleep(3)
    print("\n[EMAS:S7] set_extinguisher_rotation")
    res = await ts.UnaryAPI.set_extinguisher_rotation(context.conn,actor1["id"],-45,0)
    print(res)
    time.sleep(3)
    print("\n[EMAS:3] ExtinguishFire")
    res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    print(res)
    print("\n[EMAS:3] ExtinguishFire")
    res =await ts.UnaryAPI.extinguish_fire(context.conn,actor1["id"])
    print(res)



def main():
    """主入口：建立连接并运行综合演示。"""
    print("[INFO] 连接到 TongSim ...")
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        # 在一个异步任务里跑完整流程
        ue.context.sync_run(mas_api_full_demo(ue.context))
    print("[INFO] 演示完成。")


if __name__ == "__main__":
    main()

    # 1) ResetLevel
    # print("\n[1] ResetLevel ...")
    # ok = await ts.UnaryAPI.reset_level(context.conn)
    # print("  - done:", ok)


    ### 以下为灭火器配置测试代码
    # print("\n[EMAS:3] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    # time.sleep(18)
    # print("\n[2] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    # time.sleep(3)
    # print("\n[EMAS:S6] set_extinguisher")
    # res =await ts.UnaryAPI.set_extinguisher(context.conn,actor["id"],2,20)
    # print(res)
    # time.sleep(3)
    # print("\n[4] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    # time.sleep(3)
    # print("\n[6] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    # time.sleep(3)
    # print("\n[7] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    # time.sleep(30)
    # print("\n[8] set_extinguisher")
    # res =await ts.UnaryAPI.set_extinguisher(context.conn,actor["id"],3,10)
    # print(res)
    # print("\n[7] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    # print(res)
    ### 以上为灭火器配置测试代码end

    # print("\n[3] GetPerceptionInfo")
    # list1 = await ts.UnaryAPI.perception_info(context.conn,actor["id"])
    # print(list1)

    # print("\n[4] ExtinguishFire")
    # res =await ts.UnaryAPI.extinguish(context.conn,actor["id"])
    # print(res)

    # print("\n[2] SimpleMoveTowards (KEEP_CURRENT) ->", MOVE_TARGET_KEEP)
    # cur_loc, hit = await ts.UnaryAPI.simple_move_towards(
    #     context.conn,
    #     actor_id=spawned["id"],
    #     target_location=ts.Vector3(1045, 200, 1000),
    #     orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
    #     timeout=3600.0,
    #     speed_uu_per_sec = 200.0,
    # )
    # if not spawned:
    #     print("  - spawn failed!")
    #     return
    # print("  - spawned actor:")
    # print("    ", hit)
    # id1 = spawned["id"]