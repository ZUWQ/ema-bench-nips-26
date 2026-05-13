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
    """
    端到端测试 DemoRLService 的所有 API。
    """
    print("\n[EMAS:S1] refresh_map_actors")
    await ts.UnaryAPI.refresh_actors_map(context.conn)
    time.sleep(3)

    print("\n[RL:2] query_info")
    state_list = await ts.UnaryAPI.query_info(context.conn)
    print(f"  - actor count: {len(state_list)}")
    for actor in state_list:
        if "Door" in actor["name"]:
            print(actor["name"], actor["location"])
    time.sleep(3)

    print("\n[RL:1] Generate FireDog ...")
    actor = await ts.UnaryAPI.spawn_actor(
        context.conn,
        blueprint=FireDog_BP,
        transform=ts.Transform(
            location=pos1,
            # rotation=ts.math.euler_to_quaternion(ts.Vector3(0,0,90),is_degree=True),
        ),
        name="FireDog",
        tags=["FireDog"],
        timeout=5.0,
    )
    time.sleep(3)

    print("\n[EMAS:N2] QueryNavDistance (pick first two actors)")
    if len(state_list) >= 2:
        actor_a = actor
        actor_b = state_list[804]
        nav_dist = await ts.UnaryAPI.query_nav_distance(
            context.conn,
            agent_id=actor_a["id"],
            target_id=actor_b["id"],
            allow_partial=True,
        )
        print(
            f"  - {actor_a['name']} -> {actor_b['name']}:{actor_b['location']} nav_distance: {nav_dist:.2f}"
        )
    else:
        print("  - actor 数量不足 2，跳过 QueryNavDistance 演示")
    time.sleep(3)

    print("\n[EMAS:S2] start_to_burn")
    await ts.UnaryAPI.start_to_burn(context.conn)
    time.sleep(30)

    print("\n[EMAS:G3] get_burned_area")
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(burned_state)
    time.sleep(3)

    print("\n[EMAS:S3] pause_scene")
    await ts.UnaryAPI.pause_scene(context.conn, True)
    time.sleep(10)

    print("\n[EMAS:S3] run_scene")
    await ts.UnaryAPI.pause_scene(context.conn, False)
    time.sleep(3)

    print("\n[EMAS:G3] get_burned_area")
    burned_state = await ts.UnaryAPI.get_burned_area(context.conn)
    print(burned_state)
    time.sleep(3)

    time.sleep(30)
    print("\n[EMAS:G8] get_destroyed_objects")
    destroyed_objects = await ts.UnaryAPI.get_destroyed_objects(context.conn)
    print("destroyed_objects:", len(destroyed_objects))
    print(destroyed_objects)
    time.sleep(3)

    print(f"\n[EMAS:N1] NavigateToLocation ")
    resp = await ts.UnaryAPI.navigate_to_location(
            context.conn,
            actor_id=actor["id"],
            target_location=door_pos,
            accept_radius=ACCEPT_RADIUS,
            allow_partial=ALLOW_PARTIAL,
            speed_uu_per_sec=SPEED_UU_PER_SEC,
            timeout=MOVE_TIMEOUT,
        )
    print("navigate resp:", resp)
    time.sleep(3)

    print(f"\n[EMAS:N1] NavigateToLocation ")
    resp = await ts.UnaryAPI.navigate_to_location(
            context.conn,
            actor_id=actor["id"],
            target_location=door_pos2,
            accept_radius=ACCEPT_RADIUS,
            allow_partial=ALLOW_PARTIAL,
            speed_uu_per_sec=SPEED_UU_PER_SEC,
            timeout=MOVE_TIMEOUT,
        )
    print("navigate resp:", resp)
    time.sleep(3)

    print(f"\n[EMAS:N1] NavigateToLocation ")
    resp = await ts.UnaryAPI.navigate_to_location(
            context.conn,
            actor_id=actor["id"],
            target_location=tv_pos,
            accept_radius=ACCEPT_RADIUS,
            allow_partial=ALLOW_PARTIAL,
            speed_uu_per_sec=SPEED_UU_PER_SEC,
            timeout=MOVE_TIMEOUT,
        )
    print("navigate resp:", resp)
    time.sleep(3)

    print("\n[EMAS:S5] ExtinguishFire")
    res =await ts.UnaryAPI.extinguish_fire(context.conn,actor["id"])
    print(res)

    print("\n[EMAS:G1] GetPerceptionInfo")
    perception = await ts.UnaryAPI.get_embodied_perception(context.conn,actor["id"])
    print("actor 数量:", len(perception["actor_info"]))
    print("actor info:", perception["actor_info"])
    print("NPC 数量:", len(perception["npc_info"]))
    for npc in perception["npc_info"]:
        print(npc["object_info"]["name"], npc["health"], npc["position"])
    time.sleep(3)

    print("\n[EMAS:S7] set_extinguisher_rotation")
    res = await ts.UnaryAPI.set_extinguisher_rotation(context.conn,actor["id"],30,60)
    print(res)
    time.sleep(3)

    print("\n[EMAS:G9] get_agent_extinguished_objects")
    res = await ts.UnaryAPI.get_agent_extinguished_objects(context.conn)
    print(res)
    time.sleep(3)

    print("\n[EMAS:S4-1] sendfollow")
    resfollow = await ts.UnaryAPI.sendfollow(context.conn,actor["id"])
    print(resfollow)
    time.sleep(5)

    print(f"\n[EMAS:N1] NavigateToLocation ")
    resp = await ts.UnaryAPI.navigate_to_location(
            context.conn,
            actor_id=actor["id"],
            target_location=Hall_Target,
            accept_radius=ACCEPT_RADIUS,
            allow_partial=ALLOW_PARTIAL,
            speed_uu_per_sec=SPEED_UU_PER_SEC,
            timeout=MOVE_TIMEOUT,
        )
    print("navigate resp:", resp)
    time.sleep(3)

    print("\n[EMAS:S4-2] sendstopfollow")
    resstopfollow = await ts.UnaryAPI.sendstopfollow(context.conn,actor["id"])
    print(resstopfollow) 
    time.sleep(2)

    print("\n[EMAS:G2] get_selfstate")
    resstopfollow = await ts.UnaryAPI.get_selfstate(context.conn,actor["id"])
    print(resstopfollow) 
    time.sleep(2)

    print("\n[EMAS:G4] get_obj_residual")
    obj_health = await ts.UnaryAPI.get_obj_residual(context.conn)
    print("objhealth:",obj_health)
    time.sleep(3)

    print("\n[EMAS:G5] get_npc_health")
    npc_health = await ts.UnaryAPI.get_npc_health(context.conn)
    print("npc_health:",npc_health)
    # npc_id = npc_health.keys()
    # for i in npc_id:
    #     ast = await ts.UnaryAPI.get_actor_state(context.conn, i)
    #     print("  - actor_state:")
    #     print(ast)
    time.sleep(3)

    print("\n[EMAS:G6] GetFireState")
    OutFire = await ts.UnaryAPI.get_outfire_state(context.conn)
    print("OutFire:", OutFire)
    time.sleep(3)

    print("\n[EMAS:G7] GetNPCPostions")
    postions = await ts.UnaryAPI.get_npc_postions(context.conn)
    print("NPCpostions:", postions)
    time.sleep(3)

    print("\n[RL:2] query_info")
    state_list = await ts.UnaryAPI.query_info(context.conn)
    print(f"  - actor count: {len(state_list)}")
    time.sleep(3)

    # print("\n[EMAS:R1] listen_sos---------")
    # asyncio.create_task(listen_sos(context, [id1, id2]))
    # print("SOS listener running")
    # await asyncio.sleep(999)

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