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

GRPC_ENDPOINT = "127.0.0.1:5726"

SPAWN_BLUEPRINT = "/Game/Developer/DemoCoin/BP_DemoCoin.BP_DemoCoin_C"
AGENT_BP = "/Game/Developer/Characters/UE4Mannequin/BP_UE4Mannequin.BP_UE4Mannequin_C"


# SimpleMoveTowards 的三个目标点
MOVE_TARGET_KEEP = ts.Vector3(1200, -2000, 0)
MOVE_TARGET_FACE = ts.Vector3(6500, -1300, 0)
MOVE_TARGET_GIVEN = ts.Vector3(400, -2500, 0)

# 给定朝向的 forward 向量（仅用 XY 分量；Z 会被服务端忽略）
GIVEN_FORWARD = ts.Vector3(0, 1, 0)  # 面向 +Y 方向
# =================================


async def rl_api_full_demo(context: WorldContext):
    """
    端到端测试 DemoRLService 的所有 API。
    """
    # 1) ResetLevel
    # print("\n[1] ResetLevel ...")
    # ok = await ts.UnaryAPI.reset_level(context.conn)
    # print("  - done:", ok)

    # 2) QueryState（全局）
    print("\n[2] QueryState (global) ...")
    state_list = await ts.UnaryAPI.query_info(context.conn)
    print(f"  - actor count: {len(state_list)}")
    if state_list:
        print("  - first actor sample:")
        print(state_list[0])

    # 3) SpawnActor（带 Transform / name / tags）
    print("\n[3] SpawnActor ...")
    spawned = await ts.UnaryAPI.spawn_actor(
        context.conn,
        blueprint=SPAWN_BLUEPRINT,
        transform=ts.Transform(
            location=ts.Vector3(350, -2000, 200),
        ),
        name="DemoRL_Spawned",
        tags=["RL_Target"],
        timeout=5.0,
    )
    if not spawned:
        print("  - spawn failed!")
        return
    print("  - spawned actor:")
    print("    ", spawned)
    spawned_id = spawned["id"]

    # # 4) GetActorTransform
    # print("\n[4] GetActorTransform ...")
    # tf_before = await ts.UnaryAPI.get_actor_transform(context.conn, spawned_id)
    # print("  - transform (before set):")
    # print(tf_before)

    # # 5) SetActorTransform（TeleportPhysics）
    print("\n[5] SetActorTransform (teleport) ...")
    set_ok = await ts.UnaryAPI.set_actor_transform(
        context.conn,
        spawned_id,
        ts.Transform(
            location=ts.Vector3(450, -2000, 200),
        ),
    )
    # print("  - done:", set_ok)

    # # 再 GetActorTransform 校验
    # tf_after = await ts.UnaryAPI.get_actor_transform(context.conn, spawned_id)
    # print("  - transform (after set):")
    # print(tf_after)

    # # 6) GetActorState
    # print("\n[6] GetActorState ...")
    # ast = await ts.UnaryAPI.get_actor_state(context.conn, spawned_id)
    # print("  - actor_state:")
    # print(ast)


    agent = await ts.UnaryAPI.spawn_actor(
        context.conn,
        blueprint=AGENT_BP,
        transform=ts.Transform(
            location=ts.Vector3(250, -2000, 200),
        ),
        name="DemoRL_Agent",
        timeout=5.0,
    )
    
    agent_2 = await ts.UnaryAPI.spawn_actor(
        context.conn,
        blueprint=AGENT_BP,
        transform=ts.Transform(
            location=ts.Vector3(1050, -2000, 200),
        ),
        name="DemoRL_Agent2",
        timeout=5.0,
    )

    # 7) SimpleMoveTowards —— ORIENTATION_KEEP_CURRENT
    print("\n[7] SimpleMoveTowards (KEEP_CURRENT) ->", MOVE_TARGET_KEEP)
    cur_loc, hit = await ts.UnaryAPI.simple_move_towards(
        context.conn,
        actor_id=agent["id"],
        target_location=MOVE_TARGET_KEEP,
        orientation_mode=RLDemoOrientationMode.ORIENTATION_KEEP_CURRENT,
        timeout=3600.0,
    )
    print("  - current_location:", (cur_loc))
    print("  - hit_result:", (hit) if hit else "None")
    
    # if hit["hit_actor"].tag == "RL_Coin":
    #     await ts.UnaryAPI.destroy_actor(context.conn, hit["hit_actor"].object_info.id.guid)

    # 8) SimpleMoveTowards —— ORIENTATION_FACE_MOVEMENT
    print("\n[8] SimpleMoveTowards (FACE_MOVEMENT) ->", MOVE_TARGET_FACE)
    cur_loc, hit = await ts.UnaryAPI.simple_move_towards(
        context.conn,
        actor_id=spawned_id,
        target_location=MOVE_TARGET_FACE,
        orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
        timeout=3600.0,
    )
    print("  - current_location:", (cur_loc))
    print("  - hit_result:", (hit) if hit else "None")

    # 9) SimpleMoveTowards —— ORIENTATION_GIVEN（given_forward 仅使用 XY 分量）
    # print(
    #     "\n[9] SimpleMoveTowards (GIVEN forward=",
    #     GIVEN_FORWARD,
    #     ") ->",
    #     MOVE_TARGET_GIVEN,
    # )
    # cur_loc, hit = await ts.UnaryAPI.simple_move_towards(
    #     context.conn,
    #     actor_id=agent["id"],
    #     target_location=MOVE_TARGET_GIVEN,
    #     orientation_mode=RLDemoOrientationMode.ORIENTATION_GIVEN,
    #     given_forward=GIVEN_FORWARD,
    #     speed_uu_per_sec = 100.0,
    #     timeout=3600.0,
    # )
    # print("  - current_location:", (cur_loc))
    # print("  - hit_result:", (hit) if hit else "None")


    # 10) single_line_trace_by_object
    print(await ts.UnaryAPI.single_line_trace_by_object(
        context.conn,
        jobs=[
            {
                "start": ts.Vector3(100, -1000, 200), 
                "end": ts.Vector3(1000, -1000, 200),
                "object_types": [
                    # CollisionObjectType.OBJECT_WORLD_STATIC, 
                    # CollisionObjectType.OBJECT_WORLD_DYNAMIC, 
                    CollisionObjectType.OBJECT_PAWN, 
                ],
            },
        ]
    ))

def main():
    """主入口：建立连接并运行综合演示。"""
    print("[INFO] 连接到 TongSim ...")
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        # 在一个异步任务里跑完整流程
        ue.context.sync_run(rl_api_full_demo(ue.context))
    print("[INFO] 演示完成。")


if __name__ == "__main__":
    main()