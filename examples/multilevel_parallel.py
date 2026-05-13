"""
Multi-level Parallel Training Demo

运行方式（示例）：
  uv run --with numpy ./examples/multilevel_parallel.py
"""
import asyncio
import random
import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode
GRPC_ENDPOINT = "127.0.0.1:5726"

# 你自己的关卡资源（同一资源可多实例）：
LEVELS = [
    "/Game/Maps/Sublevels/SubLevel_005.SubLevel_005",
    "/Game/Maps/Sublevels/SubLevel_005.SubLevel_005",
    "/Game/Maps/Sublevels/SubLevel_005.SubLevel_005",
    "/Game/Maps/Sublevels/SubLevel_005.SubLevel_005",
    "/Game/Maps/Sublevels/SubLevel_005.SubLevel_005",
]

# 在每个 Arena 里生成的蓝图类
AGENT_BP = "/Game/TongSim/Characters/Weiguo_V01/BP_Weiguo.BP_Weiguo_C"
SPAWN_BLUEPRINT = "/Game/Developer/DemoCoin/BP_DemoCoin.BP_DemoCoin_C"

def arena_anchor_at(x: float) -> ts.Transform:
    # 将每个 Arena 沿 X 方向间隔摆放，避免重叠
    return ts.Transform(location=ts.Vector3(x, 0, 0))

async def run_one_arena(context: WorldContext, arena_id: str):
    # 在该 Arena 里生成一个 Actor（局部 (0,0,0)）
    spawned = await ts.UnaryAPI.spawn_actor_in_arena(
        context.conn, arena_id, AGENT_BP, ts.Transform(location=ts.Vector3(300, 580, 100))
    )
    
    for _ in range(4):
        tgt = ts.Vector3(random.uniform(-100, 100), random.uniform(-100, 100), 0)
        await ts.UnaryAPI.spawn_actor_in_arena(
            context.conn, arena_id, SPAWN_BLUEPRINT, ts.Transform(location=ts.Vector3(300, 580, 100) + tgt)
        )
    
    if not spawned:
        print(f"[Arena {arena_id}] spawn failed.")
        return
    agent_id = spawned["id"]
    print(f"[Arena {arena_id}] agent:", spawned)

    # 并行执行若干 step（局部随机位移）
    for _ in range(30):
        tgt = ts.Vector3(random.uniform(-100, 100), random.uniform(-100, 100), 0)
        cur_transform = await ts.UnaryAPI.get_actor_transform(context.conn, agent_id)
        loc, hit = await ts.UnaryAPI.simple_move_towards(
            context.conn,
            actor_id= agent_id,
            target_location= cur_transform.location + tgt,
            speed_uu_per_sec = 120.0,
            timeout=3600.0,
        )
        if hit :
            print(hit)
        if hit and hit["hit_actor"].tag == "RL_Coin":
            print("hit")
            await ts.UnaryAPI.destroy_actor(context.conn, hit["hit_actor"].object_info.id.guid)
        # await asyncio.sleep(0.05)  # 20Hz

async def run(context):
    # 1) 加载多个 Arena（依次沿 X 方向平移 3000 单位）
    arena_ids: list[str] = []
    state_list = await ts.UnaryAPI.query_info(context.conn)

    for a in state_list:
        print(a["name"])
    print(f"  - actor count: {len(state_list)}")
    for i, level in enumerate(LEVELS):
        aid = await ts.UnaryAPI.load_arena(
            context.conn, level_asset_path=level, anchor=arena_anchor_at(3000 * i), make_visible=True
        )
        if not aid:
            print(f"[load_arena] failed: {level}")
            continue
        arena_ids.append(aid)

    print("\nArenas:", await ts.UnaryAPI.list_arenas(context.conn))
    state_list = await ts.UnaryAPI.query_info(context.conn)
    print(f"  - actor count: {len(state_list)}")
    for a in state_list:
        print(a["name"])
    # # # 2) 并行控制各 Arena
    await asyncio.gather(*(run_one_arena(context, aid) for aid in arena_ids))
    state_list = await ts.UnaryAPI.query_info(context.conn)
    for a in state_list:
        print(a["name"])
    # # 3) 重置 & 销毁（示例）
    for aid in arena_ids:
        # await ts.UnaryAPI.reset_arena(context.conn, aid)
        await ts.UnaryAPI.destroy_arena(context.conn, aid)
        

    
def main():
    import time
    print("[INFO] Connecting to TongSim ...")
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        ue.context.sync_run(ts.UnaryAPI.reset_level(ue.context.conn))
        while True:
            ue.context.sync_run(run(ue.context))
            time.sleep(3.0)
        
    print("[INFO] Done.")

if __name__ == "__main__":
    main()

