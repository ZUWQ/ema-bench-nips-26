from utils_main import *
import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode,CollisionObjectType
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import gymnasium as gym
from typing import List, Dict
from collections import defaultdict
import asyncio
import traceback

GRPC_ENDPOINT = "127.0.0.1:5726"

# [修改] 将蓝图和关卡资源路径放在文件顶部，方便管理
LEVEL_FOR_ARENA = "/Game/Developer/Maps/L_DemoRL.L_DemoRL"
SPAWN_BLUEPRINT_COIN = "/Game/developer/DemoCoin/BP_DemoCoin.BP_DemoCoin_C"
AGENT_BP = "/Game/Developer/Characters/UE4Mannequin/BP_UE4Mannequin.BP_UE4Mannequin_C"
SPAWN_BLUEPRINT_POISON = "/Game/developer/DemoCoin/BP_DemoPoision.BP_DemoPoision_C"

action_multiplier = 150
height=1900
width=-1900
block_ranges=[[[1200,1500],[-1500,-1000]]]#可能有多个block_range,每个形式为（最小x，最大x，最小y，最大y）
coin_scale=1.2
x_bounds = (50, height)
y_bounds = (-50, width)

def arena_anchor_at(x: float, y: float = 0.0) -> ts.Transform:
    """
    [修改] 辅助函数，用于计算每个Arena的摆放位置，避免重叠。
    可以扩展到二维布局。
    """
    return ts.Transform(location=ts.Vector3(x, y, 0))

async def get_actor_transform_safe(conn, actor_id_list):
    if not actor_id_list:
        return {}
    tasks = [ts.UnaryAPI.get_actor_transform(conn, actor_id) for actor_id in actor_id_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    id_to_transform_map = {}
    for i, res in enumerate(results):
        if isinstance(res, ts.Transform):
            actor_id = actor_id_list[i]
            id_to_transform_map[actor_id] = res
            
    return id_to_transform_map

async def spawn_actors_concurrently(
    context: "WorldContext",
    arena_id: str,
    count: int,
    blueprint_path: str,
    actor_type_name: str,
    rect_corner1: "ts.Vector3",
    rect_corner2: "ts.Vector3",
    spawn_z: float
) -> List[dict]:
    
    tasks = []
    for i in range(count):
        rand_x, rand_y = generate_safe_random_location(x_bounds, y_bounds, block_ranges)
        
        tasks.append(ts.UnaryAPI.spawn_actor_in_arena(
            context.conn,
            arena_id,
            blueprint_path,
            ts.Transform(location=ts.Vector3(rand_x, rand_y, spawn_z)),
            15.0,
        ))

    spawn_results = await asyncio.gather(*tasks, return_exceptions=True)
    successful_spawns = []
    failed_spawns = []

    for i, res in enumerate(spawn_results):
        if isinstance(res, dict):
            successful_spawns.append(res)
        else:
            failed_spawns.append((i, res))

    if failed_spawns:
        print(f"  - [警告] Arena {arena_id} 有 {len(failed_spawns)} 个 {actor_type_name} 生成失败:")
        for idx, error in failed_spawns:
            print(f"    - 索引 {idx}: {error}")
        raise RuntimeError(
            f"Arena {arena_id} 生成失败！"
            "可能是连接问题或资源路径错误。"
        )

    return successful_spawns

class waterworld(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(
        self,
        config=None,
        num_arenas: int = 4,
        n_pursuers=5,
        n_evaders=10,
        n_poisons=5,
        n_coop=2,
        n_sensors=30,
        sensor_range=500.0,
        pursuer_max_accel=0.5,
        evader_speed=0.15,
        poison_speed=0.15,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.01,
        local_ratio=0.9,
        speed_features=True,
        max_cycles=500,
        render_mode=None,
        env_seed=0,
    ):
        
        super().__init__()
        self.ue = ts.TongSim(grpc_endpoint=GRPC_ENDPOINT)
        self.context = self.ue.context
        self.conn = self.context.conn
        self.num_arenas = num_arenas
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poisons = n_poisons
        self.n_coop = n_coop
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range
        self.pursuer_max_accel = pursuer_max_accel
        self.evader_speed = evader_speed
        self.poison_speed = poison_speed
        self.poison_reward_num = poison_reward
        self.food_reward_num = food_reward
        self.encounter_reward_num = encounter_reward
        self.thrust_penalty = thrust_penalty
        self.local_ratio = local_ratio
        self.speed_features = speed_features
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.steering_strength = 0.1
        self.env_seed = env_seed
        
        # [新增] 动态计算每个arena的actor总数
        self.actors_per_arena = self.n_pursuers + self.n_evaders + self.n_poisons
        
        # [修改] 状态变量结构
        self.agents = [f"pursuer_{i}" for i in range(self.n_pursuers)]
        self.num_agents = self.n_pursuers
        self.agent_ids_map = [{} for _ in range(self.num_arenas)]
        self.arena_ids = []
        self.arena_anchors = {} # [新增] 保存每个Arena的锚点位置 {arena_id: ts.Transform}
        self.arenas_data = []

        self.obs_feature_sizes = {"agent": 3, "coin": 2, "poison": 2, "wall": 1, "obstacle": 1}
        self.obs_feature_order = ["agent", "coin", "poison", "wall", "obstacle"]
        self.obs_indices = {}
        current_index = 0
        for feature_name in self.obs_feature_order:
            self.obs_indices[feature_name] = current_index
            current_index += self.obs_feature_sizes[feature_name]
        self.single_sensor_dim = current_index

        self.get_spaces()
        self._seed(self.env_seed)

    def get_spaces(self):
        obs_dim = self.single_sensor_dim * self.n_sensors + 2

        obs_space = {agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for i, agent in enumerate(self.agents)}
        act_space = {agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for i, agent in enumerate(self.agents)}
        
        self.observation_space = obs_space
        self.action_space = act_space
        self.obs_dim = obs_dim


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self):
        pass

    def _initialize_arena_data_structure(self):
        """[新增] 辅助函数，用于创建空的Arena状态容器"""
        self.arenas_data = []
        for i in range(self.num_arenas):
            self.arenas_data.append({
                'pos': {'coins': {}, 'poisons': {}, 'agents': {}},
                'ids_of_coins': [],
                'ids_of_poisons': [],
                'ids_of_agents': [],
                'orientation': {},
                'velocities': {},
                'count_step': 0,
                'hit_coin': {},
                'hit_poison': {}
            })

    async def _async_reset_all_logic(self):
        """
        [重构] 这个方法现在只用于完全重置所有环境，通常只在开始时调用一次。
        """
        print("  - [Async] 开始执行所有Arenas的【完全】异步重置任务...")

        # 1) 重置UE关卡，销毁一切
        print("  - [Async] 1. 重置UE主关卡 (销毁所有旧Arenas)...")
        await ts.UnaryAPI.reset_level(self.conn)
        self.arena_ids = []
        self.arena_anchors = {}


        # 2) 并发加载多个Arena
        print(f"  - [Async] 2. 并发加载 {self.num_arenas} 个新Arenas...")
        load_tasks = []
        for i in range(self.num_arenas):
            # [修改] 保存每个Arena的锚点位置
            anchor = arena_anchor_at(3000 * i)
            load_tasks.append(ts.UnaryAPI.load_arena(
                self.conn, 
                level_asset_path=LEVEL_FOR_ARENA, 
                anchor=anchor,
                make_visible=True
            ))
        loaded_arena_ids = await asyncio.gather(*load_tasks)

        # state_list = await ts.UnaryAPI.query_info(self.conn)
        # print(f"  - actor count: {len(state_list)}")
        # if state_list:
        #     print("  - first actor sample:")
        #     print(state_list[0])
        
        # [修改] 将ID和锚点关联起来
        for i, aid in enumerate(loaded_arena_ids):
            if aid:
                self.arena_ids.append(aid)
                self.arena_anchors[aid] = arena_anchor_at(3000 * i)

        if len(self.arena_ids) != self.num_arenas:
            print(f"[警告] 期望加载 {self.num_arenas} 个Arenas, 实际成功 {len(self.arena_ids)} 个。")
        print(f"  - [Async] 成功加载的Arenas: {self.arena_ids}")

        # 3) 为每个Arena并发生成所有类型的 Actor
        # ... (这部分逻辑与你原来的代码相同) ...
        print("  - [Async] 3. 为每个Arena并发生成Actors...")
        all_spawn_tasks = []
        for arena_id in self.arena_ids:
            all_spawn_tasks.append(spawn_actors_concurrently(
                self.context, arena_id, self.n_evaders, SPAWN_BLUEPRINT_COIN, "DemoRL_Coin",
                ts.Vector3(50, -50, 200), ts.Vector3(height, width, 200), 200))
            all_spawn_tasks.append(spawn_actors_concurrently(
                self.context, arena_id, self.n_poisons, SPAWN_BLUEPRINT_POISON, "DemoRL_Poison", 
                ts.Vector3(50, -50, 200), ts.Vector3(height, width, 200), 200))
            all_spawn_tasks.append(spawn_actors_concurrently(
                self.context, arena_id, self.n_pursuers, AGENT_BP, "DemoRL_Agent", 
                ts.Vector3(50, -50, 200), ts.Vector3(height, width, 200), 200))

        all_spawn_results = await asyncio.gather(*all_spawn_tasks)
        
        # 4) 初始化所有Arena的状态
        print("  - [Async] 4. 初始化所有Arenas的状态...")
        all_get_transform_tasks = []
        for i, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[i]
            coin_results = all_spawn_results[i * 3]
            poison_results = all_spawn_results[i * 3 + 1]
            agent_results = all_spawn_results[i * 3 + 2]

            arena_data['ids_of_coins'] = [r['id'] for r in coin_results]
            arena_data['ids_of_poisons'] = [r['id'] for r in poison_results]
            arena_data['ids_of_agents'] = [r['id'] for r in agent_results]
            
            self.agent_ids_map[i] = {self.agents[j]: agent_id for j, agent_id in enumerate(arena_data['ids_of_agents'])}

            all_ids_in_arena = arena_data['ids_of_coins'] + arena_data['ids_of_poisons'] + arena_data['ids_of_agents']
            all_get_transform_tasks.append(get_actor_transform_safe(self.conn, all_ids_in_arena))

        all_id_to_transform_maps = await asyncio.gather(*all_get_transform_tasks)

        for i, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[i]
            id_to_transform = all_id_to_transform_maps[i]

            for id_list_name in ['coins', 'poisons', 'agents']:
                arena_data['pos'][id_list_name] = {}
                id_list = arena_data[f'ids_of_{id_list_name}']
                for actor_id in id_list:
                    if actor_id in id_to_transform:
                        pos = id_to_transform[actor_id]
                        arena_data['pos'][id_list_name][actor_id] = (pos.location.x, pos.location.y, pos.location.z)
                        if id_list_name == 'coins':
                            random_dir = self.np_random.standard_normal(2)
                            random_dir /= np.linalg.norm(random_dir) + 1e-8
                            arena_data['velocities'][actor_id] = random_dir * self.evader_speed
                        elif id_list_name == 'poisons':
                            random_dir = self.np_random.standard_normal(2)
                            random_dir /= np.linalg.norm(random_dir) + 1e-8
                            arena_data['velocities'][actor_id] = random_dir * self.poison_speed
                        elif id_list_name == 'agents':
                            arena_data['orientation'][actor_id] = np.array([1.0, 0.0, 0.0])

        # 5) 构建并返回所有Arenas的初始观测
        print("  - [Async] 5. 构建所有Arenas的初始观测...")
        jobs, rays_directions_map = self.build_observation_rays()#这里构建的观察有问题
        if not jobs:
            return [{} for _ in self.arena_ids]

        ray_results = await ts.UnaryAPI.multi_line_trace_by_object(self.context.conn, jobs=jobs)
        observations = self.process_ray_results(ray_results, rays_directions_map)
        
        print("  - [Async] 所有Arenas的【完全】异步重置任务完成。")
        return observations

    def reset(self, *, seed=None, options=None):
        """
        [重构] 重置【所有】并行环境到初始状态。通常只在训练开始时调用一次。
        """
        super().reset(seed=seed)
        print("\n================== [ENV FULL RESET] ==================")
        
        # 1. 清理并重建所有Arena的状态容器
        print("[1] 清理内部状态...")
        self._initialize_arena_data_structure()

        # 2. 运行所有Arenas的异步重置逻辑
        print("[2] 运行所有Arenas的异步【完全】重置逻辑...")
        initial_observations=self.context.sync_run(self._async_reset_all_logic())
        
        #为每个Arena单独重置并构建观察
        initial_observations = [self.context.sync_run(self._async_reset_one_env_logic(i)) for i in range(self.num_arenas)]

        print("================== [FULL RESET DONE] ==================\n")
        
        infos = [{} for _ in range(self.num_arenas)]
        return initial_observations, infos

    async def _async_reset_one_env_logic(self, arena_idx: int):
        """
        [新增] 只重置指定索引的Arena，通过重新定位Actors而不是重新生成。
        """
        arena_id = self.arena_ids[arena_idx]
        arena_data = self.arenas_data[arena_idx]
        print(f"  - [Async] 开始对 Arena {arena_idx} (ID: {arena_id}) 进行【独立】重置...")

        # 1. 清理该Arena的内部状态
        arena_data['count_step'] = 0
        arena_data['velocities'] = {}
        arena_data['orientation'] = {}
        arena_data['hit_coin'] = {}
        arena_data['hit_poison'] = {}
        
        # 2. 并发地将该Arena内的所有Actor移动到新的随机位置
        move_tasks = []
        all_actor_ids = arena_data['ids_of_agents'] + arena_data['ids_of_coins'] + arena_data['ids_of_poisons']
        
        for actor_id in all_actor_ids:
            rand_x, rand_y = generate_safe_random_location(x_bounds, y_bounds, block_ranges)
            
            # [修改] 使用保存的锚点计算世界坐标
            anchor_loc = self.arena_anchors[arena_id].location
            target_location = ts.Vector3(anchor_loc.x + rand_x, anchor_loc.y + rand_y, 200)
            # 这里有一次重置，如果是coin，需要修改scale 
            if actor_id in arena_data['ids_of_coins']:
                scale=ts.Vector3(coin_scale,coin_scale,coin_scale)
            else:
                scale=ts.Vector3(1.0,1.0,1.0)
            move_tasks.append(ts.UnaryAPI.set_actor_transform(self.conn, actor_id, ts.Transform(location=target_location, scale=scale)))

        await asyncio.gather(*move_tasks, return_exceptions=True)
        # print(f"  - [Arena {arena_idx}] 所有Actors位置已重置。")

        # 3. 重新获取所有Actor的位置并初始化速度/朝向
        id_to_transform = await get_actor_transform_safe(self.conn, all_actor_ids)
        for actor_id in all_actor_ids:
            if actor_id in id_to_transform:
                pos = id_to_transform[actor_id]
                loc_tuple = (pos.location.x, pos.location.y, pos.location.z)
                
                if actor_id in arena_data['ids_of_agents']:
                    arena_data['pos']['agents'][actor_id] = loc_tuple
                    arena_data['orientation'][actor_id] = np.array([1.0, 0.0, 0.0])
                elif actor_id in arena_data['ids_of_coins']:
                    arena_data['pos']['coins'][actor_id] = loc_tuple
                    random_dir = self.np_random.standard_normal(2)
                    random_dir /= np.linalg.norm(random_dir) + 1e-8
                    arena_data['velocities'][actor_id] = random_dir * self.evader_speed
                elif actor_id in arena_data['ids_of_poisons']:
                    arena_data['pos']['poisons'][actor_id] = loc_tuple
                    random_dir = self.np_random.standard_normal(2)
                    random_dir /= np.linalg.norm(random_dir) + 1e-8
                    arena_data['velocities'][actor_id] = random_dir * self.poison_speed

        # 4. 为这个Arena生成新的观测
        jobs, rays_directions_map = self.build_observation_rays_for_single_arena(arena_idx)
        ray_results = await ts.UnaryAPI.multi_line_trace_by_object(self.context.conn, jobs=jobs)
        arena_obs = self.process_ray_results_for_single_arena(
            ray_results, rays_directions_map, arena_idx
        )

        # print(f"  - [Async] Arena {arena_idx} 【独立】重置完成。")
        return arena_obs
    

    def _build_rays_for_arena(self, arena_idx: int, on_missing_agent: str):
        """
        [私有辅助函数] 为单个Arena中的所有智能体构建射线。
        
        Args:
            arena_idx (int): Arena的索引。
            on_missing_agent (str): 当智能体位置数据缺失时的行为。
                                    'raise' -> 抛出ValueError
                                    'skip' -> 跳过该智能体
        
        Returns:
            tuple: 包含任务列表(jobs)和生成的射线方向列表(rays)的元组。
        """
        arena_data = self.arenas_data[arena_idx]
        
        jobs = []
        rays_for_arena = []
        
        for agent_id in arena_data['ids_of_agents']:
            # 统一使用更安全的方式获取朝向，提供默认值
            face_vector = arena_data['orientation'].get(agent_id, np.array([1., 0., 0.]))
            start_point = arena_data['pos']['agents'].get(agent_id)
            
            if not start_point:
                if on_missing_agent == 'raise':
                    raise ValueError(f"Agent ID {agent_id} in Arena {arena_idx} has no position data.")
                elif on_missing_agent == 'skip':
                    continue
            
            generated_rays = generate_circular_rays(face_vector, self.n_sensors, radius=self.sensor_range)
            rays_for_arena.extend(generated_rays)
            
            for ray in generated_rays:
                jobs.append({
                    "start": ts.Vector3(*start_point),
                    "end": ts.Vector3(*(np.array(start_point) + ray)),
                    "object_types": [
                        CollisionObjectType.OBJECT_WORLD_STATIC,
                        CollisionObjectType.OBJECT_WORLD_DYNAMIC,
                        CollisionObjectType.OBJECT_PAWN
                    ],
                    "actors_to_ignore": [agent_id],
                })
        
        return jobs, rays_for_arena
    
    def build_observation_rays_for_single_arena(self, arena_idx: int):
        """
        [新增] 只为单个Arena构建观察射线。
        """
        # 调用辅助函数，并指定在数据缺失时抛出异常
        jobs, all_generated_rays = self._build_rays_for_arena(arena_idx, on_missing_agent='raise')
        return jobs, all_generated_rays
    
    def build_observation_rays(self):
        """
        [修改] 构建所有Arenas中所有智能体的观察射线。
        """
        all_jobs = []
        all_generated_rays = []
        rays_directions_map = {'rays': [], 'offsets': {}}
        for arena_idx, arena_id in enumerate(self.arena_ids):
            # 记录当前arena在总任务列表中的起始索引
            rays_directions_map['offsets'][arena_idx] = len(all_jobs)
            
            # 调用辅助函数，并指定在数据缺失时跳过
            arena_jobs, arena_rays = self._build_rays_for_arena(arena_idx, on_missing_agent='raise')
            
            all_jobs.extend(arena_jobs)
            all_generated_rays.extend(arena_rays)
            
        rays_directions_map['rays'] = all_generated_rays
        return all_jobs, rays_directions_map
    

    def reset_one_env(self, arena_idx: int):
        """
        [新增] 同步接口，用于在训练循环中重置一个已结束的环境。
        """
        # print(f"\n--- [ENV SINGLE RESET] Arena {arena_idx} ---")
        new_observation = self.context.sync_run(self._async_reset_one_env_logic(arena_idx))
        # print(f"--- [SINGLE RESET DONE] Arena {arena_idx} ---\n")
        return new_observation, {}
    

    def _process_rays_for_one_arena(
        self, 
        arena_idx: int, 
        all_ray_results: List[dict], 
        all_ray_directions: List, 
        result_offset: int
    ) -> np.ndarray:
        """
        [私有辅助函数] 处理单个Arena的射线检测结果，并返回Numpy观测数组。
        
        Args:
            arena_idx (int): 正在处理的Arena的索引。
            all_ray_results (List[dict]): 包含所有Arena的射线检测结果的列表。
            all_ray_directions (List): 包含所有射线方向向量的列表。
            result_offset (int): 当前Arena的结果在 all_ray_results 中的起始索引。
        Returns:
            np.ndarray: 为该Arena生成的大小为 (n_pursuers, obs_dim) 的观测数组。
        """
        arena_data = self.arenas_data[arena_idx]
        arena_obs = np.full((self.n_pursuers, self.obs_dim), -1.0, dtype=np.float32)
        rays_per_agent = self.n_sensors
        for agent_local_idx in range(self.n_pursuers):
            for sensor_idx in range(rays_per_agent):
                # 计算在全局结果列表中的索引
                ray_idx = result_offset + agent_local_idx * rays_per_agent + sensor_idx
                if ray_idx >= len(all_ray_results):
                    raise IndexError("Ray index out of bounds for the provided ray results.")
                ray = all_ray_results[ray_idx]
                closest_flags = {'agent': False, 'coin': False, 'poison': False, 'wall': False, 'obstacle': False}
                
                if len(ray['hits']) > 0:
                    for hit in ray['hits']:
                        hit_distance = max(0.0, min(1.0, hit['distance'] / (self.sensor_range + 1e-8)))
                        actor_state = hit['actor_state']
                        actor_id = actor_state["id"]
                        tag = actor_state["tag"]
                        
                        base_col = sensor_idx * self.single_sensor_dim
                        
                        # --- 命中逻辑 (完全复用) ---
                        if tag == "RL_Agent" and not closest_flags['agent']:
                            start_col = base_col + self.obs_indices['agent']
                            orientation = arena_data['orientation'].get(actor_id, [0, 0])
                            arena_obs[agent_local_idx, start_col:start_col+3] = [hit_distance, orientation[0], orientation[1]]
                            closest_flags['agent'] = True
                        elif tag == "RL_Coin" and not closest_flags['coin']:
                            velocity = arena_data['velocities'].get(actor_id, np.zeros(2))
                            ray_direction = np.array(all_ray_directions[ray_idx])[:2]
                            velocity_projection = np.dot(velocity, ray_direction) / (np.linalg.norm(ray_direction) + 1e-8)
                            start_col = base_col + self.obs_indices['coin']
                            arena_obs[agent_local_idx, start_col:start_col+2] = [hit_distance, velocity_projection]
                            closest_flags['coin'] = True
                        elif tag == "RL_Poison" and not closest_flags['poison']:
                            velocity = arena_data['velocities'].get(actor_id, np.zeros(2))
                            ray_direction = np.array(all_ray_directions[ray_idx])[:2]
                            velocity_projection = np.dot(velocity, ray_direction) / (np.linalg.norm(ray_direction) + 1e-8)
                            start_col = base_col + self.obs_indices['poison']
                            arena_obs[agent_local_idx, start_col:start_col+2] = [hit_distance, velocity_projection]
                            closest_flags['poison'] = True
                        
                        elif tag == "RL_Wall" and not closest_flags['wall']:
                            start_col = base_col + self.obs_indices['wall']
                            arena_obs[agent_local_idx, start_col] = hit_distance
                            closest_flags['wall'] = True
                        
                        elif tag == "RL_Block" and not closest_flags['obstacle']:
                            start_col = base_col + self.obs_indices['obstacle']
                            arena_obs[agent_local_idx, start_col] = hit_distance
                            closest_flags['obstacle'] = True
        # --- 碰撞标记处理 (完全复用) ---
        if arena_data.get('hit_coin'):
            for agent_idx in arena_data['hit_coin']:
                arena_obs[agent_idx, -2] = 1.0
        if arena_data.get('hit_poison'):
            for agent_idx in arena_data['hit_poison']:
                arena_obs[agent_idx, -1] = 1.0
        
        return arena_obs
    
    def process_ray_results_for_single_arena(self, ray_results: List[dict], rays_directions: List, arena_idx: int):
        """
        [新增] 只处理单个Arena的射线检测结果。
        """
        # 调用辅助函数，偏移量为0
        arena_obs = self._process_rays_for_one_arena(
            arena_idx=arena_idx,
            all_ray_results=ray_results,
            all_ray_directions=rays_directions,
            result_offset=0
        )
        
        # 将Numpy数组转换为最终的字典格式
        return {self.agents[i]: arena_obs[i] for i in range(self.n_pursuers)}

    def process_ray_results(self, ray_results: List[dict], rays_directions_map: Dict):
        """
        [修改] 处理来自所有Arenas的射线检测结果。
        """
        observations_all_arenas = [{} for _ in range(self.num_arenas)]
        
        all_ray_directions = rays_directions_map['rays']
        
        for arena_idx, arena_id in enumerate(self.arena_ids):
            # 获取当前arena在全局结果中的偏移量
            result_offset = rays_directions_map['offsets'][arena_idx]
            
            # 调用辅助函数处理这个arena
            arena_obs = self._process_rays_for_one_arena(
                arena_idx=arena_idx,
                all_ray_results=ray_results,
                all_ray_directions=all_ray_directions,
                result_offset=result_offset
            )
            
            # 将Numpy数组转换为最终的字典格式，并存入列表
            observations_all_arenas[arena_idx] = {self.agents[i]: arena_obs[i] for i in range(self.n_pursuers)}
        return observations_all_arenas

    async def _async_step_logic(self, all_actions: List[Dict[str, np.ndarray]]):
        all_move_tasks = []
        
        for arena_idx, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[arena_idx]
            actions = all_actions[arena_idx]

            arena_data['hit_coin'] = {}
            arena_data['hit_poison'] = {}
            arena_data['control_rewards'] = np.zeros(self.n_pursuers)
            arena_data['food_reward'] = np.zeros(self.n_pursuers)
            arena_data['poison_reward'] = np.zeros(self.n_pursuers)

            for i, agent_name in enumerate(self.agents):
                agent_id = self.agent_ids_map[arena_idx][agent_name]
                action_array = np.array(actions[agent_name])
                action_array =  [np.clip(action_array[i],-1,1) for i in range (len(action_array))]

                arena_data['control_rewards'][i] = self.thrust_penalty * (np.linalg.norm(action_array))
                

                
                # print(action_array)
            
                current_pos_tuple = arena_data['pos']['agents'].get(agent_id, (0, 0, 0))
                target_location = ts.Vector3(
                    current_pos_tuple[0] + action_array[0] * action_multiplier,
                    current_pos_tuple[1] + action_array[1] * action_multiplier,
                    current_pos_tuple[2]
                )
                norm = np.linalg.norm(action_array) + 1e-8
                arena_data['orientation'][agent_id] = np.array([action_array[0]/norm, action_array[1]/norm, 0.0])

                coro = ts.UnaryAPI.simple_move_towards(self.conn, actor_id=agent_id, target_location=target_location, timeout=60.0, orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT)
                all_move_tasks.append(run_and_time_task(coro, agent_id))

            for id_list_name in ['coins', 'poisons']:
                speed = self.evader_speed if id_list_name == 'coins' else self.poison_speed
                for actor_id in arena_data[f'ids_of_{id_list_name}']:
                    current_pos_tuple = arena_data['pos'][id_list_name].get(actor_id, (0,0,0))
                    current_velocity = arena_data['velocities'].get(actor_id, np.zeros(2))
                    
                    steering = self.np_random.standard_normal(2)
                    new_velocity = current_velocity + steering * self.steering_strength
                    new_velocity /= np.linalg.norm(new_velocity) + 1e-8
                    final_velocity = new_velocity * speed
                    arena_data['velocities'][actor_id] = final_velocity

                    target_location = ts.Vector3(
                        current_pos_tuple[0] + final_velocity[0] * action_multiplier,
                        current_pos_tuple[1] + final_velocity[1] * action_multiplier,
                        current_pos_tuple[2]
                    )
                    coro = ts.UnaryAPI.simple_move_towards(self.conn, actor_id=actor_id, target_location=target_location, timeout=6.0)
                    all_move_tasks.append(run_and_time_task(coro, actor_id))

        # print(f"  - [Async Step] 并发执行 {len(all_move_tasks)} 个移动任务...")
        all_results_with_timing = await asyncio.gather(*all_move_tasks, return_exceptions=True)
        
        Coin_Settlement=[defaultdict(list) for _ in range(self.num_arenas)]
        Poison_Settlement=[defaultdict(list) for _ in range(self.num_arenas)]
        
        # [修正] 修复硬编码的 '22'
        group_size = self.actors_per_arena
        sorted_groups = []
        for i in range(self.num_arenas):
            group = all_results_with_timing[i * group_size : (i + 1) * group_size]
            sorted_group = sorted(group, key=lambda r: r['end_time'])
            # [修正] 使用动态计算的值进行断言
            if len(sorted_group) != group_size:
                print(f"[警告] Arena {i} 的结果数量不匹配！预期 {group_size}, 得到 {len(sorted_group)}")
            sorted_groups.append(sorted_group)

        # ... 后续的碰撞处理和奖励计算逻辑保持不变 ...
        for arena_idx, sorted_group in enumerate(sorted_groups):
            arena_data = self.arenas_data[arena_idx]
            for res in sorted_group:
                if not isinstance(res, dict) or 'result' not in res: continue # 安全检查
                cur_loc, hit=res['result']
                actor_id=res['agent_id']

                if actor_id in arena_data['ids_of_agents']:
                    arena_data['pos']['agents'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)
                elif actor_id in arena_data['ids_of_coins']:
                    arena_data['pos']['coins'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)
                elif actor_id in arena_data['ids_of_poisons']:
                    arena_data['pos']['poisons'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)
                
                if hit:
                    hit_actor_id_str = convert_bytes_le_to_guid_string(hit["hit_actor"].object_info.id.guid)
                    if actor_id in arena_data['ids_of_agents']:
                        if hit["hit_actor"].tag == "RL_Coin":
                            if len(Coin_Settlement[arena_idx][hit_actor_id_str]) < self.n_coop and actor_id not in Coin_Settlement[arena_idx][hit_actor_id_str]:
                                Coin_Settlement[arena_idx][hit_actor_id_str].append(actor_id)
                        elif hit["hit_actor"].tag == "RL_Poison":
                            if len(Poison_Settlement[arena_idx][hit_actor_id_str]) < 1:
                                Poison_Settlement[arena_idx][hit_actor_id_str].append(actor_id)
                    elif actor_id in arena_data['ids_of_coins']: 
                        if hit["hit_actor"].tag == "RL_Agent":
                            if len(Coin_Settlement[arena_idx][actor_id]) < self.n_coop and hit_actor_id_str not in Coin_Settlement[arena_idx][actor_id]:
                                Coin_Settlement[arena_idx][actor_id].append(hit_actor_id_str)
                        else:
                            old_velocity = arena_data['velocities'].get(actor_id, np.zeros(2))
                            new_velocity = calculate_bounce_velocity_sample(old_velocity, hit['hit_actor'].unit_forward_vector, restitution=1.0)
                            arena_data['velocities'][actor_id] = new_velocity
                    elif actor_id in arena_data['ids_of_poisons']:
                        if hit["hit_actor"].tag == "RL_Agent":
                            if len(Poison_Settlement[arena_idx][actor_id]) < 1:
                                Poison_Settlement[arena_idx][actor_id].append(hit_actor_id_str)
                        else:
                            old_velocity = arena_data['velocities'].get(actor_id, np.zeros(2))
                            new_velocity = calculate_bounce_velocity_sample(old_velocity, hit['hit_actor'].unit_forward_vector, restitution=1.0)
                            arena_data['velocities'][actor_id] = new_velocity

        coins_to_destroy=[[] for _ in range(self.num_arenas)]
        for arena_idx, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[arena_idx]
            for coin, agent_list in Coin_Settlement[arena_idx].items():
                capture_flag=len(agent_list)>=self.n_coop
                for agent in agent_list:
                    if agent not in arena_data['ids_of_agents']: continue
                    agent_idx=arena_data['ids_of_agents'].index(agent)
                    arena_data['food_reward'][agent_idx]+=self.encounter_reward_num
                    arena_data['hit_coin'][agent_idx]=1
                    if capture_flag:    
                        arena_data['food_reward'][agent_idx]+=self.food_reward_num
                
                # 无论是否被捕获，金币都要反弹
                old_velocity = arena_data['velocities'].get(coin, np.zeros(2))
                new_velocity = calculate_bounce_velocity_sample(old_velocity, [1,1,0], restitution=1.0)
                arena_data['velocities'][coin] = new_velocity

                if capture_flag:
                    coins_to_destroy[arena_idx].append(coin)
                    # print(f"Arena {arena_idx} 的金币 {coin} 被捕获，参与的智能体: {agent_list}")

        poisons_to_destroy=[[] for _ in range(self.num_arenas)]
        for arena_idx, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[arena_idx]
            for poison, agent_list in Poison_Settlement[arena_idx].items():
                for agent in agent_list:
                    if agent not in arena_data['ids_of_agents']: continue
                    agent_idx=arena_data['ids_of_agents'].index(agent)
                    arena_data['poison_reward'][agent_idx]+=self.poison_reward_num
                    arena_data['hit_poison'][agent_idx]=1
                poisons_to_destroy[arena_idx].append(poison)
        
        # [修正] 修复硬编码的位置计算
        for arena_idx, arena_data in enumerate(self.arenas_data):
            arena_id = self.arena_ids[arena_idx]
            anchor_loc = self.arena_anchors[arena_id].location

            destroyed_coins_in_this_arena = coins_to_destroy[arena_idx]
            if destroyed_coins_in_this_arena:
                for cid in destroyed_coins_in_this_arena:
                    rand_x, rand_y = generate_safe_random_location(x_bounds, y_bounds, block_ranges)
                    world_loc = (anchor_loc.x + rand_x, anchor_loc.y + rand_y, 200)
                    #这里要改scale
                    awaitset_ok = await ts.UnaryAPI.set_actor_transform(self.context.conn,cid,ts.Transform(location=ts.Vector3(*world_loc), scale=ts.Vector3(coin_scale,coin_scale,coin_scale)))
                    if not awaitset_ok: raise RuntimeError(f"无法重置金币 {cid} 的位置。")                                                                        
                    arena_data['pos']['coins'][cid] = world_loc
                    random_dir = self.np_random.standard_normal(2)
                    random_dir /= np.linalg.norm(random_dir) + 1e-8
                    arena_data['velocities'][cid] = random_dir * self.evader_speed
                    
            destroyed_poisons_in_this_arena = poisons_to_destroy[arena_idx]
            if destroyed_poisons_in_this_arena:
                for pid in destroyed_poisons_in_this_arena:
                    rand_x, rand_y = generate_safe_random_location(x_bounds, y_bounds, block_ranges)
                    world_loc = (anchor_loc.x + rand_x, anchor_loc.y + rand_y, 200)
                    awaitset_ok = await ts.UnaryAPI.set_actor_transform(self.context.conn,pid,ts.Transform(location=ts.Vector3(*world_loc)))
                    if not awaitset_ok: raise RuntimeError(f"无法重置毒药 {pid} 的位置。")                                                                        
                    arena_data['pos']['poisons'][pid] = world_loc
                    random_dir = self.np_random.standard_normal(2)
                    random_dir /= np.linalg.norm(random_dir) + 1e-8
                    arena_data['velocities'][pid] = random_dir * self.poison_speed
        
        final_rewards_all_arenas = []
        for arena_idx, arena_id in enumerate(self.arena_ids):
            arena_data = self.arenas_data[arena_idx]
            control_rewards = arena_data['control_rewards']
            food_reward = arena_data['food_reward']
            poison_reward = arena_data['poison_reward']
            local_rewards = control_rewards + food_reward + poison_reward
            global_reward = local_rewards.mean()
            final_rewards = local_rewards * self.local_ratio + global_reward * (1 - self.local_ratio)
            # if final_rewards[0]<-1.0 or final_rewards[1]<-1.0:
            #     print("=== Step Reward Debug Info ===")
            #     print(f"Arena {arena_idx} (ID: {arena_id})")
            #     print(control_rewards)
            #     print(food_reward)
            #     print(poison_reward)
            #     print(final_rewards)

            final_rewards_all_arenas.append({self.agents[i]: final_rewards[i] for i in range(self.n_pursuers)})
            
        jobs, rays_directions_map = self.build_observation_rays()
        ray_results = await ts.UnaryAPI.multi_line_trace_by_object(self.context.conn, jobs=jobs)
        observations = self.process_ray_results(ray_results, rays_directions_map)

        terminated_all_arenas = [{agent: False for agent in self.agents} for _ in self.arena_ids]
        truncated_all_arenas = []
        for arena_idx in range(len(self.arena_ids)):
            is_truncated = self.arenas_data[arena_idx]['count_step'] >= self.max_cycles
            truncated_all_arenas.append({agent: is_truncated for agent in self.agents})

        info= [{'agent_mask': {agent: True for agent in self.agents}} for _ in self.arena_ids]

        return observations, final_rewards_all_arenas, terminated_all_arenas, truncated_all_arenas, info

    def step(self, actions: List[Dict[str, np.ndarray]]):
        for arena_data in self.arenas_data:
            arena_data['count_step'] += 1
        
        obs, rewards, terminated, truncated, infos = self.context.sync_run(self._async_step_logic(actions))
        return obs, rewards, terminated, truncated, infos
        
    def close(self):
        print("[INFO] 关闭 waterworld 环境...")
        try:
            # self.context.sync_run(ts.UnaryAPI.reset_level(self.conn))
            print("[INFO] 环境已关闭。")
        except Exception as e:
            print(f"[ERROR] 关闭环境时发生错误: {e}")


def proper_training_loop(env, total_steps=3000):
    """
    [修改] 演示处理“矢量化”环境的【正确】训练循环。
    """
    print("[INFO] 开始训练循环，首先调用 env.reset() 进行完全初始化...")
    observations, infos = env.reset() # `reset` 返回批量的观测
    
    for step in range(total_steps):
        actions = []
        for i in range(env.num_arenas):
            arena_actions = {agent: env.action_space[agent].sample() for agent in env.agents}
            actions.append(arena_actions)
        
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)


        if step % 1000 == 0:
            print(f"\n[INFO] 环境 Step {step+1} ...")
            print_memory_usage(ue_process_name_hint='UnrealEditor')
        
        # [修改] 检查是否有任何一个环境结束，并独立重置它
        for i in range(env.num_arenas):
            # 检查这个环境是否结束 (terminated 或 truncated)
            # all() 确保该环境内的所有agent都结束了
            if all(terminateds[i].values()) or all(truncateds[i].values()):
                print(f"[!!!] Arena {i} 的回合结束于第 {step+1} 步. 正在独立重置...")
                
                # 调用新的独立重置函数
                new_obs_for_arena_i, info_for_arena_i = env.reset_one_env(i)
                
                # 将返回的单个新观测，更新到整个观测批次中
                observations[i] = new_obs_for_arena_i
            else:
                # 如果环境没有结束，就用下一步的观测
                observations[i] = next_observations[i]
    
    print("[INFO] 训练循环结束。")


def main():
    print("[INFO] 连接到 TongSim ...")
    try:
        with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
            print("[INFO] 创建并行的 waterworld 环境...")
            env = waterworld(num_arenas=4, max_cycles=1000,n_pursuers=5) # 使用少量并行和较短回合方便演示
            print("[INFO] 环境创建成功！")
            print(f"[INFO] 并行环境数量: {env.num_arenas}")
            
            proper_training_loop(env, total_steps=100000)

    except Exception as e:
        print(f"[ERROR] 演示过程中发生严重异常: {e}")
        traceback.print_exc()
    
    print("[INFO] 演示完成。")


if __name__ == "__main__":
    main()