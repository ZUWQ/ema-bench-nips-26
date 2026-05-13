import math
from utils import convert_bytes_le_to_guid_string, generate_circular_rays,run_and_time_task,calculate_bounce_velocity_sample
import tongsim as ts
from tongsim.core.world_context import WorldContext
from tongsim.type.rl_demo import RLDemoOrientationMode,CollisionObjectType
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import gymnasium as gym
from typing import List
from collections import defaultdict

GRPC_ENDPOINT = "127.0.0.1:5726"


SPAWN_BLUEPRINT_COIN = "/Game/Developer/DemoCoin/BP_DemoCoin.BP_DemoCoin_C"
AGENT_BP = "/Game/Developer/Characters/UE4Mannequin/BP_UE4Mannequin.BP_UE4Mannequin_C"
SPAWN_BLUEPRINT_POISON = "/Game/Developer/DemoCoin/BP_DemoPoision.BP_DemoPoision_C"

# =================================
import asyncio
import random


action_multiplier = 20  # 动作缩放因子，控制最大速度

async def get_actor_transform_safe(conn, actor_id_list):
    """
    安全地获取多个 Actor 的 Transform 信息，忽略不存在的 Actor。
    """
    tasks = [ts.UnaryAPI.get_actor_transform(conn, actor_id) for actor_id in actor_id_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


async def spawn_actors_concurrently(
    context: WorldContext,
    count: int,
    blueprint_path: str,
    base_name: str,
    tags: List[str],
    actor_type_name: str,  # 用于打印日志的描述性名称，例如 "金币" 或 "蝎子"
    spawn_area_center: ts.Vector3,
    spawn_radius: float,
    spawn_z: float
) -> List[dict]:
    """
    并发地在指定区域的随机位置生成多个同类型的Actor。

    Args:
        context: WorldContext 对象，包含连接信息。
        count: 要生成的Actor数量。
        blueprint_path: Actor的蓝图路径字符串。
        base_name: 用于生成唯一名称的基础名称 (例如 "DemoRL_Coin")。
        tags: 要附加到每个Actor的标签列表。
        actor_type_name: 用于在日志中显示的Actor类型名称 (例如 "金币")。
        spawn_area_center: 生成区域的中心点 (Vector3)。
        spawn_radius: 在中心点周围生成的半径。
        spawn_z: 生成的Actor的Z轴高度。

    Returns:
        一个列表，包含成功生成的 actor 信息字典。
    """
    print(f"\n[+] 准备并发生成 {count} 个 {actor_type_name}...")
    
    tasks = []
    
    for i in range(count):
        # 在指定区域内计算随机位置
        rand_x = spawn_area_center.x + random.uniform(-spawn_radius, spawn_radius)
        rand_y = spawn_area_center.y + random.uniform(-spawn_radius, spawn_radius)
        
        # 创建唯一的Actor名称
        actor_name = f"{base_name}_{i}"
        
        # 创建一个 spawn_actor 协程任务
        tasks.append(ts.UnaryAPI.spawn_actor(
            context.conn,
            blueprint=blueprint_path,
            transform=ts.Transform(
                location=ts.Vector3(rand_x, rand_y, spawn_z),
            ),
            # name=actor_name,
            tags=tags,
            timeout=15.0,
        ))

    print(f"  - 已创建 {len(tasks)} 个 {actor_type_name} 生成任务，现在开始并发执行...")
    spawn_results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"  - 所有 {actor_type_name} 生成任务已执行完毕。")

    # 过滤并返回成功生成的结果
    successful_spawns = [res for res in spawn_results if isinstance(res, dict) ]
    return successful_spawns

class waterworld_demo(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(
        self,
        context: WorldContext,
        n_pursuers=2,
        n_evaders=10,
        n_poisons=10,
        n_coop=1,
        n_sensors=30,
        sensor_range=200.0,
        radius=800.0,
        pursuer_max_accel=0.5,
        evader_speed=1,
        poison_speed=1,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        local_ratio=0.9,
        speed_features=True,
        max_cycles=500,
        render_mode=None,
    ):
        """Input keyword arguments.
        n_pursuers: number of agents
        n_evaders: number of food particles present
        n_poisons: number of poisons present
        n_coop: number of agents required to capture a food particle
        n_sensors: number of sensors on each agent
        sensor_range: range of the sensor
        radius: radius of the agent
        pursuer_max_accel: maximum acceleration of the agents
        evader_speed: maximum speed of the food particles
        poison_speed: maximum speed of the poison particles
        poison_reward: reward (or penalty) for getting a poison particle
        food_reward: reward for getting a food particle
        encounter_reward: reward for being in the presence of food
        thrust_penalty: penalty for using thrust (negative value)
        local_ratio: ratio of local observation to global observation (1.0 = all local, 0.0 = all global)
        speed_features: if True, include speed features in observation
        max_cycles: maximum number of cycles per episode
        render_mode: rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        self.context = context
        self.conn = context.conn
        self.n_pursuers = n_pursuers
        self.n_evaders = n_evaders
        self.n_poisons = n_poisons
        self.n_coop = n_coop
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range
        self.radius = radius
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

        self.velocities = {}  # 存储金币和毒药的速度向量 {actor_id: np.array([vx, vy])}
        self.steering_strength = 0.3 # 控制转向的力度，值越小轨迹越平滑

        self.obs_feature_sizes = {
            "agent": 3,   # [distance, orientation_x, orientation_y]
            "coin": 2,    # [distance, velocity_projection]
            "poison": 2,  # [distance, velocity_projection]
            "wall": 1,    # [distance]
            "obstacle": 1 # [distance]
        }
        # 保证一个固定的处理顺序
        self.obs_feature_order = ["agent", "coin", "poison", "wall", "obstacle"]
        self.agents=["pursuer_"+str(i) for i in range(self.n_pursuers)]

        # 2. 动态计算每个特征块的起始索引
        self.obs_indices = {}
        current_index = 0
        for feature_name in self.obs_feature_order:
            self.obs_indices[feature_name] = current_index
            current_index += self.obs_feature_sizes[feature_name]
        
        # 3. 计算单个传感器的总维度
        self.single_sensor_dim = current_index

        self.get_spaces()
        self._seed()
        self.pos = {}  # 用于存储每个 agent 的当前位置
        self.ids_of_coins = []  # 用于存储所有金币的 ID
        self.ids_of_scorpion = []  # 用于存储所有蝎子的 ID
        self.ids_of_agent = []  # 用于存储所有 agent 的 ID
        self.ids_of_poison = []  # 用于存储所有毒药的 ID
        self.orientation = {}  # 用于存储每个 agent 的朝向向量
        self.count_step=0

    def get_spaces(self):
        """Define the action and observation spaces for all of the agents."""
        if self.speed_features:
            obs_dim = self.single_sensor_dim * self.n_sensors + 2
        else:
            obs_dim = 5 * self.n_sensors + 2

        obs_space = spaces.Box(
            low=np.float32(-np.sqrt(2)),
            high=np.float32(np.sqrt(2)),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = [obs_space for i in range(self.n_pursuers)]
        self.action_space = [act_space for i in range(self.n_pursuers)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    async def _async_reset_logic(self):
        """
        [新增] 包含所有异步重置逻辑的私有方法。
        这个方法会被 reset() 通过 sync_run() 调用一次。
        """
        print("  - [Async] 开始执行异步重置任务...")

        # 1) 重置关卡
        print("  - [Async] 1. Resetting level...")
        await ts.UnaryAPI.reset_level(self.conn)

        # 2) 并发生成所有类型的 Actor
        print("  - [Async] 2. Spawning all actors concurrently...")
        
        # 创建所有生成任务
        spawn_tasks = [
            # 金币生成任务
            spawn_actors_concurrently(
                context=self.context, count=self.n_evaders, blueprint_path=SPAWN_BLUEPRINT_COIN,
                base_name="DemoRL_Coin", tags=["RL_Coin"], actor_type_name="金币",
                spawn_area_center=ts.Vector3(550, -2000, 200), spawn_radius=self.radius, spawn_z=200
            ),
            # 毒药生成任务
            spawn_actors_concurrently(
                context=self.context, count=self.n_poisons, blueprint_path=SPAWN_BLUEPRINT_POISON,
                base_name="DemoRL_Poison", tags=["RL_Poison"], actor_type_name="毒药",
                spawn_area_center=ts.Vector3(650, -2000, 200), spawn_radius=self.radius, spawn_z=200
            ),
            # Agent生成任务
            spawn_actors_concurrently(
                context=self.context, count=self.n_pursuers, blueprint_path=AGENT_BP,
                base_name="DemoRL_Agent", tags=["RL_Agent"], actor_type_name="Agent",
                spawn_area_center=ts.Vector3(250, -2000, 200), spawn_radius=self.radius, spawn_z=200
            )
        ]
        
        # 并发执行所有生成任务
        coin_results, poison_results, agent_results = await asyncio.gather(*spawn_tasks)
        print("  - [Async] 所有 Actor 生成完毕。")

        # 3) 处理生成结果并获取初始 Transform
        print("  - [Async] 3. Processing spawn results and fetching initial transforms...")
        self.ids_of_coins = [coin["id"] for coin in coin_results]
        self.ids_of_poison = [poison["id"] for poison in poison_results]
        self.ids_of_agent = [agent["id"] for agent in agent_results]

        # 创建获取 Transform 的任务
        transform_tasks = [
            get_actor_transform_safe(self.conn, self.ids_of_coins),
            get_actor_transform_safe(self.conn, self.ids_of_poison),
            get_actor_transform_safe(self.conn, self.ids_of_agent)
        ]
        
        # 并发执行所有获取 Transform 的任务
        coin_positions, poison_positions, agent_positions = await asyncio.gather(*transform_tasks)
        print("  - [Async] 所有初始 Transform 获取完毕。")

        # 4) 初始化 Actor 状态（位置、速度、朝向）
        print("  - [Async] 4. Initializing actor states (position, velocity, orientation)...")
        self.pos['coins'] = {}
        for coin_id, pos in zip(self.ids_of_coins, coin_positions):
            if isinstance(pos, ts.Transform):
                self.pos['coins'][coin_id] = (pos.location.x, pos.location.y, pos.location.z)
                random_dir = self.np_random.random(2) * 2 - 1
                random_dir /= np.linalg.norm(random_dir) + 1e-8
                self.velocities[coin_id] = random_dir * self.evader_speed

        self.pos['poisons'] = {}
        for poison_id, pos in zip(self.ids_of_poison, poison_positions):
            if isinstance(pos, ts.Transform):
                self.pos['poisons'][poison_id] = (pos.location.x, pos.location.y, pos.location.z)
                random_dir = self.np_random.random(2) * 2 - 1
                random_dir /= np.linalg.norm(random_dir) + 1e-8
                self.velocities[poison_id] = random_dir * self.poison_speed

        self.pos['agents'] = {}
        for agent_id, pos in zip(self.ids_of_agent, agent_positions):
            if isinstance(pos, ts.Transform):
                self.pos['agents'][agent_id] = (pos.location.x, pos.location.y, pos.location.z)
                self.orientation[agent_id] = np.array([1.0, 0.0, 0.0]) # 初始朝向X轴正方向

        # 5) 构建并返回初始观测
        print("  - [Async] 5. Building initial observation...")
        jobs, rays_directions = self.build_observation_rays()
        ray_results = await ts.UnaryAPI.multi_line_trace_by_object(self.context.conn, jobs=jobs)
        observation = self.process_ray_results(ray_results, rays_directions)
        
        print("  - [Async] 异步重置任务完成。")
        return observation

    def reset(self, *, seed=None, options=None):
        """
        [重构] 重置环境到初始状态，并返回符合 Gymnasium API 的初始观测和信息。
        """
        super().reset(seed=seed) # 处理随机种子
        
        print("\n================== [ENV RESET] ==================")
        
        # 1. 清理所有同步状态变量
        print("[1] Clearing internal states...")
        self.pos = {}
        self.ids_of_coins = []
        self.ids_of_scorpion = [] # 即使未使用，也保持清理
        self.ids_of_agent = []
        self.ids_of_poison = []
        self.orientation = {}
        self.velocities = {}
        self.count_step = 0
        self.hit_coin = {}
        self.hit_poison = {}

        # 2. 单次调用 sync_run 来执行所有异步重置逻辑
        print("[2] Running async reset logic...")
        initial_observation = self.context.sync_run(self._async_reset_logic())
        
        print("================== [RESET DONE] ==================\n")

        # 3. 返回符合 Gymnasium API 标准的元组 (observation, info)
        info = {} # info 字典可用于传递调试信息，此处为空
        return initial_observation, info

    def process_ray_results(self, ray_results, rays_directions):
        observations = np.zeros((self.n_pursuers, self.observation_space[0].shape[0]), dtype=np.float32)
        rays_per_agent = self.n_sensors
        
        for i in range(self.n_pursuers):
            for j in range(rays_per_agent):
                closest_agent = False
                closest_coin = False
                closest_poison = False
                closest_wall = False
                closest_obstacle = False

                ray_idx = i * rays_per_agent + j
                ray = ray_results[ray_idx]
                
                if len(ray['hits']) > 0:
                    for hit in ray['hits']:
                        hit_distance = hit['distance'] / self.sensor_range  # 归一化距离
                        actor_state = hit['actor_state']
                        actor_id = actor_state["id"]
                        tag = actor_state["tag"]
                        
                        # 计算当前传感器的观测值在总观测向量中的起始列
                        base_col = j * self.single_sensor_dim
                        
                        if tag == "RL_Agent" and not closest_agent:
                            start_col = base_col + self.obs_indices['agent']
                            end_col = start_col + self.obs_feature_sizes['agent']
                            # 确保切片长度为3
                            
                            observations[i, start_col:end_col] = [hit_distance, self.orientation[actor_id][0], self.orientation[actor_id][1]]
                            closest_agent = True

                        elif tag == "RL_Coin" and not closest_coin:
                            velocity = self.velocities[actor_id]
                            ray_direction = np.array(rays_directions[ray_idx])[:2]
                            velocity_projection = np.dot(velocity, ray_direction) / (np.linalg.norm(ray_direction) + 1e-8)# 计算速度在射线方向上的投影

                            start_col = base_col + self.obs_indices['coin']
                            end_col = start_col + self.obs_feature_sizes['coin']
                            
                            observations[i, start_col:end_col] = [hit_distance, velocity_projection]
                            closest_coin = True

                        elif tag == "RL_Poison" and not closest_poison:
                            velocity = self.velocities[actor_id]
                            ray_direction = np.array(rays_directions[ray_idx])[:2]
                            velocity_projection = np.dot(velocity, ray_direction) / (np.linalg.norm(ray_direction) + 1e-8)# 计算速度在射线方向上的投影

                            start_col = base_col + self.obs_indices['poison']
                            end_col = start_col + self.obs_feature_sizes['poison']
                            
                            observations[i, start_col:end_col] = [hit_distance, velocity_projection]
                            closest_poison = True

                        elif tag == "RL_Wall" and not closest_wall:
                            start_col = base_col + self.obs_indices['wall']
                            end_col = start_col + self.obs_feature_sizes['wall']
                            
                            observations[i, start_col:end_col] = [hit_distance]
                            closest_wall = True

                        elif tag == "RL_Block" and not closest_obstacle:
                            start_col = base_col + self.obs_indices['obstacle']
                            end_col = start_col + self.obs_feature_sizes['obstacle']

                            observations[i, start_col:end_col] = [hit_distance]
                            closest_obstacle = True
                else:
                    pass  # 没有命中，保持为零

        if self.hit_coin:
            for agent_idx in self.hit_coin:
                observations[agent_idx, -2] = 1.0  # 标记吃到金币
        if self.hit_poison:
            for agent_idx in self.hit_poison:
                observations[agent_idx, -1] = 1.0  # 标记吃到毒药
                
        return observations

    async def _async_step_logic(self, actions):
        """
        这个新的异步函数包含了单步中所有的异步逻辑。
        它会被 sync_run 一次性执行。
        """
        # print(await ts.UnaryAPI.get_actor_transform(self.context.conn, self.ids_of_agent[0]))
        self.hit_coin={}
        self.hit_poison={}
        # 移动 agent
        task=[]
        self.control_rewards = np.zeros(self.n_pursuers)
        self.food_reward=np.zeros(self.n_pursuers)
        self.poison_reward=np.zeros(self.n_pursuers)
        for i,agent_id in enumerate(self.ids_of_agent):
            current_pos_tuple = self.pos['agents'].get(agent_id, (0, 0, 0)) 
            # --- 1. 移动智能体 (Agent) ---
            action_array = np.array(actions[i])
            accel_penalty = self.thrust_penalty * math.sqrt((action_array**2).sum())
            penalty_distribution = (
                (accel_penalty / self.n_pursuers)
                * np.ones(self.n_pursuers)
                * (1 - 1)
            )
            penalty_distribution[i] += accel_penalty * self.local_ratio
            self.control_rewards += penalty_distribution.tolist()

            target_location = ts.Vector3(
                current_pos_tuple[0] + actions[i][0] * action_multiplier,
                current_pos_tuple[1] + actions[i][1] * action_multiplier,
                current_pos_tuple[2]
            )
            self.orientation[agent_id] = np.array([actions[i][0], actions[i][1], 0.0])/np.linalg.norm(np.array([actions[i][0], actions[i][1], 0.0])+1e-8)
            coro=ts.UnaryAPI.simple_move_towards(
                self.conn,
                actor_id=self.ids_of_agent[i],
                target_location=target_location,
                orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
                timeout=60.0,
            )
            task.append(run_and_time_task(coro,self.ids_of_agent[i])) 

        # 移动金币与毒药
        for coin_id in self.ids_of_coins:
            current_pos_tuple = self.pos['coins'].get(coin_id, (0, 0, 0))
            current_velocity = self.velocities.get(coin_id, np.zeros(2))

            # 计算平滑转向
            steering = self.np_random.random(2) * 2 - 1 # 随机转向向量
            new_velocity = current_velocity + steering * self.steering_strength
            new_velocity /= np.linalg.norm(new_velocity) + 1e-8 # 归一化
            final_velocity = new_velocity * self.evader_speed
            self.velocities[coin_id] = final_velocity # 更新速度

            target_location = ts.Vector3(
                current_pos_tuple[0] + final_velocity[0] * action_multiplier,
                current_pos_tuple[1] + final_velocity[1] * action_multiplier,
                current_pos_tuple[2]
            )
            coro = ts.UnaryAPI.simple_move_towards(
                self.conn,
                actor_id=coin_id,
                target_location=target_location,
                orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
                timeout=6.0,
            )
            task.append(run_and_time_task(coro, coin_id))
        for poison_id in self.ids_of_poison:
            current_pos_tuple = self.pos['poisons'].get(poison_id, (0, 0, 0))
            current_velocity = self.velocities.get(poison_id, np.zeros(2))
            
            # 计算平滑转向
            steering = self.np_random.random(2) * 2 - 1
            new_velocity = current_velocity + steering * self.steering_strength
            new_velocity /= np.linalg.norm(new_velocity) + 1e-8
            final_velocity = new_velocity * self.poison_speed
            self.velocities[poison_id] = final_velocity

            target_location = ts.Vector3(
                current_pos_tuple[0] + final_velocity[0] * action_multiplier,
                current_pos_tuple[1] + final_velocity[1] * action_multiplier,
                current_pos_tuple[2]
            )
            coro = ts.UnaryAPI.simple_move_towards(
                self.conn,
                actor_id=poison_id,
                target_location=target_location,
                orientation_mode=RLDemoOrientationMode.ORIENTATION_FACE_MOVEMENT,
                timeout=60.0,
            )
            task.append(run_and_time_task(coro, poison_id))

        results_with_timing=await asyncio.gather(*task,return_exceptions=True)
        # 1. 按照完成时间对结果进行排序，以了解完成顺序
        completion_order = sorted(results_with_timing, key=lambda r: r['end_time'])
        # completion_order = []
        Coin_Settlement=defaultdict(list)
        Poison_Settlement=defaultdict(list)
        # print("--- 任务完成顺序 ---")
        for i, res in enumerate(completion_order):
            # print(f"第 {i+1} 个完成: Agent ID {res['agent_id']}, 耗时 {res['duration']:.4f} 秒, 状态: {res['status']}")
            #是否是agent的移动
            cur_loc, hit=res['result']
            actor_id=res['agent_id']

            if actor_id in self.ids_of_agent:
                self.pos['agents'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)
            elif actor_id in self.ids_of_coins:
                self.pos['coins'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)
            elif actor_id in self.ids_of_poison:
                self.pos['poisons'][actor_id] = (cur_loc.x, cur_loc.y, cur_loc.z)

            if hit:
                hit_actor_id_str = convert_bytes_le_to_guid_string(hit["hit_actor"].object_info.id.guid)
                # Agent 撞到东西
                # print(hit)
                if actor_id in self.ids_of_agent:
                    if hit["hit_actor"].tag == "RL_Coin":
                        if len(Coin_Settlement[hit_actor_id_str]) < self.n_coop and actor_id not in Coin_Settlement[hit_actor_id_str]:
                            Coin_Settlement[hit_actor_id_str].append(actor_id)
                    elif hit["hit_actor"].tag == "RL_Poison":
                        if len(Poison_Settlement[hit_actor_id_str]) < 1:
                            Poison_Settlement[hit_actor_id_str].append(actor_id)
                # 金币/毒药撞到 Agent
                elif actor_id in self.ids_of_coins: 
                    if hit["hit_actor"].tag == "RL_Agent":
                        if len(Coin_Settlement[actor_id]) < self.n_coop and hit_actor_id_str not in Coin_Settlement[actor_id]:
                            Coin_Settlement[actor_id].append(hit_actor_id_str)
                    else:
                        # 撞到墙壁，计算反弹
                        # print(f"Coin {actor_id} 撞到墙壁，计算反弹")
                        old_velocity = self.velocities.get(actor_id, np.zeros(2))
                        # print(f"  - 撞前速度: {old_velocity}")
                        # print(f"  - 撞击法线: {[hit['hit_actor'].unit_forward_vector.x, hit['hit_actor'].unit_forward_vector.y]}")
                        new_velocity = calculate_bounce_velocity_sample(old_velocity, hit['hit_actor'].unit_forward_vector, restitution=1.0)
                        # print(f"  - 撞后速度: {new_velocity}")
                        self.velocities[actor_id] = new_velocity

                elif actor_id in self.ids_of_poison:
                    if hit["hit_actor"].tag == "RL_Agent":
                        if len(Poison_Settlement[actor_id]) < 1:
                            Poison_Settlement[actor_id].append(hit_actor_id_str)
                    else:
                        # 撞到墙壁，计算反弹
                        old_velocity = self.velocities.get(actor_id, np.zeros(2))
                        # print(f"  - 撞前速度: {old_velocity}")
                        # print(f"  - 撞击墙壁的法线向量: {hit['hit_actor'].unit_forward_vector}")
                        new_velocity = calculate_bounce_velocity_sample(old_velocity, hit['hit_actor'].unit_forward_vector, restitution=1.0)
                        # print(f"  - 撞后速度: {new_velocity}")
                        self.velocities[actor_id] = new_velocity

        coins_to_destroy=[]
        for coin, agent_list in Coin_Settlement.items():
            capture_flag=len(agent_list)>=self.n_coop
            for agent in agent_list:
                agent_idx=self.ids_of_agent.index(agent)
                self.food_reward[agent_idx]+=self.encounter_reward_num
                self.hit_coin[agent_idx]=1#标记
                if capture_flag:    
                    self.food_reward[agent_idx]+=self.food_reward_num
            if capture_flag:
                coins_to_destroy.append(coin)

        poisons_to_destroy=[]
        for poison, agent_list in Poison_Settlement.items():
            for agent in agent_list:
                agent_idx=self.ids_of_agent.index(agent)
                self.poison_reward[agent_idx]+=self.poison_reward_num
                self.hit_poison[agent_idx]=1#标记
            poisons_to_destroy.append(poison)
        
        #销毁金币与毒药
        destroy_tasks= [ts.UnaryAPI.destroy_actor(self.conn, cid) for cid in coins_to_destroy]
        destroy_tasks += [ts.UnaryAPI.destroy_actor(self.conn, pid) for pid in poisons_to_destroy]
        if destroy_tasks:
            await asyncio.gather(*destroy_tasks, return_exceptions=True)
            print(f"  - 销毁了 {len(destroy_tasks)} 个物体（金币和毒药）。")
            self.ids_of_coins = [cid for cid in self.ids_of_coins if cid not in coins_to_destroy]
            self.ids_of_poison = [pid for pid in self.ids_of_poison if pid not in poisons_to_destroy]

            for cid in coins_to_destroy: del self.velocities[cid]; del self.pos['coins'][cid]
            for pid in poisons_to_destroy: del self.velocities[pid]; del self.pos['poisons'][pid]

        # 生成数量与销毁数量相同的金币与毒药
        if coins_to_destroy:
            new_coin_results=await spawn_actors_concurrently(
                context=self.context,
                count=len(coins_to_destroy),
                blueprint_path=SPAWN_BLUEPRINT_COIN,
                base_name="DemoRL_Coin",
                tags=["RL_Coin"],
                actor_type_name="金币",
                spawn_area_center=ts.Vector3(650, -2000, 200),
                spawn_radius=self.radius,
                spawn_z=200
            )
            new_coin_ids = [coin["id"] for coin in new_coin_results]
            self.ids_of_coins.extend(new_coin_ids)
            new_coin_positions = await get_actor_transform_safe(self.conn, new_coin_ids)
            for coin_id, pos in zip(new_coin_ids, new_coin_positions):
                self.pos['coins'][coin_id] = (pos.location.x, pos.location.y, pos.location.z)
                random_dir = self.np_random.random(2) * 2 - 1
                random_dir /= np.linalg.norm(random_dir) + 1e-8
                self.velocities[coin_id] = random_dir * self.evader_speed
            print(f"  - 生成了 {len(new_coin_ids)} 个新的金币。")
        if poisons_to_destroy:
            new_poison_results=await spawn_actors_concurrently(
                context=self.context,
                count=len(poisons_to_destroy),
                blueprint_path=SPAWN_BLUEPRINT_POISON,
                base_name="DemoRL_Poison",
                tags=["RL_Poison"],
                actor_type_name="毒药",
                spawn_area_center=ts.Vector3(1100, -2000, 100),
                spawn_radius=self.radius,
                spawn_z=200
            )
            new_poison_ids = [poison["id"] for poison in new_poison_results]
            self.ids_of_poison.extend(new_poison_ids)
            new_poison_positions = await get_actor_transform_safe(self.conn, new_poison_ids)
            for poison_id, pos in zip(new_poison_ids, new_poison_positions):
                self.pos['poisons'][poison_id] = (pos.location.x, pos.location.y, pos.location.z)
                random_dir = self.np_random.random(2) * 2 - 1
                random_dir /= np.linalg.norm(random_dir) + 1e-8
                self.velocities[poison_id] = random_dir * self.poison_speed
            print(f"  - 生成了 {len(new_poison_ids)} 个新的毒药。")
        
        #计算奖励
        local_rewards = self.control_rewards + self.food_reward + self.poison_reward
        global_reward = local_rewards.mean()
        final_rewards = local_rewards * self.local_ratio + global_reward * (1 - self.local_ratio)
               
        #构建观察
        jobs,ray_directions=self.build_observation_rays()
        ray_results=await ts.UnaryAPI.multi_line_trace_by_object(self.context.conn,jobs=jobs)
        observation=self.process_ray_results(ray_results,ray_directions)

        terminated = False  # 在这个逻辑中，没有明确的“成功/失败”终止条件
        truncated = self.count_step >= self.max_cycles # 达到最大步数是截断

        # 返回 gym.step 所需的完整元组
        return observation, final_rewards.tolist(), terminated, truncated, {}
        #

    def step(self, actions):
        """
        同步的 step 函数，现在它只负责调用 sync_run。
        """
        # 调用 sync_run 来执行我们打包好的异步逻辑函数
        # 所有 await 调用都在 _async_step_logic 内部，由 sync_run 管理的事件循环来处理
        self.count_step+=1
        observation, rewards, terminated, truncated, info =self.context.sync_run(self._async_step_logic(actions))
        return observation, rewards, terminated, truncated, info
        

    def build_observation_rays(self):
        """构建所有智能体的观察向量列表。"""
        jobs=[]
        all_generated_rays = []
        for i,agent_id in enumerate(self.ids_of_agent):
            face_vector=self.orientation[agent_id]
            start_point=self.pos['agents'][agent_id]
            # start_point=(start_point[0]+50,start_point[1],start_point[2])
            generated_rays=generate_circular_rays(face_vector,self.n_sensors,radius=self.sensor_range)
            all_generated_rays.extend(generated_rays)
            for ray in generated_rays:
                jobs.append({
                "start": ts.Vector3(*start_point), 
                "end": ts.Vector3(*(start_point + ray)),
                "object_types": [
                    CollisionObjectType.OBJECT_WORLD_STATIC, 
                    CollisionObjectType.OBJECT_WORLD_DYNAMIC, 
                    CollisionObjectType.OBJECT_PAWN, 
                ],
                "actors_to_ignore": [agent_id],# 忽略自己
                })
        return jobs, all_generated_rays


def proper_training_loop(env, total_steps=3000):
    """一个演示正确处理 reset 的训练循环。"""
    
    # 1. 在循环开始前，必须先 reset 一次
    print("[INFO] 开始训练循环，首先调用 env.reset()...")
    observation, info = env.reset()
    
    for step in range(total_steps):
        print(f"\n[INFO] 环境 Step {step+1} ...")
        
        # 2. 根据当前 observation 生成动作 (这里仍然使用随机动作)
        actions = [env.action_space[i].sample() for i in range(env.n_pursuers)]
        
        # 3. 执行一步，并获取所有返回值
        observation, rewards, terminated, truncated, info = env.step(actions)
        
        # 4. 关键步骤：检查回合是否结束
        if terminated or truncated:
            print(f"[!!!] 回合结束于第 {step+1} 步. 原因: {'Terminated' if terminated else 'Truncated'}")
            print("[INFO] 调用 env.reset() 来开始新回合...")
            
            # 5. 如果结束，就调用 reset，并用新的 observation 开始下一轮
            observation, info = env.reset()    
    

def main():
    """主入口：建立连接，创建Gym环境并运行。"""
    print("[INFO] 连接到 TongSim ...")
    try:
        # 1. 使用 'with' 语句创建并管理 TongSim 连接。
        #    这会保证程序结束时连接被正确关闭。
        with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
            
            # 2. 将已经存在的 ue.context 对象直接传递给环境的构造函数。
            #    这个 ue.context 正是环境所需要的。
            print("[INFO] 创建 waterworld_demo 环境...")
            env = waterworld_demo(context=ue.context)
            print("[INFO] 环境创建成功！")
            print(f"[INFO] Action Space: {env.action_space}")
            print(f"[INFO] Observation Space: {env.observation_space}")
            print(f"[INFO] 共有 {env.n_pursuers} 个智能体 (Agents)。")
            print(f"[INFO] 共有 {env.n_evaders} 个金币 (Coins)。")
            print(f"[INFO] 共有 {env.n_poisons} 个毒药 (Poisons)。")
            print(f"[INFO] 每个智能体有 {env.n_sensors} 个传感器 (Sensors)，每个传感器的范围为 {env.sensor_range} 单位。")
            proper_training_loop(env)

    except Exception as e:
        # 捕获并打印任何可能发生的异常，方便调试。
        import traceback
        print(f"[ERROR] 演示过程中发生严重异常: {e}")
        traceback.print_exc()
    
    print("[INFO] 演示完成。")


if __name__ == "__main__":
    main()
