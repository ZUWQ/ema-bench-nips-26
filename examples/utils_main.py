import uuid
import time
import tongsim as ts
import numpy as np
import psutil
import os


import random
from typing import List, Tuple, Optional
def _is_location_in_block_ranges(x: float, y: float, block_ranges: List[List[List[float]]]) -> bool:
    """
    [辅助函数] 检查给定的 (x, y) 坐标是否落在任何一个禁区矩形内。
    
    Args:
        x: 要检查的 x 坐标。
        y: 要检查的 y 坐标。
        block_ranges: 禁区列表，每个禁区是 [[x_min, x_max], [y_min, y_max]]。
    
    Returns:
        如果坐标在禁区内，返回 True，否则返回 False。
    """
    for block in block_ranges:
        x_range, y_range = block[0], block[1]
        # 检查坐标是否同时在 x 和 y 的范围内
        if (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1]):
            return True  # 坐标在禁区内
    return False  # 坐标是安全的
def generate_safe_random_location(
    x_bounds: Tuple[float, float], 
    y_bounds: Tuple[float, float], 
    block_ranges: List[List[List[float]]], 
    max_retries: int = 100
) -> Optional[Tuple[float, float]]:
    """
    [新增封装函数] 在指定边界内生成一个不在任何禁区内的随机坐标。
    
    采用“拒绝采样”方法：随机生成一个点，如果它在禁区内，就抛弃并重试。
    
    Args:
        x_bounds: 一个元组 (min_x, max_x)，定义 x 坐标的生成范围。
        y_bounds: 一个元组 (min_y, max_y)，定义 y 坐标的生成范围。
        block_ranges: 禁区列表。
        max_retries: 最大尝试次数，以防止因空间不足而导致的无限循环。
        
    Returns:
        一个包含 (x, y) 的元组，如果成功找到。
        如果在 max_retries 次尝试后仍未找到，则返回 None。
    """
    for _ in range(max_retries):
        # 1. 在指定边界内生成一个随机的候选坐标
        rand_x = random.uniform(x_bounds[0], x_bounds[1])
        rand_y = random.uniform(y_bounds[0], y_bounds[1])
        
        # 2. 检查该坐标是否在禁区内
        if not _is_location_in_block_ranges(rand_x, rand_y, block_ranges):
            # 3. 如果坐标是安全的，立即返回结果
            return rand_x, rand_y
            
    # 如果循环结束仍未返回，说明所有尝试都失败了
    print(f"Warning: Failed to find a safe location after {max_retries} attempts.")
    return None

def print_memory_usage(ue_process_name_hint='UnrealEditor'):
    """
    打印当前系统内存、GPU显存、当前Python进程以及Unreal Engine进程的内存占用。

    Args:
        ue_process_name_hint (str, optional): UE项目可执行文件名的部分或完整名称，
                                            用于在进程列表中查找UE进程。
                                            例如，如果你的游戏是 "MyGame.exe"，可以传入 "MyGame"。
                                            默认为 None，将只查找通用的UE编辑器进程。
    """
    # 确保必要的库已导入
    print(f"\n{'='*60}")
    
    # 1. 打印系统内存信息
    mem = psutil.virtual_memory()
    print(f"[System Memory] Total: {mem.total / (1024**3):.2f} GB, "
          f"Available: {mem.available / (1024**3):.2f} GB, "
          f"Used: {mem.used / (1024**3):.2f} GB, "
          f"Percent: {mem.percent:.1f}%")
    
    # 2. 打印当前Python进程的内存占用
    process = psutil.Process(os.getpid())
    process_mem = process.memory_info()
    print(f"[Current Python Process] PID: {os.getpid()}, "
          f"RSS: {process_mem.rss / (1024**3):.2f} GB, "
          f"VMS: {process_mem.vms / (1024**3):.2f} GB, "
          f"Percent: {process.memory_percent():.2f}%")
    
    # 3. 打印子进程的内存占用（如果有多进程）
    children = process.children(recursive=True)
    if children:
        total_children_mem = sum([child.memory_info().rss for child in children])
        print(f"[Child Processes] Count: {len(children)}, "
              f"Total RSS: {total_children_mem / (1024**3):.2f} GB")
        # 只显示内存占用最高的前5个子进程
        sorted_children = sorted(children, key=lambda p: p.memory_info().rss, reverse=True)
        for i, child in enumerate(sorted_children[:5]):
            try:
                child_mem = child.memory_info()
                print(f"  - Child {i+1} (PID: {child.pid}, Name: {child.name()}): "
                      f"RSS: {child_mem.rss / (1024**3):.2f} GB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue # 如果子进程在获取信息时已退出，则跳过
        if len(children) > 5:
            print(f"  ... and {len(children) - 5} more child processes")

    # 4. 新增：查找并打印Unreal Engine进程的内存占用
    # 定义要查找的UE进程名称列表（不区分大小写）
    target_names = ["unrealeditor", "ue4editor"] # 通用编辑器名称
    if ue_process_name_hint:
        # 如果用户提供了项目名，也加入到查找列表
        target_names.append(ue_process_name_hint.lower())

    found_ue_processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            proc_name_lower = proc.info['name'].lower()
            # 使用 startswith 来匹配，例如 "MyGame" 可以匹配 "MyGame.exe" 和 "MyGame-Win64-Shipping.exe"
            if any(proc_name_lower.startswith(name) for name in target_names):
                found_ue_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 某些系统进程可能无法访问，或者进程在迭代时已消失，忽略即可
            continue
            
    if found_ue_processes:
        print("[Unreal Engine Processes]")
        for ue_proc in found_ue_processes:
            try:
                ue_mem = ue_proc.memory_info()
                print(f"  - Process: {ue_proc.name()} (PID: {ue_proc.pid}), "
                      f"RSS: {ue_mem.rss / (1024**3):.2f} GB, "
                      f"VMS: {ue_mem.vms / (1024**3):.2f} GB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # 如果进程在获取内存信息时已退出
                print(f"  - Process: {ue_proc.name()} (PID: {ue_proc.pid}) - Could not retrieve info (process may have exited).")
    else:
        print("[Unreal Engine Processes] Not found. (Hint: pass ue_process_name_hint='YourGameName')")

    # # 5. 打印GPU显存信息
    # if torch.cuda.is_available():
    #     for i in range(torch.cuda.device_count()):
    #         allocated = torch.cuda.memory_allocated(i) / (1024**3)
    #         reserved = torch.cuda.memory_reserved(i) / (1024**3)
    #         total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    #         print(f"[GPU {i}] Total: {total:.2f} GB, "
    #               f"Allocated: {allocated:.2f} GB, "
    #               f"Reserved: {reserved:.2f} GB, "
    #               f"Free (within reserved): {(reserved - allocated):.2f} GB, "
    #               f"Free (total): {total - reserved:.2f} GB")
    # else:
    #     print("[GPU] No CUDA device available")
        
    print(f"{'='*60}\n")

def calculate_bounce_velocity_sample(
    velocity_vector: np.ndarray, 
    wall_normal_vector: ts.Vector3, # 这个参数暂时用不到，但保留以方便日后切换
    restitution: float = 1.0,
    scatter_strength: float = 0.4 # 随机扰动强度 (建议值 0.2 ~ 0.4)
) -> np.ndarray:
    """
    [临时模拟方案] 引入随机扰动来处理反弹，避免在没有精确法线时卡死。
    这个函数将在周二功能实现后被物理精确的版本替换。

    :param velocity_vector: 物体当前的2D速度向量。
    :param wall_normal_vector: 未使用的参数，保持函数签名一致以便后续替换。
    :param restitution: 弹力系数，模拟能量损失。
    :param scatter_strength: 随机扰动的强度。0.0表示纯粹的速度反向，1.0表示反弹方向完全随机。
    :return: 反弹后的新2D速度向量。
    """
    # 1. 基本的速度反向作为基础方向
    base_reflection = -velocity_vector

    # 2. 创建一个随机的2D方向向量作为扰动
    # np.random.standard_normal(2) 会生成一个服从标准正态分布的二维向量
    random_scatter = np.random.standard_normal(2)
    norm_scatter = np.linalg.norm(random_scatter)
    if norm_scatter > 1e-6:
        random_scatter /= norm_scatter # 归一化为单位向量，只保留方向

    # 3. 将基础反射和随机散射按权重结合起来
    # (1 - scatter_strength) 保证了主要方向还是反弹
    # scatter_strength 引入了逃离振荡的随机性
    new_velocity_direction = (1 - scatter_strength) * base_reflection + scatter_strength * random_scatter
    
    # 4. 归一化最终的混合方向，并应用原始速度大小和能量损失
    norm_new_velocity = np.linalg.norm(new_velocity_direction)
    if norm_new_velocity < 1e-6:
        # 如果结果向量意外为0，则直接使用随机向量作为方向
        final_velocity = random_scatter * np.linalg.norm(velocity_vector) * restitution
    else:
        # 使用混合后的方向，保持碰撞前的速率，并乘以弹力系数
        final_velocity = (new_velocity_direction / norm_new_velocity) * np.linalg.norm(velocity_vector) * restitution

    return final_velocity

def calculate_bounce_velocity(
    velocity_vector: np.ndarray, 
    wall_normal_vector: ts.Vector3,
    restitution: float = 1.0  # 弹力系数 (1.0表示完美反弹)
) -> np.ndarray:
    """
    根据入射速度和墙壁法线计算反弹后的速度向量。

    Args:
        velocity_vector (np.ndarray): 物体撞墙前的速度向量 (例如 [vx, vy])。
        wall_normal_vector (ts.Vector3): 墙壁在碰撞点的表面法线向量 (由引擎提供)。
        restitution (float): 弹力系数，1.0为完美反弹，<1.0会有能量损失。

    Returns:
        np.ndarray: 反弹后的新速度向量。
    """
    # 1. 将 tongsim 的 Vector3 转换为 numpy 数组，并只取 XY 分量。
    #    【修正】直接使用引擎提供的法线，不再进行反转。
    wall_normal = np.array([wall_normal_vector.x, wall_normal_vector.y])
    
    # 2. 确保法线向量是单位向量，以保证计算的准确性。
    norm_of_normal = np.linalg.norm(wall_normal)
    if norm_of_normal < 1e-8:
        # 如果法线向量几乎为零，则无法计算，直接返回原速度避免崩溃。
        return velocity_vector
    wall_normal /= norm_of_normal

    # 3. 计算速度向量与表面法线的点积。
    dot_product = np.dot(velocity_vector, wall_normal)

    # 4. 【修正】如果点积大于或等于0，说明物体正在远离或平行于墙面，不应发生反弹。
    #    只有当点积为负（即物体正撞向墙面）时，才继续计算。
    if dot_product >= 0:
        return velocity_vector

    # 5. 应用标准的反弹公式: V_new = V_old - 2 * dot(V_old, N) * N
    bounce_vector = velocity_vector - 2 * dot_product * wall_normal
    
    # 6. 应用弹力系数，模拟能量损失，并返回最终结果。
    return bounce_vector * restitution

async def run_and_time_task(coro,agent_id):
    """
    一个包装协程，用于执行任务、计时并返回结构化结果。
    coro: The coroutine to run (e.g., simple_move_towards).
    agent_id: The ID of the agent this task belongs to.
    task_index: The original index in the task list.
    """
    start_time = time.monotonic()
    try:
        # 执行原始的异步任务
        original_result = await coro
        return {
            "agent_id":agent_id,
            "status": "completed",
            "result": original_result,
            "end_time": time.monotonic(),
            "duration": time.monotonic() - start_time,
        }
    except Exception as e:
        # 如果任务本身抛出异常，也进行记录
        return {
            "agent_id":agent_id,
            "status": "failed",
            "error": e,
            "end_time": time.monotonic(),
            "duration": time.monotonic() - start_time,
        }

def convert_bytes_le_to_guid_string(guid_bytes_le: bytes) -> str:
    """
    将小端序（little-endian）的16字节GUID转换为标准的大写连字符分隔字符串格式。

    Args:
        guid_bytes_le: 16字节的bytes对象，代表小端序的GUID。
                       例如: b'q\\x15\\xa8\\xe0dC\\xef\\x92\\x9b\\x16?\\x94x\\x0fZ\\x87'

    Returns:
        一个标准的GUID字符串。
        例如: 'E0A81571-4364-92EF-9B16-3F94780F5A87'
    """
    # 1. 使用`bytes_le`参数从字节数据创建UUID对象
    #    这会自动处理所有字节序（endianness）的转换问题。
    uuid_obj = uuid.UUID(bytes_le=guid_bytes_le)

    # 2. 将UUID对象转换为标准的字符串格式（默认为小写）
    guid_string = str(uuid_obj)

    # 3. 返回大写形式的字符串，以匹配最终要求
    return guid_string.upper()


import numpy as np
import matplotlib.pyplot as plt

def generate_circular_rays(forward_vector: np.ndarray, num_rays: int = 30, radius=1.0) -> np.ndarray:
    """
    根据给定的正前方单位向量，在XY平面上生成一组呈圆形分布的射线。

    这个函数通过将初始向量围绕Z轴旋转来生成所有射线。

    参数:
    - forward_vector (np.ndarray): 一个三维NumPy数组，表示actor的正前方单位向量。
    - num_rays (int): 要生成的射线总数。默认为30。

    返回:
    - np.ndarray: 一个形状为 (num_rays, 3) 的NumPy数组，其中每一行都是一条射线向量。
    """
    forward_vector = np.array(forward_vector, dtype=float)
    norm = np.linalg.norm(forward_vector)
    if norm == 0:
        raise ValueError("输入向量不能是零向量。")
    initial_vector = forward_vector / norm

    angle_increment_rad = np.deg2rad(360.0 / num_rays)

    rays = []
    for i in range(num_rays):
        current_angle_rad = i * angle_increment_rad
        c, s = np.cos(current_angle_rad), np.sin(current_angle_rad)
        rotation_matrix_z = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        new_ray = rotation_matrix_z @ initial_vector
        new_ray=new_ray*radius
        rays.append(new_ray)
        
    return np.array(rays)

# --- 可视化部分 ---
if __name__ == "__main__":
    # 1. 定义初始条件
    actor_forward_vector = np.array([1, 0, 0])  # 初始向量指向X轴正方向
    number_of_rays = 30
    
    # 2. 生成射线数据
    generated_rays = generate_circular_rays(actor_forward_vector, num_rays=number_of_rays)
    
    # 3. 创建绘图窗口和坐标轴
    # figsize=(8, 8)确保画布是正方形，看起来更舒服
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 4. 绘制单位圆作为参考
    # 这会画一个半径为1，圆心在(0,0)的灰色虚线圆
    unit_circle = plt.Circle((0, 0), 1, color='gray', linestyle='--', fill=False, label='Unit Circle')
    ax.add_artist(unit_circle)
    
    # 5. 绘制所有生成的射线（蓝色）
    # ax.quiver 用于绘制箭头(向量)
    # 我们将所有射线的起点都设为(0,0)
    # X, Y: 箭头的起点坐标 (这里都是0)
    # U, V: 箭头的方向和长度 (这里是射线的x和y分量)
    # angles='xy', scale_units='xy', scale=1: 这些参数确保箭头按1:1的比例绘制
    ax.quiver(
        np.zeros(number_of_rays),  # 所有箭头的X起点
        np.zeros(number_of_rays),  # 所有箭头的Y起点
        generated_rays[:, 0],      # 所有箭头的X分量
        generated_rays[:, 1],      # 所有箭头的Y分量
        color='blue',              # 颜色
        alpha=0.7,                 # 透明度
        label='Generated Rays',    # 图例标签
        angles='xy', scale_units='xy', scale=1
    )
    
    # 6. 突出显示初始向量（红色）
    # 我们在所有蓝色射线的上面再画一次初始向量，用红色覆盖
    ax.quiver(
        0, 0, 
        actor_forward_vector[0], actor_forward_vector[1], 
        color='red', 
        label=f'Initial Vector (0°)',
        angles='xy', scale_units='xy', scale=1,
        zorder=10 # 确保红色箭头在最上层
    )
    
    # 7. 设置图表样式
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_aspect('equal', adjustable='box') # 关键：确保X和Y轴比例相同，圆形不会被压扁成椭圆
    ax.grid(True, linestyle=':')
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.set_title(f"Visualization of {number_of_rays} Circular Rays", fontsize=14)
    ax.legend() # 显示图例
    
    # 8. 显示图表
    plt.show()
   

