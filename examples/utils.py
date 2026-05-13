import uuid
import time
import tongsim as ts
import numpy as np

def calculate_bounce_velocity_sample(
    velocity_vector: np.ndarray, 
    wall_normal_vector: ts.Vector3,
    restitution: float = 1.0  # 弹力系数 (1.0表示完美反弹)
) -> np.ndarray:
    return -1*velocity_vector

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
   

