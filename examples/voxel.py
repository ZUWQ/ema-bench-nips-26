"""Voxel Query & Demo (TongSim)

Run:
    uv run --with numpy, matplotlib ./examples/voxel.py
(若未安装 matplotlib，会自动回退为控制台 ASCII 预览)
"""

import asyncio
import random
import time
from pathlib import Path

import numpy as np

import tongsim as ts
from tongsim.core.world_context import WorldContext

# ====== 可调参数 ======
GRPC_ENDPOINT = "127.0.0.1:5726"
START_LOC = ts.Vector3(200, -2000, 0)
STOP_LOC = ts.Vector3(2000, -2000, 0)
VELOCITY = 200.0  # 每秒移动速度（世界单位）
QUERY_PERIOD = 1.0  # 秒；体素查询与渲染周期
RES = (128, 128, 128)  # 体素分辨率 (X, Y, Z)
EXT = ts.Vector3(512, 512, 512)  # half-extent（盒体半尺寸）
FRAMES_DIR = Path("./voxel_frames")  # 图片输出目录
MAX_DIM_TO_RENDER = 32  # 渲染时各轴最大尺寸（自动降采样保证性能）
SAVE_EVERY_N = 1  # 每 N 次查询保存一张图
# =====================


# ===== voxel 解码 =====
def decode_voxel_byte(byte: int) -> list[bool]:
    """保持 LSB→MSB 的位序，返回 8 个布尔值。"""
    return [((byte >> i) & 1) == 1 for i in range(8)]


def decode_voxel(voxel_bytes: bytes, voxel_resolution: tuple[int, int, int]) -> np.ndarray:
    """
    高效体素位流解码（LSB-first）：
    - 输入：bytes（长度应为 ceil(X*Y*Z/8)）
    - 输出：形状 (X, Y, Z) 的 bool ndarray
    - 自动裁掉尾部多余 bit
    """
    x, y ,z = voxel_resolution
    num_voxel = x * y * z
    need_bytes = (num_voxel + 7) // 8
    if len(voxel_bytes) != need_bytes:
        raise ValueError(f"voxel_bytes length mismatch: expected {need_bytes}, got {len(voxel_bytes)}")

    buf = np.frombuffer(voxel_bytes, dtype=np.uint8, count=need_bytes)
    bits = np.unpackbits(buf, bitorder="little")  # LSB-first
    bits = bits[:num_voxel].astype(bool, copy=False)
    return bits.reshape((x, y, z), order="C")


# ===== 辅助&渲染 =====
def _rand_nearby(center: ts.Vector3, radius_xy: float = 300.0, z_jitter: float = 0.0) -> ts.Vector3:
    """在中心点附近随机一个目标点（圆形范围）"""
    ang = random.random() * 2.0 * np.pi
    r = (0.3 + 0.7 * random.random()) * radius_xy  # 避免太靠近
    return ts.Vector3(
        center.x + float(r * np.cos(ang)),
        center.y + float(r * np.sin(ang)),
        center.z + (random.uniform(-z_jitter, z_jitter) if z_jitter > 0 else 0.0),
    )


def _v3_to_np(v: ts.Vector3) -> np.ndarray:
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=np.float64)


def _np_to_v3(a: np.ndarray) -> ts.Vector3:
    return ts.Vector3(float(a[0]), float(a[1]), float(a[2]))


def _downsample_vox(vox: np.ndarray, max_dim: int = MAX_DIM_TO_RENDER) -> np.ndarray:
    """各轴独立降采样到不超过 max_dim，返回布尔体素体。"""
    sx = max(1, int(np.ceil(vox.shape[0] / max_dim)))
    sy = max(1, int(np.ceil(vox.shape[1] / max_dim)))
    sz = max(1, int(np.ceil(vox.shape[2] / max_dim)))
    return vox[::sx, ::sy, ::sz]


def render_voxels(vox: np.ndarray, save_path: Path, title: str | None = None) -> None:
    """
    """

    import matplotlib
    # 使用无界面后端，避免线程/窗口依赖
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (激活 3D)

    dv = _downsample_vox(vox, MAX_DIM_TO_RENDER)
    if dv.size == 0:
        print("[WARN] 空体素数据, 跳过渲染。")
        return

    fig = matplotlib.pyplot.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")  # 单独一个 3D 图
    # 注意：不指定颜色，使用默认；避免渲染样式干扰
    ax.voxels(dv)  # 简洁直接；若仍然偏慢，可改为 ax.scatter(np.where(dv))
    if title:
        ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    matplotlib.pyplot.close(fig)


# ===== 体素展示任务 =====
async def show_voxel(context: WorldContext) -> None:
    """
    周期查询代理周围体素并做简易渲染
    可根据需要替换为引擎内 DebugDraw。
    """
    # 运动：在 START_LOC 与 STOP_LOC 之间往返（ping-pong）
    p0 = _v3_to_np(START_LOC)
    p1 = _v3_to_np(STOP_LOC)
    seg = p1 - p0
    total_dist = float(np.linalg.norm(seg)) if np.linalg.norm(seg) > 1e-6 else 1.0
    dir_unit = seg / total_dist

    cur = p0.copy()
    direction = +1.0  # +1 往 p1，-1 往 p0
    start_transform = ts.Transform(location=_np_to_v3(cur))

    tick = 0
    # last_t = time.perf_counter()

    while True:
        # === 移动 transform ===
        # now = time.perf_counter()
        # dt = max(QUERY_PERIOD, now - last_t)  # 以周期为步长（更稳定）
        # last_t = now

        step = VELOCITY * QUERY_PERIOD * direction
        cur = cur + dir_unit * step

        # 到达端点则反向，并夹紧到线段上
        to_p0 = float(np.dot(cur - p0, dir_unit))
        if to_p0 <= 0.0:
            cur = p0.copy()
            direction = +1.0
        elif to_p0 >= total_dist:
            cur = p1.copy()
            direction = -1.0

        start_transform.location = _np_to_v3(cur)

        # === 查询体素 ===
        voxel_bytes = await ts.UnaryAPI.query_voxel(
            context.conn, start_transform, RES[0], RES[1], RES[2], EXT
        )
        vox = decode_voxel(voxel_bytes, RES)

        # === 绘制 ===
        tick += 1
        if tick % SAVE_EVERY_N == 0:
            fname = FRAMES_DIR / f"voxel_{tick:05d}.png"
            title = f"Tick {tick}  Loc=({cur[0]:.1f},{cur[1]:.1f},{cur[2]:.1f})  Res={RES}"
            render_voxels(vox, fname, title)
            print(f"[RENDER] Saved: {fname}")

        await asyncio.sleep(QUERY_PERIOD)


# ===== 入口 =====
def main() -> None:
    print("[INFO] 连接到 TongSim ...")
    with ts.TongSim(grpc_endpoint=GRPC_ENDPOINT) as ue:
        # 重置场景
        ue.context.sync_run(ts.UnaryAPI.reset_level(ue.context.conn))
        # 启动体素查询展示（异步后台任务）
        ue.context.async_task(show_voxel(ue.context), "voxel")

        # 主线程保持存活（让后台任务运行）；可改为其他主逻辑
        time.sleep(300.0)

    print("[INFO] 演示完成。")


if __name__ == "__main__":
    main()
