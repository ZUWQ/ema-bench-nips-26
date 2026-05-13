
import OpenEXR, Imath, numpy as np, cv2

def normalize_to_8bit(depth, min_val=None, max_val=None):
    if min_val is None:
        min_val = np.nanmin(depth)
    if max_val is None:
        max_val = np.nanmax(depth)
    d = depth.copy()
    # 可处理 inf/nan
    d = np.nan_to_num(d, nan=min_val, posinf=max_val, neginf=min_val)
    if max_val - min_val < 1e-8:
        return (np.clip(d, min_val, max_val) * 0).astype(np.uint8)
    norm = (d - min_val) / (max_val - min_val)
    return (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)

def linearize_opengl_depth(d, near, far):
    # d: float array in [0,1]
    z_ndc = d * 2.0 - 1.0
    z_eye = (2.0 * near * far) / (far + near - z_ndc * (far - near))
    return z_eye

def read_exr(path, channel=None):
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    if channel is None:
        chans = list(header['channels'].keys())
        # 尝试优先 Z, depth, R
        for prefer in ['Z','depth','R','G','B']:
            if prefer in chans:
                channel = prefer
                print(f"Using channel {channel} .")
                break
        if channel is None:
            channel = chans[0]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(channel, pt)
    arr = np.frombuffer(raw, dtype=np.float32).reshape((h, w))
    return arr

# path = 'logs/DevCapture_1_640x480.depth.exr'
path = '/home/zwq/文档/EMAS/tongsim_lite/PythonClient/examples/spray_1_actor_971396dbbd45a32ae4a387b8b29f0762_depth.exr'
depth = read_exr(path)
# 去重：误差在 1 内视为同一值，保留代表值
flat = depth.ravel()
flat = flat[np.isfinite(flat)]
flat = np.sort(flat)
unique_vals = []
for v in flat:
    if len(unique_vals) == 0 or v > unique_vals[-1] + 5:
        unique_vals.append(v)
print("depth 去重后（误差≤1 合并）:", unique_vals)
# 如果是 z-buffer，线性化（给定 near/far）
depth_lin = linearize_opengl_depth(depth, near=0.1, far=1000.0)
# 否则直接用 depth

# 可视化（自动用 1%~99% 做对比）
vmin = np.nanpercentile(depth_lin, 1)
vmax = np.nanpercentile(depth_lin, 99)
depth_8 = normalize_to_8bit(depth_lin, vmin, vmax)
cv2.imwrite('depth_vis.png', depth_8)
cv2.imwrite('depth_vis_color.png', cv2.applyColorMap(depth_8, cv2.COLORMAP_JET))
