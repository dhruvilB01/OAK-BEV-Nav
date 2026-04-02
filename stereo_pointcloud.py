#!/usr/bin/env python3
"""
OAK-D — stereo depth → dense point cloud + floor detection overlay + BEV

World coords: X=left, Y=up, Z=forward
All cameras run at RGB_W × RGB_H (lowest common resolution).
Depth is aligned to the RGB camera so the floor overlay is pixel-accurate.

Keys: Q=quit  R=reset BEV
"""

import threading
import cv2
import depthai as dai
import numpy as np
import open3d as o3d


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
RGB_W, RGB_H = 640, 400       # all cameras run at this resolution

BEV_RES_MM   = 40
BEV_X_MM     = 4000
BEV_Z_MIN_MM = 200
BEV_Z_MAX_MM = 6000
BEV_Y_MIN_MM = -1000
BEV_Y_MAX_MM = 3000
BEV_DECAY    = 0.92

# Floor: world Y below this → magenta.  Set to ≈ -(camera mount height in mm).
FLOOR_Y_MM          = -500
FLOOR_OVERLAY_ALPHA = 0.55    # 0=invisible, 1=solid magenta on RGB overlay

VOXEL_MM          = 15        # point-cloud voxel grid (mm); 0 = disabled
OUTLIER_NEIGHBORS = 20
OUTLIER_STD       = 2.0

_DEPTH_EMA_ALPHA  = 0.25
# ═══════════════════════════════════════════════════════════════════════════════

BEV_W = int(2 * BEV_X_MM / BEV_RES_MM)
BEV_H = int((BEV_Z_MAX_MM - BEV_Z_MIN_MM) / BEV_RES_MM)
_LUT  = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

# ── Calibration ───────────────────────────────────────────────────────────────
# Depth is aligned to CAM_A (RGB) → use RGB intrinsics for back-projection.
def get_calibration():
    with dai.Device() as dev:
        cal = dev.readCalibration()
        M   = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, RGB_W, RGB_H)
        ext = cal.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                       dai.CameraBoardSocket.CAM_B)
        baseline_mm = abs(np.array(ext)[0, 3] * 10)
    return dict(fx=M[0][0], fy=M[1][1], cx=M[0][2], cy=M[1][2],
                baseline_mm=baseline_mm)


CAL = get_calibration()
print(f"[cal] RGB intrinsics  fx={CAL['fx']:.1f}  fy={CAL['fy']:.1f}  "
      f"cx={CAL['cx']:.1f}  cy={CAL['cy']:.1f}  "
      f"baseline={CAL['baseline_mm']:.1f} mm")

_cal_adjusted = False


def adjust_cal_for_frame(h: int, w: int):
    global _cal_adjusted
    if _cal_adjusted:
        return
    _cal_adjusted = True
    sx, sy = w / RGB_W, h / RGB_H
    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        CAL['fx'] *= sx;  CAL['cx'] *= sx
        CAL['fy'] *= sy;  CAL['cy'] *= sy
        print(f"[cal] depth frame {w}×{h} → rescaled intrinsics "
              f"fx={CAL['fx']:.1f} cx={CAL['cx']:.1f}")


# ── Back-projection (full-res, no decimation) ─────────────────────────────────
def depth_to_xyz(depth_mm: np.ndarray) -> np.ndarray:
    """World coords: X=left, Y=up, Z=forward."""
    h, w = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    d = depth_mm.astype(np.float32)
    valid = (d > BEV_Z_MIN_MM) & (d < BEV_Z_MAX_MM)
    z = d[valid]
    x = -(uu[valid] - CAL['cx']) * z / CAL['fx']
    y = -(vv[valid] - CAL['cy']) * z / CAL['fy']
    return np.column_stack((x, y, z))


# ── Floor mask (same 2-D shape as depth frame) ────────────────────────────────
def make_floor_mask(depth_mm: np.ndarray) -> np.ndarray:
    h = depth_mm.shape[0]
    vv = np.arange(h)[:, None]          # column vector, broadcast across cols
    d  = depth_mm.astype(np.float32)
    valid = d > 0
    z = np.where(valid, d, 1.0)          # avoid /0 on invalid pixels
    y_world = -(vv - CAL['cy']) * z / CAL['fy']
    return valid & (y_world < FLOOR_Y_MM)


# ── RGB floor overlay ─────────────────────────────────────────────────────────
_MAGENTA = np.array([255, 0, 255], dtype=np.float32)   # BGR


def overlay_floor(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8),
                          (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = rgb.astype(np.float32)
    out[mask] = (1 - FLOOR_OVERLAY_ALPHA) * out[mask] + FLOOR_OVERLAY_ALPHA * _MAGENTA
    return out.astype(np.uint8)


# ── Point-cloud cleaning ──────────────────────────────────────────────────────
def clean_xyz(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) < OUTLIER_NEIGHBORS + 1:
        return xyz
    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(xyz)
    tmp, _ = tmp.remove_statistical_outlier(nb_neighbors=OUTLIER_NEIGHBORS,
                                             std_ratio=OUTLIER_STD)
    if VOXEL_MM > 0:
        tmp = tmp.voxel_down_sample(VOXEL_MM)
    return np.asarray(tmp.points)


# ── BEV ───────────────────────────────────────────────────────────────────────
_bev_z = np.full((BEV_H, BEV_W), np.inf, dtype=np.float32)


def reset_bev():
    _bev_z[:] = np.inf


def update_bev(xyz: np.ndarray) -> np.ndarray:
    finite = np.isfinite(_bev_z)
    _bev_z[finite] /= BEV_DECAY
    _bev_z[_bev_z > BEV_Z_MAX_MM * 1.1] = np.inf
    if len(xyz) > 0:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ok = ((x >= -BEV_X_MM) & (x <= BEV_X_MM) &
              (y >= BEV_Y_MIN_MM) & (y <= BEV_Y_MAX_MM))
        x, y, z = x[ok], y[ok], z[ok]
        if len(x) > 0:
            col = ((x + BEV_X_MM) / BEV_RES_MM).astype(np.int32)
            row = (BEV_H - 1 - (z - BEV_Z_MIN_MM) / BEV_RES_MM).astype(np.int32)
            ib  = (col >= 0) & (col < BEV_W) & (row >= 0) & (row < BEV_H)
            np.minimum.at(_bev_z, (row[ib], col[ib]), z[ib])
    occ = np.isfinite(_bev_z)
    z_n = np.clip((_bev_z - BEV_Z_MIN_MM) / (BEV_Z_MAX_MM - BEV_Z_MIN_MM), 0, 1)
    bev = _LUT[(z_n * 255).astype(np.uint8), 0].copy()
    bev[~occ] = 0
    return bev


def draw_bev(bev: np.ndarray) -> np.ndarray:
    out = bev.copy()
    cx  = BEV_W // 2
    cv2.line(out, (cx, 0), (cx, BEV_H), (60, 60, 60), 1)
    for d in range(1000, BEV_Z_MAX_MM, 1000):
        r = BEV_H - 1 - int((d - BEV_Z_MIN_MM) / BEV_RES_MM)
        if 0 <= r < BEV_H:
            cv2.line(out, (0, r), (BEV_W, r), (60, 60, 60), 1)
            cv2.putText(out, f"{d//1000}m", (2, r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
    cv2.drawMarker(out, (cx, BEV_H - 1), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 12, 2)
    return out


def colorize_depth(d: np.ndarray) -> np.ndarray:
    v = d[d > 0]
    if v.size == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n   = np.clip((d.astype(np.float32) - lo) / (hi - lo + 1), 0, 1)
    out = _LUT[(n * 255).astype(np.uint8), 0].copy()
    out[d == 0] = 0
    return out


# ── Shared state ──────────────────────────────────────────────────────────────
_lock              = threading.Lock()
_latest_depth      = None
_latest_rgb        = None
_latest_floor_mask = None
_running           = True

_DEPTH_EMA: np.ndarray | None = None


def apply_depth_ema(raw: np.ndarray) -> np.ndarray:
    global _DEPTH_EMA
    f = raw.astype(np.float32)
    if _DEPTH_EMA is None or _DEPTH_EMA.shape != f.shape:
        _DEPTH_EMA = f.copy()
        return raw
    valid_both  = (f > 0) & (_DEPTH_EMA > 0)
    valid_new   = (f > 0) & (_DEPTH_EMA == 0)
    became_zero = (f == 0) & (_DEPTH_EMA > 0)
    _DEPTH_EMA[valid_both]  = (_DEPTH_EMA_ALPHA * f[valid_both] +
                                (1 - _DEPTH_EMA_ALPHA) * _DEPTH_EMA[valid_both])
    _DEPTH_EMA[valid_new]   = f[valid_new]
    _DEPTH_EMA[became_zero] *= 0.7
    _DEPTH_EMA[_DEPTH_EMA < 1] = 0
    return _DEPTH_EMA.astype(np.uint16)


# ── Camera thread ─────────────────────────────────────────────────────────────
def camera_thread():
    global _latest_depth, _latest_rgb, _latest_floor_mask, _running
    pipeline = dai.Pipeline()

    # Mono cameras at target resolution
    monoL  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoR  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    monoL.requestOutput((RGB_W, RGB_H)).link(stereo.left)
    monoR.requestOutput((RGB_W, RGB_H)).link(stereo.right)

    # RGB camera at same resolution — depth will be aligned to it
    camRgb     = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_stream = camRgb.requestOutput((RGB_W, RGB_H), dai.ImgFrame.Type.BGR888p)

    # Stereo
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(5)
    stereo.setExtendedDisparity(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # depth → RGB perspective
    stereo.setOutputSize(RGB_W, RGB_H)                  # must be explicit when aligning to RGB

    # Device-side filters
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    try:
        cfg = stereo.initialConfig.get()
        cfg.postProcessing.spatialFilter.enable          = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations   = 1
        cfg.postProcessing.spatialFilter.alpha           = 0.5
        cfg.postProcessing.spatialFilter.delta           = 20
        cfg.postProcessing.temporalFilter.enable         = True
        cfg.postProcessing.temporalFilter.alpha          = 0.2
        cfg.postProcessing.speckleFilter.enable          = True
        cfg.postProcessing.speckleFilter.speckleRange    = 50
        stereo.initialConfig.set(cfg)
        print("[stereo] device filters: 7×7 median + spatial + temporal + speckle")
    except Exception as e:
        print(f"[stereo] filter config failed ({e}), using median only")

    depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
    rgb_q   = rgb_stream.createOutputQueue(maxSize=2, blocking=False)

    with pipeline:
        pipeline.start()
        try:
            pipeline.setIrLaserDotProjectorIntensity(0.8)
            print("[IR] dot projector ON")
        except Exception:
            pass

        while _running and pipeline.isRunning():
            df = depth_q.get()
            rf = rgb_q.tryGet()
            if df is None:
                continue
            raw = df.getFrame().astype(np.uint16)
            adjust_cal_for_frame(raw.shape[0], raw.shape[1])
            dm         = apply_depth_ema(raw)
            floor_mask = make_floor_mask(dm)
            with _lock:
                _latest_depth      = dm
                _latest_rgb        = rf.getCvFrame() if rf else None
                _latest_floor_mask = floor_mask
    _running = False


# ── Open3D view ───────────────────────────────────────────────────────────────
def _set_view(vis):
    vc = vis.get_view_control()
    vc.set_up([0, 1, 0])
    vc.set_front([0, -0.25, -1])
    vc.set_lookat([0, 0, 2500])
    vc.set_zoom(0.25)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global _running

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud", width=1024, height=768)
    vis.get_render_option().background_color = np.zeros(3)
    vis.get_render_option().point_size = 2.0
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=500))
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    _set_view(vis)

    cam = threading.Thread(target=camera_thread, daemon=True)
    cam.start()
    print(f"Resolution {RGB_W}×{RGB_H} | BEV {BEV_W}×{BEV_H} | "
          f"floor Y < {FLOOR_Y_MM} mm  Q=quit  R=reset BEV")

    view_set = False
    prev_key = -1

    while _running:
        with _lock:
            depth_mm   = _latest_depth
            rgb_img    = _latest_rgb
            floor_mask = _latest_floor_mask

        if depth_mm is not None:
            cv2.imshow("Depth", colorize_depth(depth_mm))

            if rgb_img is not None and floor_mask is not None:
                cv2.imshow("Floor overlay (magenta=floor)", overlay_floor(rgb_img, floor_mask))

            xyz_raw = depth_to_xyz(depth_mm)
            xyz     = clean_xyz(xyz_raw)

            bev   = update_bev(xyz)
            scale = min(1.0, 700 / BEV_H)
            cv2.imshow("BEV", cv2.resize(draw_bev(bev),
                               (int(BEV_W * scale), int(BEV_H * scale)),
                               interpolation=cv2.INTER_NEAREST))

            if len(xyz) > 0:
                z_n    = np.clip((xyz[:, 2] - BEV_Z_MIN_MM) /
                                 (BEV_Z_MAX_MM - BEV_Z_MIN_MM), 0, 1)
                colors = _LUT[(z_n * 255).astype(np.uint8), 0, ::-1] / 255.0
                colors[xyz[:, 1] < FLOOR_Y_MM] = [1.0, 0.0, 1.0]   # magenta floor
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                if not view_set:
                    view_set = True
                    _set_view(vis)

        key = cv2.waitKey(1) & 0xFF
        if key != prev_key:
            if key == ord('q'):
                _running = False
                break
            elif key == ord('r'):
                reset_bev()
                print("BEV reset.")
        prev_key = key if key != 0xFF else -1

        if not vis.poll_events():
            _running = False
            break
        vis.update_renderer()

    vis.destroy_window()
    cv2.destroyAllWindows()
    cam.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()
