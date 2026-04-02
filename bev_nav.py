#!/usr/bin/env python3
"""
bev_nav.py — OAK-D stereo depth → nav-ready semantic BEV costmap
             + YOLO-World open-vocabulary 3-D object detection

Costmap cell values:
  UNKNOWN  (  0) → black          no data yet
  FREE     ( 64) → green          traversable floor
  INFLATION(160) → orange         within robot radius of obstacle
  OCCUPIED (255) → red            obstacle point / detected object

Detected objects are projected as cyan labelled rectangles on the BEV.
RGB view shows floor overlay (magenta) + YOLO bounding boxes.

World coords:  X = left,  Y = up,  Z = forward
Keys:  Q = quit   R = reset costmap
"""

import threading
import time

import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLOWorld


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS — tune these to your setup
# ═══════════════════════════════════════════════════════════════════════════════
RGB_W, RGB_H = 640, 400

# Costmap grid
CELL_MM      = 40          # mm per BEV cell
BEV_X_MM     = 4000        # ± lateral extent
BEV_Z_MIN_MM = 200         # nearest depth plane shown
BEV_Z_MAX_MM = 6000        # farthest depth plane shown

# Navigation geometry
ROBOT_RADIUS_MM = 300      # inflation radius around obstacles
FLOOR_Y_MM      = -200     # world Y below this → traversable floor
                           # set to ≈ -(camera mount height in mm)
OBS_Y_MIN_MM    = -100     # world Y above this → obstacle (filters floor noise)
OBS_Y_MAX_MM    = 2000     # world Y below this → obstacle (ignores ceiling)

# Open-vocabulary detection
DETECT_CLASSES   = ["person", "chair", "table", "cardboard box",
                    "door", "wall", "bag", "bicycle", "trash can"]
DETECT_CONF      = 0.25
DETECT_INTERVAL  = 0.10    # seconds between GPU inference runs (~10 fps)
YOLO_MODEL       = "yolov8s-worldv2.pt"

# Depth EMA temporal smoothing
DEPTH_EMA_ALPHA  = 0.25

# Costmap decay: each frame cells fade slightly toward UNKNOWN so stale data
# doesn't linger.  Values here are subtracted per frame from cell counters.
FREE_DECAY     = 1          # FREE  cells fade in ~2 s at 30 fps
OCCUPIED_DECAY = 2          # OCCUPIED fades in ~4 s at 30 fps
# ═══════════════════════════════════════════════════════════════════════════════

# Cell values
UNKNOWN   =   0
FREE      =  64
INFLATION = 160
OCCUPIED  = 255

BEV_W        = int(2 * BEV_X_MM / CELL_MM)
BEV_H        = int((BEV_Z_MAX_MM - BEV_Z_MIN_MM) / CELL_MM)
ROBOT_CELLS  = max(1, int(ROBOT_RADIUS_MM / CELL_MM))

# Pre-build circular inflation kernel
def _circle_kernel(r):
    d = 2 * r + 1
    k = np.zeros((d, d), np.uint8)
    cv2.circle(k, (r, r), r, 1, -1)
    return k

_INF_KERNEL = _circle_kernel(ROBOT_CELLS)

# Jet LUT for depth visualisation
_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)


# ── Calibration ───────────────────────────────────────────────────────────────
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
print(f"[cal] fx={CAL['fx']:.1f}  fy={CAL['fy']:.1f}  "
      f"cx={CAL['cx']:.1f}  cy={CAL['cy']:.1f}  "
      f"baseline={CAL['baseline_mm']:.1f} mm")

_cal_adjusted = False


def _adjust_cal(h, w):
    global _cal_adjusted
    if _cal_adjusted:
        return
    _cal_adjusted = True
    sx, sy = w / RGB_W, h / RGB_H
    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
        CAL['fx'] *= sx;  CAL['cx'] *= sx
        CAL['fy'] *= sy;  CAL['cy'] *= sy
        print(f"[cal] rescaled for {w}×{h} → "
              f"fx={CAL['fx']:.1f} cx={CAL['cx']:.1f}")


# ── Back-projection ───────────────────────────────────────────────────────────
def depth_to_xyz(depth_mm: np.ndarray) -> np.ndarray:
    """Return Nx3 array (X=left, Y=up, Z=forward) for all valid pixels."""
    h, w = depth_mm.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    d = depth_mm.astype(np.float32)
    valid = (d > BEV_Z_MIN_MM) & (d < BEV_Z_MAX_MM)
    z = d[valid]
    x = -(uu[valid] - CAL['cx']) * z / CAL['fx']
    y = -(vv[valid] - CAL['cy']) * z / CAL['fy']
    return np.column_stack((x, y, z))


def bbox_depth_sample(depth_mm: np.ndarray,
                      x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Median depth of inner 50% of a 2-D bounding box. Returns mm or None."""
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = max(1, (x2 - x1) // 4), max(1, (y2 - y1) // 4)
    roi = depth_mm[max(0, my - bh):my + bh, max(0, mx - bw):mx + bw]
    v = roi[roi > 0]
    return float(np.median(v)) if v.size > 0 else None


# ── Floor mask ────────────────────────────────────────────────────────────────
def make_floor_mask(depth_mm: np.ndarray) -> np.ndarray:
    h = depth_mm.shape[0]
    vv = np.arange(h)[:, None]
    d  = depth_mm.astype(np.float32)
    valid = d > 0
    z = np.where(valid, d, 1.0)
    y_world = -(vv - CAL['cy']) * z / CAL['fy']
    return valid & (y_world < FLOOR_Y_MM)


# ── Depth EMA ─────────────────────────────────────────────────────────────────
_DEPTH_EMA: np.ndarray | None = None


def apply_depth_ema(raw: np.ndarray) -> np.ndarray:
    global _DEPTH_EMA
    f = raw.astype(np.float32)
    if _DEPTH_EMA is None or _DEPTH_EMA.shape != f.shape:
        _DEPTH_EMA = f.copy()
        return raw
    both        = (f > 0) & (_DEPTH_EMA > 0)
    new_only    = (f > 0) & (_DEPTH_EMA == 0)
    lost        = (f == 0) & (_DEPTH_EMA > 0)
    _DEPTH_EMA[both]     = DEPTH_EMA_ALPHA * f[both] + (1 - DEPTH_EMA_ALPHA) * _DEPTH_EMA[both]
    _DEPTH_EMA[new_only] = f[new_only]
    _DEPTH_EMA[lost]    *= 0.7
    _DEPTH_EMA[_DEPTH_EMA < 1] = 0
    return _DEPTH_EMA.astype(np.uint16)


# ── Semantic costmap ──────────────────────────────────────────────────────────
# Two internal accumulators: one for free evidence, one for occupied evidence.
# Final cell value is determined each frame from the accumulators.
_free_acc = np.zeros((BEV_H, BEV_W), np.float32)
_occ_acc  = np.zeros((BEV_H, BEV_W), np.float32)
_OCC_MAX  = 10.0   # saturation cap for accumulators
_FREE_MAX  = 10.0


def reset_costmap():
    _free_acc[:] = 0.0
    _occ_acc[:]  = 0.0


def _world_to_cell(x_mm: float, z_mm: float):
    col = int((x_mm + BEV_X_MM) / CELL_MM)
    row = int(BEV_H - 1 - (z_mm - BEV_Z_MIN_MM) / CELL_MM)
    return col, row


def update_costmap(xyz: np.ndarray) -> np.ndarray:
    """
    Ingest new point cloud, update accumulators, return rendered BGR costmap.
    """
    # --- decay accumulators toward zero ---
    _free_acc[:] = np.clip(_free_acc - 0.05, 0, _FREE_MAX)
    _occ_acc[:]  = np.clip(_occ_acc  - 0.07, 0, _OCC_MAX)

    if len(xyz) > 0:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        in_range = ((z > BEV_Z_MIN_MM) & (z < BEV_Z_MAX_MM) &
                    (x > -BEV_X_MM)    & (x < BEV_X_MM))

        # floor points → free evidence
        floor_mask = in_range & (y < FLOOR_Y_MM)
        if floor_mask.any():
            col = np.clip(((x[floor_mask] + BEV_X_MM) / CELL_MM).astype(np.int32),
                          0, BEV_W - 1)
            row = np.clip((BEV_H - 1 - (z[floor_mask] - BEV_Z_MIN_MM) / CELL_MM
                           ).astype(np.int32), 0, BEV_H - 1)
            np.add.at(_free_acc, (row, col), 1.0)
            _free_acc[:] = np.minimum(_free_acc, _FREE_MAX)

        # above-floor points → occupied evidence
        obs_mask = in_range & (y >= OBS_Y_MIN_MM) & (y < OBS_Y_MAX_MM)
        if obs_mask.any():
            col = np.clip(((x[obs_mask] + BEV_X_MM) / CELL_MM).astype(np.int32),
                          0, BEV_W - 1)
            row = np.clip((BEV_H - 1 - (z[obs_mask] - BEV_Z_MIN_MM) / CELL_MM
                           ).astype(np.int32), 0, BEV_H - 1)
            np.add.at(_occ_acc, (row, col), 1.0)
            _occ_acc[:] = np.minimum(_occ_acc, _OCC_MAX)

    # --- derive costmap from accumulators ---
    costmap = np.full((BEV_H, BEV_W), UNKNOWN, np.uint8)

    free_cells = _free_acc > 0.5
    occ_cells  = _occ_acc  > 0.5

    # free wins only where no occupied evidence
    costmap[free_cells & ~occ_cells] = FREE
    # occupied always wins
    costmap[occ_cells] = OCCUPIED

    # inflation: dilate occupied, apply over free/unknown but not occupied
    occ_layer = (costmap == OCCUPIED).astype(np.uint8)
    dilated   = cv2.dilate(occ_layer, _INF_KERNEL)
    inf_mask  = (dilated > 0) & (costmap != OCCUPIED)
    costmap[inf_mask] = INFLATION

    return costmap


def render_costmap(costmap: np.ndarray) -> np.ndarray:
    img = np.zeros((BEV_H, BEV_W, 3), np.uint8)
    img[costmap == FREE]      = (  0, 200,   0)   # green
    img[costmap == INFLATION] = (  0, 140, 255)   # orange
    img[costmap == OCCUPIED]  = (  0,   0, 255)   # red
    # UNKNOWN stays black
    return img


def draw_bev_overlay(bev_img: np.ndarray, detections: list) -> np.ndarray:
    """Add grid, camera marker, and YOLO-World object footprint boxes."""
    out = bev_img.copy()
    cx  = BEV_W // 2

    # distance grid lines
    cv2.line(out, (cx, 0), (cx, BEV_H), (55, 55, 55), 1)
    for d_m in range(1, BEV_Z_MAX_MM // 1000 + 1):
        r = BEV_H - 1 - int((d_m * 1000 - BEV_Z_MIN_MM) / CELL_MM)
        if 0 <= r < BEV_H:
            cv2.line(out, (0, r), (BEV_W, r), (55, 55, 55), 1)
            cv2.putText(out, f"{d_m}m", (2, r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1)

    # camera origin marker
    cv2.drawMarker(out, (cx, BEV_H - 1), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 12, 2)

    # object footprint rectangles
    for det in detections:
        label, conf = det['label'], det['conf']
        cx3, cz3    = det['cx3'], det['cz3']
        hw, hd      = det['half_w'], det['half_d']

        c1 = int((cx3 - hw + BEV_X_MM) / CELL_MM)
        c2 = int((cx3 + hw + BEV_X_MM) / CELL_MM)
        r1 = int(BEV_H - 1 - (cz3 + hd - BEV_Z_MIN_MM) / CELL_MM)
        r2 = int(BEV_H - 1 - (cz3 - hd - BEV_Z_MIN_MM) / CELL_MM)
        c1, c2 = sorted([c1, c2])
        r1, r2 = sorted([r1, r2])
        cv2.rectangle(out, (c1, r1), (c2, r2), (255, 255, 0), 1)
        cv2.putText(out, f"{label} {conf:.0%}",
                    (max(c1, 0), max(r1 - 2, 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 0), 1)

    # legend (bottom-left corner)
    items = [("FREE",      (  0, 200,   0)),
             ("INFLATION", (  0, 140, 255)),
             ("OCCUPIED",  (  0,   0, 255)),
             ("OBJECT",    (255, 255,   0))]
    for i, (name, color) in enumerate(items):
        y = BEV_H - 6 - i * 10
        cv2.rectangle(out, (2, y - 6), (10, y), color, -1)
        cv2.putText(out, name, (13, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.26, (200, 200, 200), 1)

    return out


# ── RGB annotator ─────────────────────────────────────────────────────────────
_MAGENTA = np.array([255, 0, 255], np.float32)


def annotate_rgb(rgb: np.ndarray,
                 depth_mm: np.ndarray,
                 detections: list,
                 floor_alpha: float = 0.45) -> np.ndarray:
    """Floor overlay + YOLO bounding boxes on the RGB frame."""
    # floor mask
    mask = make_floor_mask(depth_mm)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8),
                          (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = rgb.astype(np.float32)
    out[mask] = (1 - floor_alpha) * out[mask] + floor_alpha * _MAGENTA
    out = out.astype(np.uint8)

    # detection boxes
    for det in detections:
        x1, y1, x2, y2 = det['box2d']
        label, conf     = det['label'], det['conf']
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{label} {conf:.0%}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return out


# ── Depth coloriser ───────────────────────────────────────────────────────────
def colorize_depth(d: np.ndarray) -> np.ndarray:
    v = d[d > 0]
    if v.size == 0:
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n   = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    out = _LUT[(n * 255).astype(np.uint8), 0].copy()
    out[d == 0] = 0
    return out


# ── Shared state ──────────────────────────────────────────────────────────────
_cam_lock     = threading.Lock()
_latest_depth: np.ndarray | None = None
_latest_rgb:   np.ndarray | None = None

_det_lock   = threading.Lock()
_detections: list = []   # list of detection dicts (see detection_thread)

_running = True


# ── YOLO-World detection thread ───────────────────────────────────────────────
def detection_thread():
    global _detections, _running

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[det] Loading {YOLO_MODEL} on {device} …")
    model = YOLOWorld(YOLO_MODEL)
    model.set_classes(DETECT_CLASSES)
    model.to(device)
    print(f"[det] Ready — classes: {DETECT_CLASSES}")

    while _running:
        t0 = time.monotonic()

        with _cam_lock:
            rgb   = _latest_rgb
            depth = _latest_depth

        if rgb is None or depth is None:
            time.sleep(0.05)
            continue

        results = model.predict(rgb, conf=DETECT_CONF, verbose=False)[0]
        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = (DETECT_CLASSES[cls_id]
                      if cls_id < len(DETECT_CLASSES) else "object")
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            z = bbox_depth_sample(depth, x1, y1, x2, y2)
            if z is None or z < BEV_Z_MIN_MM or z > BEV_Z_MAX_MM:
                continue

            # 3-D centroid (X=left convention)
            px_cx = (x1 + x2) / 2.0
            cx3   = -(px_cx - CAL['cx']) * z / CAL['fx']

            # Approximate footprint half-extents from angular box width
            half_w = abs((x2 - x1) / 2.0 * z / CAL['fx'])
            half_d = max(half_w * 0.6, 150.0)   # assume ~square, min 300 mm

            dets.append({
                'label':  label,
                'conf':   conf,
                'box2d':  (x1, y1, x2, y2),
                'cx3':    cx3,
                'cz3':    z,
                'half_w': half_w,
                'half_d': half_d,
            })

        with _det_lock:
            _detections = dets

        sleep = DETECT_INTERVAL - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)


# ── Camera thread ─────────────────────────────────────────────────────────────
def camera_thread():
    global _latest_depth, _latest_rgb, _running

    pipeline = dai.Pipeline()

    monoL  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoR  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    monoL.requestOutput((RGB_W, RGB_H)).link(stereo.left)
    monoR.requestOutput((RGB_W, RGB_H)).link(stereo.right)

    camRgb     = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_stream = camRgb.requestOutput((RGB_W, RGB_H), dai.ImgFrame.Type.BGR888p)

    # Stereo config
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(5)
    stereo.setExtendedDisparity(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(RGB_W, RGB_H)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    try:
        cfg = stereo.initialConfig.get()
        cfg.postProcessing.spatialFilter.enable             = True
        cfg.postProcessing.spatialFilter.holeFillingRadius  = 2
        cfg.postProcessing.spatialFilter.numIterations      = 1
        cfg.postProcessing.spatialFilter.alpha              = 0.5
        cfg.postProcessing.spatialFilter.delta              = 20
        cfg.postProcessing.temporalFilter.enable            = True
        cfg.postProcessing.temporalFilter.alpha             = 0.2
        cfg.postProcessing.speckleFilter.enable             = True
        cfg.postProcessing.speckleFilter.speckleRange       = 50
        stereo.initialConfig.set(cfg)
        print("[stereo] filters: 7×7 median + spatial + temporal + speckle")
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
            _adjust_cal(raw.shape[0], raw.shape[1])
            dm = apply_depth_ema(raw)
            with _cam_lock:
                _latest_depth = dm
                _latest_rgb   = rf.getCvFrame() if rf else None

    _running = False


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global _running

    cam_t = threading.Thread(target=camera_thread,   daemon=True)
    det_t = threading.Thread(target=detection_thread, daemon=True)
    cam_t.start()
    det_t.start()

    scale = min(1.0, 700 / BEV_H)
    bev_disp_size = (int(BEV_W * scale), int(BEV_H * scale))

    print(f"BEV {BEV_W}×{BEV_H} cells @ {CELL_MM} mm/cell | "
          f"robot radius {ROBOT_RADIUS_MM} mm ({ROBOT_CELLS} cells) | "
          f"Q=quit  R=reset")

    while _running:
        with _cam_lock:
            depth_mm = _latest_depth
            rgb_img  = _latest_rgb
        with _det_lock:
            detections = list(_detections)

        if depth_mm is not None:
            # Costmap update
            xyz      = depth_to_xyz(depth_mm)
            costmap  = update_costmap(xyz)
            bev_img  = render_costmap(costmap)
            bev_vis  = draw_bev_overlay(bev_img, detections)
            cv2.imshow("BEV Costmap",
                       cv2.resize(bev_vis, bev_disp_size,
                                  interpolation=cv2.INTER_NEAREST))

            # Depth view
            cv2.imshow("Depth", colorize_depth(depth_mm))

            # RGB + floor + detections
            if rgb_img is not None:
                cv2.imshow("RGB | floor + objects",
                           annotate_rgb(rgb_img, depth_mm, detections))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _running = False
            break
        elif key == ord('r'):
            reset_costmap()
            print("[map] costmap reset")

    cv2.destroyAllWindows()
    cam_t.join(timeout=3)
    det_t.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()
