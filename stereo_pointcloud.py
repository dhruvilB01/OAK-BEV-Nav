#!/usr/bin/env python3
"""
OAK-D — Schematic Bird's Eye View

Pipeline:
  1. RGB + Depth capture (DepthAI, aligned)
  2. YOLOv8 object detection → bounding boxes + class labels
  3. Depth sampling at each detection → real-world (X, Z) position + size
  4. FastSAM floor segmentation → drivable area boundaries
  5. Schematic BEV rendering — clean diagram, NOT a camera warp

Output looks like a radar/planning display:
  - Gray road surface with boundary lines
  - Green labelled rectangles for each detected obstacle
  - Distance grid lines at 1 m intervals
  - Ego vehicle marker at bottom

Keys:  Q = quit
"""

import threading
import cv2
import depthai as dai
import numpy as np

# ── Camera ────────────────────────────────────────────────────────────────────
RGB_W, RGB_H  = 640, 400
DEPTH_MIN_MM  = 300
DEPTH_MAX_MM  = 50000                # 50 m — outdoor long range

# ── BEV schematic ─────────────────────────────────────────────────────────────
BEV_W, BEV_H      = 400, 700        # pixels (width × height of output image)
BEV_RANGE_FWD_M    = 15.0            # how far forward to show (metres)
BEV_RANGE_SIDE_M   = 8.0             # how far left/right to show (metres)
BEV_ROAD_WIDTH_M   = 6.0             # default road width if no seg (metres)

# ── Camera BEV (data-driven homography warp) ─────────────────────────────────
CAM_BEV_SIZE       = 700             # square output pixels (larger for 15 m range)
CAM_BEV_RANGE_M    = 15.0            # metres shown (robot at bottom centre)

# Derived scale: pixels per metre
_PPM_X = BEV_W / (2.0 * BEV_RANGE_SIDE_M)    # px per m, lateral
_PPM_Z = (BEV_H - 60) / BEV_RANGE_FWD_M      # px per m, forward (leave 60px for ego)

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_BG       = (40,  40,  40)
COL_ROAD     = (80,  80,  80)
COL_BOUNDARY = (180, 180, 180)
COL_GRID     = (60,  60,  60)
COL_OBS_FILL = (80,  200, 80)
COL_OBS_EDGE = (40,  160, 40)
COL_EGO      = (200, 160, 40)
COL_TEXT      = (220, 220, 220)
COL_DIST     = (100, 100, 100)

# ── Detection ─────────────────────────────────────────────────────────────────
YOLO_MODEL    = "yolov8n.pt"          # change to your model path
YOLO_CONF     = 0.50                  # higher = fewer false positives
YOLO_IOU      = 0.45                  # NMS IoU threshold
MIN_BBOX_AREA = 1500                  # ignore tiny detections (px²)
DEPTH_SAMPLE  = "bottom"              # "bottom" = base of object, "center" = bbox center, "full" = entire box
DET_EVERY_N   = 2                     # run detection every N frames
SEG_EVERY_N   = 3                     # run floor seg every N frames

# ── Depth EMA ─────────────────────────────────────────────────────────────────
DEPTH_EMA_A   = 0.3


# ══════════════════════════════════════════════════════════════════════════════
#  Calibration
# ══════════════════════════════════════════════════════════════════════════════
def get_calibration():
    with dai.Device() as dev:
        cal = dev.readCalibration()
        M    = cal.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, RGB_W, RGB_H)
        dist = cal.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A)
        ext  = cal.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C,
                                       dai.CameraBoardSocket.CAM_B)
    K = np.array(M, dtype=np.float64)
    D = np.array(dist, dtype=np.float64)
    # Optimal undistorted camera matrix (alpha=0: no black borders)
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (RGB_W, RGB_H), alpha=0)
    return dict(
        fx=K_new[0, 0], fy=K_new[1, 1],
        cx=K_new[0, 2], cy=K_new[1, 2],
        K=K, D=D, K_new=K_new,
        baseline_mm=abs(np.array(ext)[0, 3] * 10),
    )

CAL = get_calibration()
print(f"[cal] fx={CAL['fx']:.1f} fy={CAL['fy']:.1f} "
      f"cx={CAL['cx']:.1f} cy={CAL['cy']:.1f} "
      f"baseline={CAL['baseline_mm']:.1f}mm")

_cal_adjusted = False
def adjust_cal(h, w):
    global _cal_adjusted
    if _cal_adjusted: return
    _cal_adjusted = True
    sx, sy = w / RGB_W, h / RGB_H
    if abs(sx-1)>0.01 or abs(sy-1)>0.01:
        CAL['fx']*=sx; CAL['cx']*=sx; CAL['fy']*=sy; CAL['cy']*=sy


# ══════════════════════════════════════════════════════════════════════════════
#  Depth helpers
# ══════════════════════════════════════════════════════════════════════════════
_depth_ema = None

def smooth_depth(raw):
    global _depth_ema
    f = raw.astype(np.float32)
    if _depth_ema is None or _depth_ema.shape != f.shape:
        _depth_ema = f.copy(); return raw
    both = (f>0)&(_depth_ema>0)
    new  = (f>0)&(_depth_ema==0)
    _depth_ema[both] = DEPTH_EMA_A*f[both] + (1-DEPTH_EMA_A)*_depth_ema[both]
    _depth_ema[new]  = f[new]
    _depth_ema[(f==0)&(_depth_ema>0)] *= 0.7
    _depth_ema[_depth_ema<1] = 0
    return _depth_ema.astype(np.uint16)


def sample_depth_roi(depth_mm, x1, y1, x2, y2) -> float:
    """
    Robust depth sample from a bounding box.

    Modes (set by DEPTH_SAMPLE parameter):
      "bottom"  — bottom 20%, central 60% width (where object meets floor)
      "center"  — central 40% height × 60% width (object body, avoids edges)
      "full"    — entire bbox, take median (most data, but picks up background)

    Returns depth in mm, or 0 if invalid.
    """
    h, w = depth_mm.shape
    bw = max(int((x2 - x1) * 0.6), 3)
    cx_box = (x1 + x2) // 2

    if DEPTH_SAMPLE == "bottom":
        bh = max(int((y2 - y1) * 0.2), 3)
        roi_y1 = max(0, y2 - bh)
        roi_y2 = min(h, y2)
        roi_x1 = max(0, cx_box - bw // 2)
        roi_x2 = min(w, cx_box + bw // 2)

    elif DEPTH_SAMPLE == "center":
        bh = max(int((y2 - y1) * 0.4), 3)
        cy_box = (y1 + y2) // 2
        roi_y1 = max(0, cy_box - bh // 2)
        roi_y2 = min(h, cy_box + bh // 2)
        roi_x1 = max(0, cx_box - bw // 2)
        roi_x2 = min(w, cx_box + bw // 2)

    else:   # "full"
        roi_y1 = max(0, y1)
        roi_y2 = min(h, y2)
        roi_x1 = max(0, x1)
        roi_x2 = min(w, x2)

    roi = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    valid = roi[(roi > DEPTH_MIN_MM) & (roi < DEPTH_MAX_MM)]
    if len(valid) < 5:
        return 0.0
    return float(np.median(valid))


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO detection (background thread)
# ══════════════════════════════════════════════════════════════════════════════
_yolo = None
_det_lock = threading.Lock()
_detections = []       # list of (x1,y1,x2,y2, class_name, conf)
_det_busy = False

def _load_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(YOLO_MODEL)
        print(f"[det] YOLO loaded: {YOLO_MODEL}")
    return _yolo

def _det_worker(rgb):
    global _detections, _det_busy
    try:
        model = _load_yolo()
        results = model(rgb, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
        dets = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # Skip tiny detections (noise)
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_BBOX_AREA:
                    continue
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, f"cls{cls_id}")
                dets.append((x1, y1, x2, y2, name, conf))
        with _det_lock:
            _detections = dets
    except Exception as e:
        print(f"[det] failed: {e}")
    finally:
        _det_busy = False


# ══════════════════════════════════════════════════════════════════════════════
#  Floor segmentation (background thread)
# ══════════════════════════════════════════════════════════════════════════════
_fastsam = None
_seg_lock = threading.Lock()
_seg_mask = None
_seg_busy = False
_seg_mask_version = 0   # incremented each time a new mask arrives

def _load_fastsam():
    global _fastsam
    if _fastsam is None:
        from ultralytics import FastSAM
        _fastsam = FastSAM("FastSAM-s.pt")
        print("[seg] FastSAM-s loaded")
    return _fastsam

def _seg_worker(rgb):
    global _seg_mask, _seg_busy, _seg_mask_version
    try:
        model = _load_fastsam()
        results = model(rgb, texts="road or sidewalk or pavement or path", device="cuda", verbose=False, conf=0.3)
        if results and results[0].masks is not None:
            m = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            with _seg_lock:
                _seg_mask = m
                _seg_mask_version += 1
    except Exception as e:
        print(f"[seg] failed: {e}")
    finally:
        _seg_busy = False


# ══════════════════════════════════════════════════════════════════════════════
#  IMU — gravity-aligned rotation matrix
# ══════════════════════════════════════════════════════════════════════════════
_imu_lock = threading.Lock()
_imu_R    = np.eye(3, dtype=np.float64)   # camera→world rotation (updated from BNO085)

def _quat_to_rot(w, x, y, z):
    """Unit quaternion → 3×3 rotation matrix."""
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  Camera thread
# ══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()
_latest_depth = None
_latest_rgb   = None
_running      = True

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
    # FAST_ACCURACY: 5-bit subpixel, strict LR check (threshold=5),
    # no decimation, no spatial/median — best accuracy for outdoor long range
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setExtendedDisparity(False)    # keep max range (extended = near field <1m only)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(RGB_W, RGB_H)
    try:
        cfg = stereo.initialConfig.get()
        # Temporal filter: stabilise long-range readings across frames
        cfg.postProcessing.temporalFilter.enable = True
        cfg.postProcessing.temporalFilter.alpha = 0.4
        # Speckle filter: remove isolated noise pixels common at range
        cfg.postProcessing.speckleFilter.enable = True
        cfg.postProcessing.speckleFilter.speckleRange = 200
        # Threshold filter: extend range to 50 m, clip below DEPTH_MIN
        cfg.postProcessing.thresholdFilter.minRange = DEPTH_MIN_MM
        cfg.postProcessing.thresholdFilter.maxRange = DEPTH_MAX_MM
        stereo.initialConfig.set(cfg)
    except Exception: pass

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, 100)   # BNO085 fusion @ 100 Hz
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    depth_q = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
    rgb_q   = rgb_stream.createOutputQueue(maxSize=2, blocking=False)
    imu_q   = imu.out.createOutputQueue(maxSize=10, blocking=False)
    with pipeline:
        pipeline.start()
        try: pipeline.setIrLaserDotProjectorIntensity(0.8)
        except: pass
        while _running and pipeline.isRunning():
            df = depth_q.get()
            rf = rgb_q.tryGet()
            if df is None: continue
            raw = df.getFrame().astype(np.uint16)
            adjust_cal(raw.shape[0], raw.shape[1])
            rgb_frame = None
            if rf is not None:
                frame = rf.getCvFrame()
                rgb_frame = cv2.undistort(frame, CAL['K'], CAL['D'], None, CAL['K_new'])
            with _lock:
                _latest_depth = smooth_depth(raw)
                _latest_rgb   = rgb_frame
            # Update IMU rotation matrix from latest packet
            imu_data = imu_q.tryGet()
            if imu_data:
                for pkt in imu_data.packets:
                    rv = pkt.rotationVector
                    R = _quat_to_rot(rv.real, rv.i, rv.j, rv.k)
                    with _imu_lock:
                        global _imu_R
                        _imu_R = R
    _running = False


# ══════════════════════════════════════════════════════════════════════════════
#  3D measurement from detection + depth
# ══════════════════════════════════════════════════════════════════════════════
def measure_detections(depth_mm, detections):
    """
    For each (x1,y1,x2,y2, name, conf), compute:
      - z_m:  forward distance in metres
      - x_m:  lateral offset in metres (positive = left)
      - w_m:  estimated real-world width in metres
      - d_m:  estimated real-world depth in metres (along Z)
    Returns list of dicts.
    """
    measured = []
    fx, fy = CAL['fx'], CAL['fy']
    cx_c   = CAL['cx']

    for (x1, y1, x2, y2, name, conf) in detections:
        z_mm = sample_depth_roi(depth_mm, x1, y1, x2, y2)
        if z_mm < DEPTH_MIN_MM:
            continue

        z_m = z_mm / 1000.0

        # Lateral position: bbox centre → X offset
        u_centre = (x1 + x2) / 2.0
        x_m = -(u_centre - cx_c) * z_m / fx     # positive = left

        # Real-world width from bbox pixel width
        bbox_w_px = x2 - x1
        w_m = bbox_w_px * z_m / fx
        w_m = max(w_m, 0.15)   # at least 15 cm

        # Depth (along Z) — estimate from aspect ratio or use fixed
        bbox_h_px = y2 - y1
        d_m = bbox_h_px * z_m / fy * 0.4   # rough: objects are usually deeper than tall projection
        d_m = max(d_m, 0.15)
        d_m = min(d_m, w_m * 1.5)  # don't let it get absurd

        measured.append(dict(
            name=name, conf=conf,
            z_m=z_m, x_m=x_m,
            w_m=w_m, d_m=d_m,
            bbox=(x1, y1, x2, y2),
        ))

    # Sort by distance (closest first)
    measured.sort(key=lambda m: m['z_m'])
    return measured


# ══════════════════════════════════════════════════════════════════════════════
#  Road boundary from floor segmentation
# ══════════════════════════════════════════════════════════════════════════════
def get_road_boundaries(seg_mask, depth_mm):
    """
    From the floor mask, extract left and right road boundary lines
    as lists of (x_m, z_m) points in camera frame.
    """
    if seg_mask is None:
        return None, None

    h, w = depth_mm.shape
    mask = cv2.resize(seg_mask.astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

    fx, cx_c = CAL['fx'], CAL['cx']
    left_pts  = []
    right_pts = []

    # Sample every 10 rows from bottom to top
    for v in range(h - 5, int(CAL['cy']), -10):
        row_mask = mask[v, :]
        if not row_mask.any():
            continue

        cols = np.where(row_mask)[0]
        u_left  = cols[0]
        u_right = cols[-1]

        # Get depth at these boundary pixels
        # Sample a small horizontal strip for robustness
        for u_bnd, pts_list in [(u_left, left_pts), (u_right, right_pts)]:
            u_lo = max(0, u_bnd - 3)
            u_hi = min(w, u_bnd + 4)
            strip = depth_mm[max(0,v-2):v+3, u_lo:u_hi].astype(np.float32)
            valid = strip[(strip > DEPTH_MIN_MM) & (strip < DEPTH_MAX_MM)]
            if len(valid) < 2:
                continue
            z_mm = float(np.median(valid))
            z_m  = z_mm / 1000.0
            x_m  = -(u_bnd - cx_c) * z_m / fx
            pts_list.append((x_m, z_m))

    return left_pts, right_pts


# ══════════════════════════════════════════════════════════════════════════════
#  Camera BEV — data-driven homography from floor pixel depth correspondences
# ══════════════════════════════════════════════════════════════════════════════
_cam_bev_H         = None   # cached homography matrix
_cam_bev_H_version = -1     # seg mask version used to build _cam_bev_H


def render_cam_bev(rgb, depth_mm):
    """
    Compute a top-down BEV by finding a homography from floor pixel
    depth correspondences:
      1. Sample floor pixels (u,v) with valid depth z from FastSAM mask.
      2. Back-project: X = -(u-cx)*z/fx  (camera frame, mm)
      3. Map to BEV canvas: col=(R-X)/S, row=(out_size-1)-z/S
      4. cv2.findHomography(img_pts, bev_pts, RANSAC)  → H
      5. warpPerspective full RGB — no masking.
    H is cached and only recomputed when a new seg mask arrives.
    Returns a (CAM_BEV_SIZE × CAM_BEV_SIZE) BGR image.
    """
    global _cam_bev_H, _cam_bev_H_version

    blank = np.zeros((CAM_BEV_SIZE, CAM_BEV_SIZE, 3), dtype=np.uint8)
    if rgb is None:
        return blank

    h, w   = rgb.shape[:2]
    out_sz = CAM_BEV_SIZE
    # Forward: 0 → CAM_BEV_RANGE_M.  Lateral: ±CAM_BEV_RANGE_M/2
    S      = CAM_BEV_RANGE_M * 1000.0 / out_sz  # mm per BEV pixel
    R_mm   = (out_sz // 2) * S                  # lateral half-range in mm
    fx     = CAL['fx']
    cx_c   = CAL['cx']

    # ── Recompute H when a new floor mask is available ────────────────────
    with _seg_lock:
        cur_version = _seg_mask_version
        cur_mask    = _seg_mask

    if cur_mask is not None and (
            _cam_bev_H is None or cur_version != _cam_bev_H_version):
        if depth_mm is not None:
            fm = cv2.resize(cur_mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            d = depth_mm.astype(np.float32)

            vv_idx, uu_idx = np.where(fm)
            z_all = d[vv_idx, uu_idx]
            valid = (z_all > DEPTH_MIN_MM) & (z_all < DEPTH_MAX_MM)

            if valid.sum() >= 20:
                z   = z_all[valid]
                u   = uu_idx[valid].astype(np.float32)
                v   = vv_idx[valid].astype(np.float32)
                x   = -(u - cx_c) * z / fx

                bev_col = (R_mm - x) / S
                bev_row = (out_sz - 1) - z / S

                inside = ((bev_col >= 0) & (bev_col < out_sz) &
                          (bev_row >= 0) & (bev_row < out_sz))

                if inside.sum() >= 20:
                    img_pts = np.column_stack([u[inside],
                                               v[inside]]).astype(np.float32)
                    bev_pts = np.column_stack([bev_col[inside],
                                               bev_row[inside]]).astype(np.float32)
                    H, _ = cv2.findHomography(img_pts, bev_pts,
                                              cv2.RANSAC, 3.0)
                    if H is not None:
                        _cam_bev_H         = H
                        _cam_bev_H_version = cur_version

    if _cam_bev_H is None:
        return blank

    # ── Warp full RGB — no masking ────────────────────────────────────────
    result = cv2.warpPerspective(
        rgb, _cam_bev_H, (out_sz, out_sz),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

    # ── Grid lines ────────────────────────────────────────────────────────
    cells_per_m = out_sz / CAM_BEV_RANGE_M
    centre = out_sz // 2
    for d in range(1, int(CAM_BEV_RANGE_M) + 1):
        ry = int(out_sz - 1 - d * cells_per_m)
        if 0 <= ry < out_sz:
            cv2.line(result, (0, ry), (out_sz - 1, ry), (55, 55, 55), 1)
            cv2.putText(result, f"{d}m", (centre + 3, ry - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1)
    cv2.line(result, (centre, 0), (centre, out_sz - 1), (55, 55, 55), 1)

    # ── Robot marker at bottom-centre ─────────────────────────────────────
    cv2.circle(result, (centre, out_sz - 5), 4, (0, 220, 0), -1)
    cv2.arrowedLine(result, (centre, out_sz - 5),
                    (centre, out_sz - 22), (255, 180, 0), 2, tipLength=0.35)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Schematic BEV rendering
# ══════════════════════════════════════════════════════════════════════════════
def _world_to_bev(x_m, z_m):
    """Convert world (x_m, z_m) to BEV pixel (px, py)."""
    px = int(BEV_W / 2 - x_m * _PPM_X)         # positive x = left → smaller px
    py = int(BEV_H - 60 - z_m * _PPM_Z)         # bottom = ego, top = far
    return px, py


def render_schematic(measured, left_boundary, right_boundary):
    """
    Draw the clean schematic BEV diagram.

    measured:  list of dicts from measure_detections()
    left/right_boundary:  lists of (x_m, z_m) tuples
    """
    bev = np.full((BEV_H, BEV_W, 3), COL_BG, dtype=np.uint8)

    # ── Road surface ──────────────────────────────────────────────────────
    # If we have boundaries, fill between them; otherwise use default width
    if left_boundary and right_boundary and len(left_boundary) > 2 and len(right_boundary) > 2:
        # Build polygon from boundary points
        left_px  = [_world_to_bev(x, z) for x, z in left_boundary]
        right_px = [_world_to_bev(x, z) for x, z in right_boundary]
        # Polygon: left boundary (bottom to top) + right boundary (top to bottom)
        poly = np.array(left_px + right_px[::-1], dtype=np.int32)
        cv2.fillPoly(bev, [poly], COL_ROAD)
        # Draw boundary lines
        for pts_list in [left_px, right_px]:
            for i in range(len(pts_list) - 1):
                cv2.line(bev, pts_list[i], pts_list[i+1], COL_BOUNDARY, 2, cv2.LINE_AA)
    else:
        # Default road: fixed width centred
        half_road_px = int(BEV_ROAD_WIDTH_M / 2 * _PPM_X)
        cx = BEV_W // 2
        cv2.rectangle(bev, (cx - half_road_px, 0), (cx + half_road_px, BEV_H - 60), COL_ROAD, -1)
        cv2.line(bev, (cx - half_road_px, 0), (cx - half_road_px, BEV_H - 60), COL_BOUNDARY, 2)
        cv2.line(bev, (cx + half_road_px, 0), (cx + half_road_px, BEV_H - 60), COL_BOUNDARY, 2)

    # ── Distance grid lines (every 1 m) ──────────────────────────────────
    for d in range(1, int(BEV_RANGE_FWD_M) + 1):
        _, py = _world_to_bev(0, d)
        if 0 <= py < BEV_H:
            cv2.line(bev, (0, py), (BEV_W, py), COL_GRID, 1)
            cv2.putText(bev, f"{d}m", (BEV_W - 35, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_DIST, 1)

    # ── Centre line (dashed) ──────────────────────────────────────────────
    cx = BEV_W // 2
    for y in range(0, BEV_H - 60, 16):
        cv2.line(bev, (cx, y), (cx, min(y + 8, BEV_H - 60)), COL_GRID, 1)

    # ── Detected obstacles as rectangles ──────────────────────────────────
    for i, m in enumerate(measured):
        px, py = _world_to_bev(m['x_m'], m['z_m'])
        half_w = max(int(m['w_m'] * _PPM_X / 2), 4)
        half_d = max(int(m['d_m'] * _PPM_Z / 2), 4)

        # Rectangle
        x1r = px - half_w
        x2r = px + half_w
        y1r = py - half_d
        y2r = py + half_d

        cv2.rectangle(bev, (x1r, y1r), (x2r, y2r), COL_OBS_FILL, -1)
        cv2.rectangle(bev, (x1r, y1r), (x2r, y2r), COL_OBS_EDGE, 2)

        # Label
        label = f"{m['name']}"
        cv2.putText(bev, label, (x1r, y1r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_TEXT, 1)

        # Distance annotation
        dist_str = f"{m['z_m']:.1f}m"
        cv2.putText(bev, dist_str, (x1r, y2r + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, COL_DIST, 1)

    # ── Distance lines between consecutive obstacles ──────────────────────
    for i in range(len(measured) - 1):
        m1, m2 = measured[i], measured[i + 1]
        _, py1 = _world_to_bev(m1['x_m'], m1['z_m'])
        _, py2 = _world_to_bev(m2['x_m'], m2['z_m'])
        mid_x = BEV_W - 55
        gap_m = abs(m2['z_m'] - m1['z_m'])
        # Draw bracket
        cv2.line(bev, (mid_x, py1), (mid_x, py2), (100, 180, 255), 1)
        cv2.line(bev, (mid_x - 4, py1), (mid_x + 4, py1), (100, 180, 255), 1)
        cv2.line(bev, (mid_x - 4, py2), (mid_x + 4, py2), (100, 180, 255), 1)
        cv2.putText(bev, f"{gap_m:.1f}m", (mid_x + 6, (py1 + py2) // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 180, 255), 1)

    # ── Ego vehicle at bottom ─────────────────────────────────────────────
    ego_cx, ego_cy = BEV_W // 2, BEV_H - 30
    ego_w, ego_h = 20, 35
    cv2.rectangle(bev, (ego_cx - ego_w//2, ego_cy - ego_h//2),
                       (ego_cx + ego_w//2, ego_cy + ego_h//2), COL_EGO, -1)
    cv2.rectangle(bev, (ego_cx - ego_w//2, ego_cy - ego_h//2),
                       (ego_cx + ego_w//2, ego_cy + ego_h//2), (255,255,255), 1)
    # Heading arrow
    cv2.arrowedLine(bev, (ego_cx, ego_cy - ego_h//2),
                         (ego_cx, ego_cy - ego_h//2 - 15),
                         (255, 255, 255), 2, tipLength=0.4)

    # ── Info labels ───────────────────────────────────────────────────────
    cv2.putText(bev, f"depth: {DEPTH_SAMPLE}  [D]=cycle  [Q]=quit",
                (5, BEV_H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.30, COL_DIST, 1)
    cv2.putText(bev, f"objects: {len(measured)}",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_TEXT, 1)

    return bev


# ══════════════════════════════════════════════════════════════════════════════
#  Draw detections on RGB for debugging
# ══════════════════════════════════════════════════════════════════════════════
def draw_rgb_detections(rgb, detections, measured):
    """Draw bounding boxes + distance on the RGB frame."""
    out = rgb.copy()
    measured_bboxes = {tuple(m['bbox']): m for m in measured}

    for (x1, y1, x2, y2, name, conf) in detections:
        m = measured_bboxes.get((x1, y1, x2, y2))
        color = (0, 255, 0) if m else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = name
        if m:
            label += f" {m['z_m']:.1f}m"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _running, _det_busy, _seg_busy

    cam = threading.Thread(target=camera_thread, daemon=True)
    cam.start()
    print(f"[main] {RGB_W}×{RGB_H}  Q=quit")

    frame_count = 0

    while _running:
        with _lock:
            depth_mm = _latest_depth
            rgb_img  = _latest_rgb

        if depth_mm is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_count += 1

        # ── Run YOLO detection ────────────────────────────────────────────
        if frame_count % DET_EVERY_N == 0 and rgb_img is not None and not _det_busy:
            _det_busy = True
            threading.Thread(target=_det_worker,
                             args=(rgb_img.copy(),), daemon=True).start()

        # ── Run floor segmentation ────────────────────────────────────────
        if frame_count % SEG_EVERY_N == 0 and rgb_img is not None and not _seg_busy:
            _seg_busy = True
            threading.Thread(target=_seg_worker,
                             args=(rgb_img.copy(),), daemon=True).start()

        # ── Get latest results ────────────────────────────────────────────
        with _det_lock:
            detections = _detections.copy()
        with _seg_lock:
            seg_mask = _seg_mask

        # ── Measure each detection ────────────────────────────────────────
        measured = measure_detections(depth_mm, detections)

        # ── Road boundaries ───────────────────────────────────────────────
        left_bnd, right_bnd = get_road_boundaries(seg_mask, depth_mm)

        # ── Render schematic BEV ──────────────────────────────────────────
        bev = render_schematic(measured, left_bnd, right_bnd)
        cv2.imshow("BEV", bev)

        # ── Camera BEV (IPM warp of actual RGB image) ────────────────────
        cv2.imshow("Camera BEV", render_cam_bev(rgb_img, depth_mm))

        # ── RGB with detections ───────────────────────────────────────────
        if rgb_img is not None:
            rgb_disp = draw_rgb_detections(rgb_img, detections, measured)
            # Also draw floor overlay
            if seg_mask is not None:
                m = cv2.resize(seg_mask.astype(np.uint8),
                               (rgb_img.shape[1], rgb_img.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                rgb_disp = rgb_disp.astype(np.float32)
                rgb_disp[m] = 0.5 * rgb_disp[m] + 0.5 * np.array([0, 200, 0], np.float32)
                rgb_disp = rgb_disp.astype(np.uint8)
            cv2.imshow("RGB", rgb_disp)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _running = False
            break
        elif key == ord('d'):
            global DEPTH_SAMPLE
            modes = ["bottom", "center", "full"]
            idx = (modes.index(DEPTH_SAMPLE) + 1) % len(modes)
            DEPTH_SAMPLE = modes[idx]
            print(f"[depth] sampling mode → {DEPTH_SAMPLE}")

    cv2.destroyAllWindows()
    cam.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()