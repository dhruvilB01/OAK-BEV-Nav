#!/usr/bin/env python3
"""
bev_nav.py — OAK-D stereo depth + SegFormer-B2 → semantic BEV costmap
             + YOLO-World open-vocabulary 3-D object detection

Semantic BEV layers (paint priority: higher index wins):
  0  UNKNOWN      black        no depth data
  1  TRAVERSABLE  blue         floor / road / sidewalk / ground
  2  VEGETATION   dark green   grass / trees / plants
  3  FURNITURE    purple       indoor obstacles (chair, table, sofa…)
  4  STRUCTURE    dark red     walls / building / fence / ceiling
  5  VEHICLE      yellow       car / truck / bus / van / bicycle
  6  PERSON       cyan         person

Navigation overlay (drawn on top, semi-transparent):
  INFLATION ring   orange      within robot radius of any non-traversable cell

YOLO-World detected objects → cyan labelled footprint boxes on BEV.
RGB view: magenta floor overlay + YOLO bounding boxes.

World coords:  X = left,  Y = up,  Z = forward
Keys:  Q = quit   R = reset BEV
"""

import threading
import time

import cv2
import depthai as dai
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (SegformerForSemanticSegmentation,
                          SegformerImageProcessor)
from ultralytics import YOLOWorld


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
RGB_W, RGB_H = 640, 400

CELL_MM      = 40           # mm per BEV grid cell
BEV_X_MM     = 4000         # ± lateral extent
BEV_Z_MIN_MM = 200          # nearest depth plane
BEV_Z_MAX_MM = 6000         # farthest depth plane

ROBOT_RADIUS_MM = 300       # inflation radius (mm)
FLOOR_Y_MM      = -200      # world Y below this → ground plane
                            # ≈ -(camera mount height in mm)

# SegFormer model  (B2 = best quality at ~8ms on RTX 5080 fp16)
SEG_MODEL    = "nvidia/segformer-b2-finetuned-ade-512-512"
SEG_INTERVAL = 0.04         # seconds between seg inference (~25fps)

# YOLO-World
DETECT_CLASSES  = ["person", "chair", "table", "cardboard box",
                   "door", "bag", "bicycle", "trash can", "car"]
DETECT_CONF     = 0.25
DETECT_INTERVAL = 0.10      # seconds between detection inference

DEPTH_EMA_ALPHA = 0.25

# BEV decay: each frame unseen cells fade toward UNKNOWN
BEV_DECAY = 0.88            # multiply semantic confidence each frame
# ═══════════════════════════════════════════════════════════════════════════════

BEV_W       = int(2 * BEV_X_MM / CELL_MM)
BEV_H       = int((BEV_Z_MAX_MM - BEV_Z_MIN_MM) / CELL_MM)
ROBOT_CELLS = max(1, int(ROBOT_RADIUS_MM / CELL_MM))


# ── ADE20K → semantic layer mapping ──────────────────────────────────────────
# ADE20K 150-class indices → layer index (0=unknown, see LAYER_* consts below)
LAYER_UNKNOWN     = 0
LAYER_TRAVERSABLE = 1
LAYER_VEGETATION  = 2
LAYER_FURNITURE   = 3
LAYER_STRUCTURE   = 4
LAYER_VEHICLE     = 5
LAYER_PERSON      = 6

# BGR colors for each layer  (B, G, R)
LAYER_COLORS = {
    LAYER_UNKNOWN:     (   0,   0,   0),   # black
    LAYER_TRAVERSABLE: ( 230, 100,   0),   # blue  — road / floor
    LAYER_VEGETATION:  (   0, 170,   0),   # green — trees / grass
    LAYER_FURNITURE:   ( 200,   0, 200),   # magenta — indoor obstacles
    LAYER_STRUCTURE:   (  22,  22,  22),   # near-black — walls / buildings
    LAYER_VEHICLE:     (   0, 230, 230),   # yellow — cars
    LAYER_PERSON:      ( 255, 255,   0),   # cyan — people
}

# ADE20K class index → layer (unlisted = UNKNOWN → not painted)
_ADE_TO_LAYER = {
    # Traversable ground
    3:  LAYER_TRAVERSABLE,   # floor
    6:  LAYER_TRAVERSABLE,   # road
    11: LAYER_TRAVERSABLE,   # sidewalk
    13: LAYER_TRAVERSABLE,   # earth / ground
    52: LAYER_TRAVERSABLE,   # path
    54: LAYER_TRAVERSABLE,   # runway
    91: LAYER_TRAVERSABLE,   # dirt track
    94: LAYER_TRAVERSABLE,   # land / soil
    # Vegetation
    4:  LAYER_VEGETATION,    # tree
    9:  LAYER_VEGETATION,    # grass
    17: LAYER_VEGETATION,    # plant
    29: LAYER_VEGETATION,    # field
    66: LAYER_VEGETATION,    # flower
    72: LAYER_VEGETATION,    # palm
    # Furniture / indoor obstacles
    7:  LAYER_FURNITURE,     # bed
    15: LAYER_FURNITURE,     # table
    19: LAYER_FURNITURE,     # chair
    23: LAYER_FURNITURE,     # sofa
    24: LAYER_FURNITURE,     # shelf
    30: LAYER_FURNITURE,     # armchair
    31: LAYER_FURNITURE,     # seat
    33: LAYER_FURNITURE,     # desk
    41: LAYER_FURNITURE,     # box
    64: LAYER_FURNITURE,     # coffee table
    97: LAYER_FURNITURE,     # ottoman
    110: LAYER_FURNITURE,    # stool
    # Structure / hard obstacles
    0:  LAYER_STRUCTURE,     # wall
    1:  LAYER_STRUCTURE,     # building
    10: LAYER_STRUCTURE,     # cabinet
    25: LAYER_STRUCTURE,     # house
    32: LAYER_STRUCTURE,     # fence
    38: LAYER_STRUCTURE,     # railing
    42: LAYER_STRUCTURE,     # column
    48: LAYER_STRUCTURE,     # skyscraper
    53: LAYER_STRUCTURE,     # stairs
    59: LAYER_STRUCTURE,     # stairway
    84: LAYER_STRUCTURE,     # tower
    93: LAYER_STRUCTURE,     # pole
    # Vehicles
    20: LAYER_VEHICLE,       # car
    80: LAYER_VEHICLE,       # bus
    83: LAYER_VEHICLE,       # truck
    102: LAYER_VEHICLE,      # van
    116: LAYER_VEHICLE,      # motorbike
    127: LAYER_VEHICLE,      # bicycle
    # Person
    12: LAYER_PERSON,        # person
}

# Build lookup table (150 entries)
_ADE_LUT = np.zeros(150, dtype=np.uint8)   # default → UNKNOWN
for ade_id, layer in _ADE_TO_LAYER.items():
    if ade_id < 150:
        _ADE_LUT[ade_id] = layer


# Inflation kernel (circular, radius = robot_cells)
def _circle_kernel(r):
    d = 2 * r + 1
    k = np.zeros((d, d), np.uint8)
    cv2.circle(k, (r, r), r, 1, -1)
    return k

_INF_KERNEL  = _circle_kernel(ROBOT_CELLS)
_FILL_KERNEL = np.ones((5, 5), np.uint8)   # morphological close for gap filling

# FOV frustum mask — built lazily after calibration is finalised
_fov_mask: np.ndarray | None = None


def _build_fov_mask() -> np.ndarray:
    """
    Returns a (BEV_H, BEV_W) bool mask that is True only inside the
    camera's visible frustum (trapezoid: narrow near, wide far).
    """
    def _row(z):
        return int(np.clip(BEV_H - 1 - (z - BEV_Z_MIN_MM) / CELL_MM, 0, BEV_H - 1))

    def _col(x):
        return int(np.clip((x + BEV_X_MM) / CELL_MM, 0, BEV_W - 1))

    # X world-coordinate of left / right image edge at depth z:
    #   x_left  =  cx * z / fx   (pixel u=0    → +X direction)
    #   x_right = -(W-1-cx) * z / fx   (pixel u=W-1 → −X direction)
    def _x_left(z):  return  CAL['cx']           * z / CAL['fx']
    def _x_right(z): return -(RGB_W - 1 - CAL['cx']) * z / CAL['fx']

    poly = np.array([
        [_col(_x_left(BEV_Z_MIN_MM)),  _row(BEV_Z_MIN_MM)],   # near-left
        [_col(_x_right(BEV_Z_MIN_MM)), _row(BEV_Z_MIN_MM)],   # near-right
        [_col(_x_right(BEV_Z_MAX_MM)), _row(BEV_Z_MAX_MM)],   # far-right
        [_col(_x_left(BEV_Z_MAX_MM)),  _row(BEV_Z_MAX_MM)],   # far-left
    ], dtype=np.int32)

    m = np.zeros((BEV_H, BEV_W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


# Jet LUT
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
        print(f"[cal] rescaled for {w}×{h} → fx={CAL['fx']:.1f}")


# ── Depth EMA ─────────────────────────────────────────────────────────────────
_DEPTH_EMA: np.ndarray | None = None


def apply_depth_ema(raw: np.ndarray) -> np.ndarray:
    global _DEPTH_EMA
    f = raw.astype(np.float32)
    if _DEPTH_EMA is None or _DEPTH_EMA.shape != f.shape:
        _DEPTH_EMA = f.copy()
        return raw
    both     = (f > 0) & (_DEPTH_EMA > 0)
    new_only = (f > 0) & (_DEPTH_EMA == 0)
    lost     = (f == 0) & (_DEPTH_EMA > 0)
    _DEPTH_EMA[both]     = DEPTH_EMA_ALPHA * f[both] + (1 - DEPTH_EMA_ALPHA) * _DEPTH_EMA[both]
    _DEPTH_EMA[new_only] = f[new_only]
    _DEPTH_EMA[lost]    *= 0.7
    _DEPTH_EMA[_DEPTH_EMA < 1] = 0
    return _DEPTH_EMA.astype(np.uint16)


# ── Floor mask (for RGB overlay) ──────────────────────────────────────────────
def make_floor_mask(depth_mm: np.ndarray) -> np.ndarray:
    h  = depth_mm.shape[0]
    vv = np.arange(h)[:, None]
    d  = depth_mm.astype(np.float32)
    valid = d > 0
    z = np.where(valid, d, 1.0)
    y_world = -(vv - CAL['cy']) * z / CAL['fy']
    return valid & (y_world < FLOOR_Y_MM)


# ── Semantic BEV ──────────────────────────────────────────────────────────────
# Persistent BEV: layer index per cell + confidence [0,1] that decays
_bev_layer = np.zeros((BEV_H, BEV_W), dtype=np.uint8)    # layer index
_bev_conf  = np.zeros((BEV_H, BEV_W), dtype=np.float32)  # confidence


def reset_bev():
    _bev_layer[:] = LAYER_UNKNOWN
    _bev_conf[:]  = 0.0


# Pre-build pixel coordinate grids (reused every frame)
_grid_u: np.ndarray | None = None
_grid_v: np.ndarray | None = None


def _get_grids(h, w):
    global _grid_u, _grid_v
    if _grid_u is None or _grid_u.shape != (h, w):
        _grid_u, _grid_v = np.meshgrid(np.arange(w), np.arange(h))
    return _grid_u, _grid_v


def update_bev_semantic(depth_mm: np.ndarray,
                        seg_mask: np.ndarray) -> np.ndarray:
    """
    Project depth+semantics onto BEV grid.
    Returns rendered BGR BEV image (with inflation overlay).
    """
    # --- decay old observations ---
    _bev_conf[:] *= BEV_DECAY
    faded = _bev_conf < 0.05
    _bev_layer[faded] = LAYER_UNKNOWN
    _bev_conf[faded]  = 0.0

    h, w = depth_mm.shape
    uu, vv = _get_grids(h, w)

    d = depth_mm.astype(np.float32)
    valid = (d > BEV_Z_MIN_MM) & (d < BEV_Z_MAX_MM)

    z = d[valid]
    x = -(uu[valid] - CAL['cx']) * z / CAL['fx']
    y = -(vv[valid] - CAL['cy']) * z / CAL['fy']

    # Map to BEV grid cells
    col = ((x + BEV_X_MM) / CELL_MM).astype(np.int32)
    row = (BEV_H - 1 - (z - BEV_Z_MIN_MM) / CELL_MM).astype(np.int32)
    in_grid = (col >= 0) & (col < BEV_W) & (row >= 0) & (row < BEV_H)

    col = col[in_grid]
    row = row[in_grid]
    z_g = z[in_grid]
    y_g = y[in_grid]

    # Semantic labels for valid+in-grid pixels
    seg_flat = seg_mask[valid][in_grid].astype(np.int32)
    seg_flat = np.clip(seg_flat, 0, 149)
    layers   = _ADE_LUT[seg_flat]

    # Y-based fallback for pixels the seg model left as UNKNOWN:
    #   below floor threshold  → TRAVERSABLE
    #   above floor threshold  → STRUCTURE (generic obstacle)
    unknown = layers == LAYER_UNKNOWN
    layers[unknown & (y_g < FLOOR_Y_MM)]  = LAYER_TRAVERSABLE
    layers[unknown & (y_g >= FLOOR_Y_MM)] = LAYER_STRUCTURE

    # Sort far → near so nearer points overwrite stale far ones
    order = np.argsort(-z_g)
    col, row, layers = col[order], row[order], layers[order]

    # Paint: obstacles always overwrite traversable; same-priority overwrites
    current = _bev_layer[row, col]
    overwrite = (
        (layers > LAYER_TRAVERSABLE) |            # obstacle always wins over floor
        ((layers == LAYER_TRAVERSABLE) & (current == LAYER_UNKNOWN)) |
        ((layers == current) & (layers != LAYER_UNKNOWN))
    )
    _bev_layer[row[overwrite], col[overwrite]] = layers[overwrite]
    _bev_conf[ row[overwrite], col[overwrite]] = 1.0

    return _render_bev()


def _render_bev() -> np.ndarray:
    global _fov_mask

    # Build FOV mask once (needs CAL to be finalised)
    if _fov_mask is None and _cal_adjusted:
        _fov_mask = _build_fov_mask()

    # ── Stage 1: column-wise floor fill (key technique from the paper) ────────
    # For each BEV column, find the nearest (lowest row = closest to camera)
    # non-UNKNOWN cell. Everything below it (between camera and that point)
    # that is still UNKNOWN gets filled as TRAVERSABLE.
    # This produces the solid floor wedge visible in image 2.
    filled = _bev_layer.copy()

    # Inside-FOV mask (all True if not yet built)
    fov = _fov_mask if _fov_mask is not None else np.ones((BEV_H, BEV_W), bool)

    # known[r,c] = True if that cell has a real detection (not UNKNOWN)
    known = (filled != LAYER_UNKNOWN)

    # For each column, find the highest row index (nearest to camera) that is known
    # np.argmax on flipped axis gives first True from bottom
    has_any = known.any(axis=0)                         # (BEV_W,)
    # contact_row[c] = row of nearest detection in column c (BEV_H-1 if none)
    contact_row = np.where(has_any,
                           BEV_H - 1 - np.argmax(known[::-1, :], axis=0),
                           -1)                          # -1 = no detection

    # Build fill mask: rows below contact_row, inside FOV, currently UNKNOWN
    rows = np.arange(BEV_H)[:, None]                   # (BEV_H, 1)
    below_contact = rows > contact_row[None, :]         # (BEV_H, BEV_W)
    fill_mask = below_contact & fov & (filled == LAYER_UNKNOWN)
    filled[fill_mask] = LAYER_TRAVERSABLE

    # ── Stage 2: small morphological close to smooth obstacle boundaries ──────
    for layer_id in [LAYER_VEGETATION, LAYER_FURNITURE,
                     LAYER_STRUCTURE,  LAYER_VEHICLE, LAYER_PERSON]:
        m = (filled == layer_id).astype(np.uint8)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _FILL_KERNEL)
        new_cells = (closed > 0) & (filled == LAYER_UNKNOWN)
        filled[new_cells] = layer_id

    # ── Stage 3: paint image ──────────────────────────────────────────────────
    img = np.zeros((BEV_H, BEV_W, 3), np.uint8)
    for layer_id in range(LAYER_UNKNOWN + 1, LAYER_PERSON + 1):
        mask = filled == layer_id
        if mask.any():
            img[mask] = LAYER_COLORS[layer_id]

    # ── Stage 4: FOV frustum — black outside camera view ─────────────────────
    if _fov_mask is not None:
        img[~_fov_mask] = 0

    # ── Stage 5: inflation ring ───────────────────────────────────────────────
    obstacle_mask = np.isin(filled,
                            [LAYER_FURNITURE, LAYER_STRUCTURE,
                             LAYER_VEHICLE,   LAYER_PERSON,
                             LAYER_VEGETATION]).astype(np.uint8)
    dilated  = cv2.dilate(obstacle_mask, _INF_KERNEL)
    inf_mask = (dilated > 0) & (obstacle_mask == 0) & (filled == LAYER_TRAVERSABLE)
    img[inf_mask] = (img[inf_mask].astype(np.float32) * 0.35 +
                     np.array([0, 130, 255], np.float32) * 0.65).astype(np.uint8)

    return img


def draw_bev_overlay(bev_img: np.ndarray, detections: list) -> np.ndarray:
    out = bev_img.copy()
    cx  = BEV_W // 2

    # Grid
    cv2.line(out, (cx, 0), (cx, BEV_H), (55, 55, 55), 1)
    for d_m in range(1, BEV_Z_MAX_MM // 1000 + 1):
        r = BEV_H - 1 - int((d_m * 1000 - BEV_Z_MIN_MM) / CELL_MM)
        if 0 <= r < BEV_H:
            cv2.line(out, (0, r), (BEV_W, r), (55, 55, 55), 1)
            cv2.putText(out, f"{d_m}m", (2, r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 120, 120), 1)

    # Camera marker
    cv2.drawMarker(out, (cx, BEV_H - 1), (0, 255, 0),
                   cv2.MARKER_TRIANGLE_UP, 12, 2)

    # YOLO-World detection footprints
    for det in detections:
        label, conf     = det['label'], det['conf']
        cx3, cz3        = det['cx3'],   det['cz3']
        hw, hd          = det['half_w'], det['half_d']
        c1 = int((cx3 - hw + BEV_X_MM) / CELL_MM)
        c2 = int((cx3 + hw + BEV_X_MM) / CELL_MM)
        r1 = int(BEV_H - 1 - (cz3 + hd - BEV_Z_MIN_MM) / CELL_MM)
        r2 = int(BEV_H - 1 - (cz3 - hd - BEV_Z_MIN_MM) / CELL_MM)
        c1, c2 = sorted([c1, c2])
        r1, r2 = sorted([r1, r2])
        cv2.rectangle(out, (c1, r1), (c2, r2), (255, 255, 255), 1)
        cv2.putText(out, f"{label} {conf:.0%}", (max(c1, 0), max(r1 - 2, 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)

    # Legend
    legend = [
        ("ROAD/FLOOR",  LAYER_COLORS[LAYER_TRAVERSABLE]),
        ("VEGETATION",  LAYER_COLORS[LAYER_VEGETATION]),
        ("FURNITURE",   LAYER_COLORS[LAYER_FURNITURE]),
        ("STRUCTURE",   LAYER_COLORS[LAYER_STRUCTURE]),
        ("VEHICLE",     LAYER_COLORS[LAYER_VEHICLE]),
        ("PERSON",      LAYER_COLORS[LAYER_PERSON]),
        ("INFLATION",   (0, 140, 255)),
    ]
    for i, (name, color) in enumerate(reversed(legend)):
        y = BEV_H - 6 - i * 11
        cv2.rectangle(out, (2, y - 7), (11, y), color, -1)
        cv2.putText(out, name, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.27, (200, 200, 200), 1)

    return out


# ── RGB annotator ─────────────────────────────────────────────────────────────
_MAGENTA = np.array([255, 0, 255], np.float32)


def annotate_rgb(rgb: np.ndarray, depth_mm: np.ndarray,
                 detections: list) -> np.ndarray:
    # Floor magenta overlay
    mask = make_floor_mask(depth_mm)
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8),
                          (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    out = rgb.astype(np.float32)
    out[mask] = 0.55 * out[mask] + 0.45 * _MAGENTA
    out = out.astype(np.uint8)

    # YOLO detections
    for det in detections:
        x1, y1, x2, y2 = det['box2d']
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{det['label']} {det['conf']:.0%}",
                    (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return out


# ── Semantic segmentation overlay (debug) ─────────────────────────────────────
def render_seg_overlay(rgb: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
    """Colour-coded segmentation mask blended over RGB."""
    color_img = np.zeros_like(rgb)
    for ade_id, layer in _ADE_TO_LAYER.items():
        m = seg_mask == ade_id
        if m.any():
            color_img[m] = LAYER_COLORS[layer]
    if seg_mask.shape != rgb.shape[:2]:
        color_img = cv2.resize(color_img,
                               (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(rgb, 0.45, color_img, 0.55, 0)


# ── Depth coloriser ───────────────────────────────────────────────────────────
def colorize_depth(d: np.ndarray) -> np.ndarray:
    v = d[d > 0]
    if v.size == 0:
        return np.zeros((*d.shape, 3), np.uint8)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    n = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    out = _LUT[(n * 255).astype(np.uint8), 0].copy()
    out[d == 0] = 0
    return out


def bbox_depth_sample(depth_mm, x1, y1, x2, y2):
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    bw = max(1, (x2 - x1) // 4)
    bh = max(1, (y2 - y1) // 4)
    roi = depth_mm[max(0, my - bh):my + bh, max(0, mx - bw):mx + bw]
    v = roi[roi > 0]
    return float(np.median(v)) if v.size > 0 else None


# ── Shared state ──────────────────────────────────────────────────────────────
_cam_lock     = threading.Lock()
_latest_depth: np.ndarray | None = None
_latest_rgb:   np.ndarray | None = None

_seg_lock     = threading.Lock()
_latest_seg:  np.ndarray | None = None   # H×W uint8 ADE20K class IDs

_det_lock     = threading.Lock()
_detections:  list = []

_running = True


# ── Segmentation thread ───────────────────────────────────────────────────────
def seg_thread():
    global _latest_seg, _running

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[seg] Loading {SEG_MODEL} on {device} fp16={dtype==torch.float16} …")

    processor = SegformerImageProcessor.from_pretrained(SEG_MODEL)
    model     = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL)
    model.to(device, dtype=dtype).eval()
    print("[seg] Ready")

    while _running:
        t0 = time.monotonic()

        with _cam_lock:
            rgb = _latest_rgb

        if rgb is None:
            time.sleep(0.05)
            continue

        pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        inputs  = processor(images=pil_img, return_tensors="pt")
        inputs  = {k: v.to(device, dtype=dtype) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits   # [1, 150, H/4, W/4]

        # Upsample to original RGB size
        logits_up = F.interpolate(
            logits.float(),
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        seg = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        with _seg_lock:
            _latest_seg = seg

        elapsed = time.monotonic() - t0
        if elapsed < SEG_INTERVAL:
            time.sleep(SEG_INTERVAL - elapsed)


# ── YOLO-World detection thread ───────────────────────────────────────────────
def detection_thread():
    global _detections, _running

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[det] Loading YOLO-World on {device} …")
    model = YOLOWorld("yolov8s-worldv2.pt")
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

            px_cx = (x1 + x2) / 2.0
            cx3   = -(px_cx - CAL['cx']) * z / CAL['fx']
            half_w = abs((x2 - x1) / 2.0 * z / CAL['fx'])
            half_d = max(half_w * 0.6, 150.0)

            dets.append(dict(label=label, conf=conf,
                             box2d=(x1, y1, x2, y2),
                             cx3=cx3, cz3=z,
                             half_w=half_w, half_d=half_d))

        with _det_lock:
            _detections = dets

        elapsed = time.monotonic() - t0
        if elapsed < DETECT_INTERVAL:
            time.sleep(DETECT_INTERVAL - elapsed)


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
        cfg.postProcessing.spatialFilter.enable            = True
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations     = 1
        cfg.postProcessing.spatialFilter.alpha             = 0.5
        cfg.postProcessing.spatialFilter.delta             = 20
        cfg.postProcessing.temporalFilter.enable           = True
        cfg.postProcessing.temporalFilter.alpha            = 0.2
        cfg.postProcessing.speckleFilter.enable            = True
        cfg.postProcessing.speckleFilter.speckleRange      = 50
        stereo.initialConfig.set(cfg)
        print("[stereo] filters: 7×7 median + spatial + temporal + speckle")
    except Exception as e:
        print(f"[stereo] filter config failed ({e}), median only")

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
    seg_t = threading.Thread(target=seg_thread,      daemon=True)
    det_t = threading.Thread(target=detection_thread, daemon=True)
    cam_t.start()
    seg_t.start()
    det_t.start()

    scale        = min(1.0, 700 / BEV_H)
    bev_disp     = (int(BEV_W * scale), int(BEV_H * scale))
    show_seg_dbg = False   # toggle with 'S'

    print(f"BEV {BEV_W}×{BEV_H} @ {CELL_MM}mm/cell | "
          f"robot radius {ROBOT_RADIUS_MM}mm | Q=quit R=reset S=seg-debug")

    while _running:
        with _cam_lock:
            depth_mm = _latest_depth
            rgb_img  = _latest_rgb
        with _seg_lock:
            seg_mask = _latest_seg
        with _det_lock:
            detections = list(_detections)

        if depth_mm is not None and seg_mask is not None:
            bev_img = update_bev_semantic(depth_mm, seg_mask)
            bev_vis = draw_bev_overlay(bev_img, detections)
            cv2.imshow("Semantic BEV",
                       cv2.resize(bev_vis, bev_disp,
                                  interpolation=cv2.INTER_NEAREST))

            if rgb_img is not None:
                cv2.imshow("RGB | floor + detections",
                           annotate_rgb(rgb_img, depth_mm, detections))
                if show_seg_dbg:
                    cv2.imshow("Seg debug",
                               render_seg_overlay(rgb_img, seg_mask))

            cv2.imshow("Depth", colorize_depth(depth_mm))

        elif depth_mm is not None:
            # Seg not ready yet — show depth only
            cv2.imshow("Depth", colorize_depth(depth_mm))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _running = False
            break
        elif key == ord('r'):
            reset_bev()
            print("[map] BEV reset")
        elif key == ord('s'):
            show_seg_dbg = not show_seg_dbg
            if not show_seg_dbg:
                cv2.destroyWindow("Seg debug")

    cv2.destroyAllWindows()
    cam_t.join(timeout=3)
    seg_t.join(timeout=3)
    det_t.join(timeout=3)
    print("Done.")


if __name__ == "__main__":
    main()
