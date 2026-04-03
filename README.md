# OAK-D Real-Time Bird's Eye View

Real-time Bird's Eye View (BEV) pipeline for the Luxonis OAK-D stereo camera. Combines RGB, stereo depth, YOLOv8 object detection, FastSAM floor segmentation, and a data-driven homography to produce a top-down satellite-style view of the scene — no IMU mounting calibration required, no ground height assumption.

---

## Outputs

| Window | Description |
|--------|-------------|
| **Camera BEV** | Homography-warped RGB — top-down view of the real scene |
| **BEV** | Schematic radar diagram with obstacle rectangles and road boundaries |
| **RGB** | Undistorted RGB with YOLO bounding boxes and floor overlay |

---

## System Architecture

```
OAK-D
 ├── CAM_A (RGB, 640×400)  ──────────────────────────────► undistort ──► YOLO / FastSAM / BEV
 ├── CAM_B (left mono)  ─┐
 │                        ├─► StereoDepth (FAST_ACCURACY) ──► depth_mm ──► back-project / BEV
 ├── CAM_C (right mono) ─┘
 └── IMU (BNO085) ──────────────────────────────────────► _imu_R (stored, reserved)
```

All depth output is aligned to CAM_A so every depth pixel maps 1-to-1 with the RGB pixel at the same (u, v).

---

## Camera Coordinate Frame

The OAK-D uses a right-handed camera frame:

```
       Y (up)
       │
       │
       └──── X (left)
      /
     Z (forward, into scene)
```

- **X** increases to the **left** (negative = right)
- **Y** increases **upward**
- **Z** increases **forward** (depth)

This sign convention matters throughout the back-projection and BEV mapping.

---

## Stereo Depth

### Disparity to Depth

The stereo pair (CAM_B, CAM_C) produces a disparity map `d` (pixels). Depth is:

```
Z = fx * baseline / d
```

where `baseline` is the distance between the two lenses (~75 mm on OAK-D) and `fx` is the focal length in pixels.

### Preset: FAST_ACCURACY

Configured with `PresetMode.FAST_ACCURACY`:

- **5-bit subpixel**: disparity steps of 1/32 pixel → finer depth resolution at range
- **LR check (threshold 5)**: pixel must agree between left→right and right→left match; rejects occluded / uncertain pixels
- **No decimation**: full 640×400 depth output, no downsampling
- **No spatial/median filter**: avoids depth bleeding across object edges

Post-processing added on top:

| Filter | Setting | Purpose |
|--------|---------|---------|
| Temporal | α = 0.4 | EMA smoothing across frames for stable long-range readings |
| Speckle | range = 200 | Remove isolated noise pixels, common beyond 10 m |
| Threshold | 300 mm – 50 000 mm | Hard-clip invalid readings at device level |

### Why not Extended Disparity?

Extended disparity halves the minimum measurable depth (useful for < 0.5 m objects) but also halves the **maximum** range. Since we target 15–50 m outdoors, it is left off.

### Depth EMA (software)

A per-pixel exponential moving average is applied in software after receiving raw depth:

```python
depth_ema[t] = α * depth_raw[t] + (1-α) * depth_ema[t-1]    # α = 0.3
```

New pixels that appear in the current frame but had no prior reading are initialised directly. Pixels that go missing (occluded or out of range) decay by 0.7× per frame and are zeroed once they drop below 1 mm.

---

## Intrinsics and Distortion Correction

Calibration is read from device flash (written by the DepthAI Viewer calibration wizard):

```python
cal.getCameraIntrinsics(CAM_A, W, H)           → K  (3×3)
cal.getDistortionCoefficients(CAM_A)            → D  (k1, k2, p1, p2, k3, ...)
cv2.getOptimalNewCameraMatrix(K, D, (W,H), α=0) → K_new  (no black borders)
```

Every RGB frame is undistorted before any processing:

```python
rgb = cv2.undistort(raw, K, D, None, K_new)
```

`K_new` also replaces the raw intrinsics (fx, fy, cx, cy) used in all back-projection math, so pixel coordinates in the undistorted image are consistent with the camera model.

---

## Back-Projection: Pixel → 3D

Given a pixel `(u, v)` with depth `z` (mm), the 3D position in camera frame is:

```
X = -(u - cx) * z / fx      [left positive]
Y = -(v - cy) * z / fy      [up positive]
Z =  z                       [forward]
```

The negation on X and Y follows from the camera frame convention above (increasing u = moving right = decreasing X).

---

## Floor Segmentation — FastSAM

FastSAM-s runs in a background thread using the text prompt:

```
"road or sidewalk or pavement or path"
```

It returns a binary mask at the RGB resolution. The mask is updated asynchronously every `SEG_EVERY_N` frames (default: every 3rd frame). A version counter `_seg_mask_version` increments each time a fresh mask arrives so the BEV knows when to recompute the homography.

---

## Data-Driven Homography BEV

This is the core of the Camera BEV output. Instead of analytically deriving a homography from assumed ground height and camera pitch, we let the floor pixels and their measured depths define the ground plane directly.

### Step 1 — Floor Pixel Depth Correspondences

For every pixel `(u, v)` in the FastSAM floor mask with valid depth `z`:

```
X = -(u - cx) * z / fx       [camera frame, mm]
Z =  z                        [camera frame, mm]
```

These are real 3D positions on the ground plane — no assumed height, no assumed pitch.

### Step 2 — BEV Canvas Mapping

The BEV canvas is a square image of `CAM_BEV_SIZE × CAM_BEV_SIZE` pixels with:

- **Bottom centre** = robot position (Z = 0)
- **Top** = `CAM_BEV_RANGE_M` metres forward
- **Centre column** = straight ahead (X = 0)

The scale factor `S` (mm per BEV pixel):

```
S = CAM_BEV_RANGE_M * 1000 / CAM_BEV_SIZE
```

The lateral half-range in mm:

```
R_mm = (CAM_BEV_SIZE / 2) * S
```

For each floor point `(X, Z)` in camera frame mm, the destination BEV pixel is:

```
col = (R_mm - X) / S        [X left → smaller col → left side of image]
row = (N - 1) - Z / S       [Z=0 → bottom row, Z=max → row 0]
```

where `N = CAM_BEV_SIZE`.

### Step 3 — RANSAC Homography

With typically thousands of floor pixel correspondences `(u, v) → (col, row)`, we fit a perspective homography `H` (3×3) using RANSAC:

```python
H, mask = cv2.findHomography(img_pts, bev_pts, cv2.RANSAC, reprojThresh=3.0)
```

RANSAC rejects outliers (depth noise, mask leakage onto walls or kerbs) and fits only the inlier floor pixels. The result is a robust estimate of the true ground-plane homography.

**Why this is better than an analytical H:**

| Analytical H | Data-driven H |
|---|---|
| Requires known camera height | No height assumption |
| Requires known camera pitch | Handles any pitch automatically |
| Single assumed ground plane | Fits the actual depth measurements |
| Fails if camera is tilted | Robust to tilt through correspondences |

### Step 4 — Full RGB Warp

The homography is applied to the **entire undistorted RGB frame** — no masking:

```python
result = cv2.warpPerspective(rgb, H, (N, N), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(30,30,30))
```

Objects above the ground plane (people, cars, buildings) appear at their correct ground-contact position in the BEV because the homography maps their base pixels correctly. The warped appearance of tall objects will be stretched (as in any true IPM) but their lateral and forward position is accurate.

### Step 5 — H Caching

`findHomography` is expensive. The result is cached and only recomputed when `_seg_mask_version` increments (i.e. a new FastSAM result arrives):

```python
if _cam_bev_H is None or cur_version != _cam_bev_H_version:
    # recompute H
    _cam_bev_H_version = cur_version
```

Between mask updates, the previously computed `H` is reused — typically valid for many frames since camera pose and ground plane change slowly.

---

## Object Detection — YOLOv8

YOLOv8n runs in a background thread every `DET_EVERY_N` frames. For each detection:

### Depth Sampling

The `"bottom"` mode samples the bottom 20% of the bounding box (where the object meets the floor), using the central 60% of the box width:

```python
roi = depth_mm[y2 - 0.2*(y2-y1) : y2,  cx - 0.3*bw : cx + 0.3*bw]
z_mm = median(roi[roi > DEPTH_MIN and roi < DEPTH_MAX])
```

The median is used (not mean) to reject background bleed-through at object edges.

### 3D Position

```
z_m  = z_mm / 1000
x_m  = -(u_centre - cx) * z_m / fx     [lateral, positive = left]
w_m  = bbox_width_px * z_m / fx        [real-world width]
```

### Schematic BEV Mapping

```
px = BEV_W/2  -  x_m * PPM_X     [centre ± lateral offset]
py = BEV_H-60 -  z_m * PPM_Z     [bottom = ego, top = far]
```

---

## IMU Integration

The OAK-D's BNO085 IMU provides a fused ROTATION_VECTOR at 100 Hz (accelerometer + gyroscope + magnetometer). The quaternion `(w, x, y, z)` is converted to a 3×3 rotation matrix:

```
R = [[1-2(y²+z²),  2(xy-wz),  2(xz+wy)],
     [  2(xy+wz), 1-2(x²+z²), 2(yz-wx)],
     [  2(xz-wy),  2(yz+wx),  1-2(x²+y²)]]
```

`_imu_R` is stored globally and updated every frame. It is **not** currently applied to the BEV correspondences — the data-driven homography absorbs camera tilt implicitly. `_imu_R` is reserved for:

- Detecting excessive tilt (invalidate BEV if camera pitched > threshold)
- Future odometry / pose propagation between FastSAM updates

---

## Road Boundary Extraction

From the floor mask, left and right boundary lines are extracted by scanning rows bottom-to-top:

```python
for v in range(bottom, horizon, -10):
    cols = where(mask[v, :])
    u_left  = cols[0]   # leftmost floor pixel
    u_right = cols[-1]  # rightmost floor pixel
    z_m = depth_at(v, u) / 1000
    x_m = -(u - cx) * z_m / fx
```

These `(x_m, z_m)` boundary points are drawn as polylines in the schematic BEV.

---

## Configuration Reference

```python
# Depth
DEPTH_MIN_MM  = 300      # ignore pixels closer than 30 cm
DEPTH_MAX_MM  = 50000    # 50 m outdoor range
DEPTH_EMA_A   = 0.3      # temporal smoothing factor

# Camera BEV
CAM_BEV_SIZE    = 700    # output image size (square, pixels)
CAM_BEV_RANGE_M = 15.0   # forward range shown (metres)

# Schematic BEV
BEV_W, BEV_H       = 400, 700
BEV_RANGE_FWD_M    = 15.0
BEV_RANGE_SIDE_M   = 8.0

# Detection
YOLO_CONF     = 0.50
YOLO_IOU      = 0.45
MIN_BBOX_AREA = 1500     # px² — ignore tiny detections
DET_EVERY_N   = 2        # run YOLO every N frames
SEG_EVERY_N   = 3        # run FastSAM every N frames
DEPTH_SAMPLE  = "bottom" # "bottom" | "center" | "full"
```

---

## Setup

```bash
# Activate environment
conda activate bev_nav

# Run
python stereo_pointcloud.py
```

**Keys:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Cycle depth sampling mode (bottom / center / full) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| depthai | 3.5+ | OAK-D pipeline, stereo depth, IMU |
| opencv-python | 4.13+ | Undistortion, homography, warp |
| ultralytics | 8.4+ | YOLOv8 detection + FastSAM segmentation |
| numpy | 2.0+ | Array math |
| torch (CUDA) | 2.11+ | FastSAM / YOLO GPU inference |
