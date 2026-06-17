import cv2
import sys
import os
import json
import numpy as np

# ════════════════════════════════════════════════════
#  讀取 config.json
# ════════════════════════════════════════════════════
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

CFG = load_config()

_cam      = CFG["camera"]
_pre      = CFG["preprocess"]
_edge     = CFG["edge"]
_hough    = CFG["hough"]
_classify = CFG["classify"]
_pose     = CFG["pose"]

SCALE              = _pre["scale"]
BLUR_KERNEL        = (_pre["blur_kernel"], _pre["blur_kernel"])
ROI                = _cam.get("roi")
FRAME_SKIP         = _cam.get("frame_skip", 1)

CANNY_LOW          = _edge["canny_low"]
CANNY_HIGH         = _edge["canny_high"]

HOUGH_RHO          = _hough["rho"]
HOUGH_THETA        = np.pi / 180 * _hough["theta_deg"]
HOUGH_THRESHOLD    = _hough["threshold"]
MIN_LINE_LENGTH    = _hough["min_line_length"]
MAX_LINE_GAP       = _hough["max_line_gap"]

ANGLE_THRESHOLD    = _classify["angle_threshold"]
FOCAL_LENGTH_RATIO = _pose.get("focal_length_ratio", 1.2)
FOCAL_LENGTH_PX    = _pose.get("focal_length_px", None)
MIN_LINES_REQUIRED = _pose["min_lines_required"]

# 讀取相機校正檔（mtx, dist）
_calib_file = _pose.get("calibration_file", None)
CAM_MTX  = None
CAM_DIST = None
if _calib_file and os.path.exists(_calib_file):
    _calib = np.load(_calib_file)
    CAM_MTX  = _calib["mtx"]
    CAM_DIST = _calib["dist"]
    print(f"[INFO] 已載入相機校正檔：{_calib_file}")
else:
    if _calib_file:
        print(f"[WARNING] 找不到校正檔：{_calib_file}，跳過畸變校正")

RANSAC_ITERATIONS  = 200
RANSAC_THRESHOLD   = 5.0


# ════════════════════════════════════════════════════
#  影像前處理
# ════════════════════════════════════════════════════
def preprocess(frame):
    """ROI 遮罩 → 縮放 → 灰階化 → 高斯模糊"""
    img     = apply_roi(frame)
    img     = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    return img, blurred


def apply_roi(frame):
    if ROI is None:
        return frame.copy()
    x, y, w, h = ROI
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return cv2.bitwise_and(frame, frame, mask=mask)


# ════════════════════════════════════════════════════
#  邊緣與直線偵測
# ════════════════════════════════════════════════════
def detect_lines(blurred):
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO,
        theta=HOUGH_THETA,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP,
    )
    return lines


def classify_lines(lines, img_shape=None, yaw_compensation=0.0):
    """
    依傾斜角分類直線。
    yaw_compensation: 畫面傾斜量（由 estimate_gravity_direction 先算出），
                      用來補正分類窗口，避免 Yaw 改變時垂直線被誤分類，
                      導致 Pitch 估計跟著亂跳。
    橫式照片（寬 > 高）時，水平線與垂直線的角色對調。
    """
    horizontal, vertical, diagonal = [], [], []
    if lines is None:
        return horizontal, vertical, diagonal

    # 判斷是否為橫式照片
    is_landscape = False
    if img_shape is not None:
        h, w = img_shape[:2]
        is_landscape = (w > h)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # 補正：扣掉畫面傾斜量後再分類，讓水平/垂直的判斷跟著畫面旋轉
        corrected = angle - yaw_compensation
        corrected = (corrected + 90) % 180 - 90   # 正規化到 -90 ~ 90

        if abs(corrected) < ANGLE_THRESHOLD:
            if is_landscape:
                vertical.append((x1, y1, x2, y2))
            else:
                horizontal.append((x1, y1, x2, y2))
        elif abs(abs(corrected) - 90) < ANGLE_THRESHOLD:
            if is_landscape:
                horizontal.append((x1, y1, x2, y2))
            else:
                vertical.append((x1, y1, x2, y2))
        else:
            diagonal.append((x1, y1, x2, y2))
    return horizontal, vertical, diagonal


# ════════════════════════════════════════════════════
#  重力方向估計（Roll 核心）
# ════════════════════════════════════════════════════
def estimate_gravity_direction(lines):
    """
    從所有線段的方向分布中，找出最主要的兩個方向群：
    - 最多票的方向群 → 對應場景中的「垂直結構」方向
    - 由此推算重力向量在影像中的角度 → 即 Roll

    做法：
    1. 把每條線段的角度摺疊到 [0°, 180°) 避免 ±180° 重複
    2. 用角度直方圖投票，找出最大峰值
    3. 峰值方向即為場景「垂直線」在影像中的方向
    4. Roll = 垂直方向與影像垂直軸（90°）的差值
    """
    if lines is None or len(lines) == 0:
        return 0.0, None

    angles = []
    weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angles.append(angle)
        weights.append(length)   # 較長的線給較高的權重

    angles  = np.array(angles)
    weights = np.array(weights)

    # 角度直方圖（1° 一格，共 180 格）
    hist = np.zeros(180)
    for a, w in zip(angles, weights):
        idx = int(a) % 180
        hist[idx] += w

    # 高斯平滑避免單格噪點影響峰值
    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=3)

    # 找最大峰值（最主要的線條方向）
    dominant_angle = np.argmax(hist_smooth)

    # 判斷這個主要方向是垂直結構還是水平結構
    # 若角度接近 90°（垂直）→ 直接用
    # 若角度接近 0° 或 180°（水平）→ 加 90° 轉為垂直方向
    if abs(dominant_angle - 90) <= 45:
        vertical_dir = dominant_angle      # 已經是垂直方向
    else:
        vertical_dir = (dominant_angle + 90) % 180   # 水平轉垂直

    # Roll = 垂直方向偏離影像垂直軸（90°）的角度
    roll = float(vertical_dir - 90)

    # 限制在 [-90°, 90°] 範圍內
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

def estimate_gravity_direction(lines):
    """
    從所有線段方向分布找出主要方向群。
    回傳：
      roll       - 由第一峰值估計的 Roll 角
      hist_smooth - 平滑後的角度直方圖
      second_peak - 第二峰值角度（用於估計 Yaw）
    """
    if lines is None or len(lines) == 0:
        return 0.0, None, None

    from scipy.ndimage import gaussian_filter1d

    angles, weights = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angles.append(angle)
        weights.append(length)

    hist = np.zeros(180)
    for a, w in zip(angles, weights):
        hist[int(a) % 180] += w

    hist_smooth    = gaussian_filter1d(hist, sigma=3)
    dominant_angle = np.argmax(hist_smooth)

    if abs(dominant_angle - 90) <= 45:
        vertical_dir = dominant_angle
    else:
        vertical_dir = (dominant_angle + 90) % 180

    roll = float(vertical_dir - 90)
    if roll > 90:
        roll -= 180
    elif roll < -90:
        roll += 180

    return roll, hist_smooth, None


# ════════════════════════════════════════════════════
#  RANSAC 消失點求解（Yaw / Pitch）
# ════════════════════════════════════════════════════
def line_to_homogeneous(x1, y1, x2, y2):
    p1 = np.array([x1, y1, 1.0])
    p2 = np.array([x2, y2, 1.0])
    return np.cross(p1, p2)


def point_to_line_distance(px, py, x1, y1, x2, y2):
    l = line_to_homogeneous(x1, y1, x2, y2)
    a, b, c = l
    denom = np.sqrt(a**2 + b**2)
    if denom < 1e-8:
        return float('inf')
    return abs(a * px + b * py + c) / denom


def find_vanishing_point_ransac_with_inliers(lines):
    """RANSAC 消失點求解，同時回傳 inlier 索引集合。"""
    if len(lines) < 2:
        return None, set()
    best_vp, best_inliers = None, []
    for _ in range(RANSAC_ITERATIONS):
        idx = np.random.choice(len(lines), 2, replace=False)
        l1  = line_to_homogeneous(*lines[idx[0]])
        l2  = line_to_homogeneous(*lines[idx[1]])
        pt  = np.cross(l1, l2)
        if abs(pt[2]) < 1e-8:
            continue
        vp = pt[:2] / pt[2]
        inliers = [
            i for i, (x1,y1,x2,y2) in enumerate(lines)
            if point_to_line_distance(vp[0], vp[1], x1, y1, x2, y2) < RANSAC_THRESHOLD
        ]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp      = vp
    if best_vp is None or len(best_inliers) < 2:
        return None, set()
    inlier_lines = [lines[i] for i in best_inliers]
    A, b_vec = [], []
    for x1, y1, x2, y2 in inlier_lines:
        l = line_to_homogeneous(x1, y1, x2, y2)
        A.append(l[:2])
        b_vec.append(-l[2])
    vp_refined, _, _, _ = np.linalg.lstsq(
        np.array(A), np.array(b_vec), rcond=None)
    return vp_refined, set(best_inliers)


def find_vanishing_point_ransac(lines):
    if len(lines) < 2:
        return None
    best_vp      = None
    best_inliers = []
    for _ in range(RANSAC_ITERATIONS):
        idx = np.random.choice(len(lines), 2, replace=False)
        l1  = line_to_homogeneous(*lines[idx[0]])
        l2  = line_to_homogeneous(*lines[idx[1]])
        pt  = np.cross(l1, l2)
        if abs(pt[2]) < 1e-8:
            continue
        vp = pt[:2] / pt[2]
        inliers = [
            i for i, (x1,y1,x2,y2) in enumerate(lines)
            if point_to_line_distance(vp[0], vp[1], x1, y1, x2, y2) < RANSAC_THRESHOLD
        ]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_vp      = vp
    if best_vp is None or len(best_inliers) < 2:
        return None
    inlier_lines = [lines[i] for i in best_inliers]
    A, b_vec = [], []
    for x1, y1, x2, y2 in inlier_lines:
        l = line_to_homogeneous(x1, y1, x2, y2)
        A.append(l[:2])
        b_vec.append(-l[2])
    vp_refined, _, _, _ = np.linalg.lstsq(
        np.array(A), np.array(b_vec), rcond=None)
    return vp_refined


# ════════════════════════════════════════════════════
#  姿態估計
# ════════════════════════════════════════════════════
def estimate_pose(lines, h_lines, v_lines, img_shape):
    h, w         = img_shape[:2]
    cx, cy       = w / 2, h / 2
    if FOCAL_LENGTH_PX is not None:
        focal_length = FOCAL_LENGTH_PX
    else:
        focal_length = w * FOCAL_LENGTH_RATIO

    if lines is not None and len(lines) < MIN_LINES_REQUIRED:
        print(f"[WARNING] 直線數量不足（{len(lines)} 條），結果可能不準確")

    # ── Roll：重力方向估計（直方圖第一峰值）────────────
    roll, hist_smooth, second_peak = estimate_gravity_direction(lines)

    # ── Pitch：垂直消失點 y 偏移 ──────────────────────
    pitch = 0.0
    vp_v  = find_vanishing_point_ransac(v_lines) if v_lines else None
    if vp_v is not None:
        pitch = float(np.degrees(np.arctan2(vp_v[1] - cy, focal_length)))

    # ── Yaw：方向一 放寬門檻（35°）後的水平線消失點（同樣補正傾斜量）────
    yaw = 0.0
    h_lines_wide = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            corrected = angle - roll
            corrected = (corrected + 90) % 180 - 90
            if abs(corrected) < 35:
                h_lines_wide.append((x1, y1, x2, y2))
    vp_h = find_vanishing_point_ransac(h_lines_wide) if h_lines_wide else None
    if vp_h is not None:
        yaw = float(np.degrees(np.arctan2(vp_h[0] - cx, focal_length)))

    return {
        "yaw":   round(yaw,   2),
        "pitch": round(pitch, 2),
        "roll":  round(roll,  2),
    }


# ════════════════════════════════════════════════════
#  視覺化
# ════════════════════════════════════════════════════
def visualize(img, lines, pose):
    result = img.copy()
    h, w   = result.shape[:2]
    cx, cy = w // 2, h // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.line(result, (cx-30, cy), (cx+30, cy), (0, 0, 255), 2)
    cv2.line(result, (cx, cy-30), (cx, cy+30), (0, 0, 255), 2)

    text = f"Roll: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Yaw: {pose['roll']:.1f}"
    cv2.putText(result, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return result


# ════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════
def process_frame(frame, verbose=False):
    img, blurred = preprocess(frame)
    lines        = detect_lines(blurred)

    # 第一階段：先用直方圖估計畫面傾斜量（對應顯示的 Yaw）
    yaw_tilt, _, _ = estimate_gravity_direction(lines)

    # 第二階段：用傾斜量補正分類窗口，避免 Pitch 被 Yaw 干擾
    h_lines, v_lines, _ = classify_lines(lines, img.shape, yaw_compensation=yaw_tilt)

    pose = estimate_pose(lines, h_lines, v_lines, img.shape)

    if verbose:
        print(f"Yaw:   {pose['yaw']}°")
        print(f"Pitch: {pose['pitch']}°")
        print(f"Roll:  {pose['roll']}°")
        n = len(lines) if lines is not None else 0
        print(f"總線條: {n} 條  水平: {len(h_lines)}  垂直: {len(v_lines)}")

    return visualize(img, lines, pose)


def main(source):
    if source.isdigit():
        cap      = cv2.VideoCapture(int(source))
        is_video = True
    else:
        ext      = source.lower().split(".")[-1]
        is_video = ext in ["mp4", "avi", "mov", "mkv"]
        cap      = cv2.VideoCapture(source) if is_video else None

    if is_video:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            cv2.imshow("Pose Estimation", process_frame(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] 無法讀取圖片：{source}")
            return
        result = process_frame(frame, verbose=True)
        input_name  = os.path.splitext(os.path.basename(source))[0]
        output_path = f"result_{input_name}.jpg"
        cv2.imwrite(output_path, result)
        print(f"輸出：{output_path}")
        cv2.imshow("Pose Estimation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式：python main.py <圖片路徑 或 影片路徑 或 相機編號>")
        print("範例：")
        print("  python main.py photo.jpg")
        print("  python main.py video.mp4")
        print("  python main.py 0")
        sys.exit(1)
    main(sys.argv[1])