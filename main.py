import cv2
import sys
import json
import numpy as np

# ════════════════════════════════════════════════════
#  讀取 config.json
# ════════════════════════════════════════════════════
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

CFG = load_config()

# 展開各區塊參數
_cam      = CFG["camera"]
_pre      = CFG["preprocess"]
_edge     = CFG["edge"]
_hough    = CFG["hough"]
_classify = CFG["classify"]
_pose     = CFG["pose"]

SCALE              = _pre["scale"]
BLUR_KERNEL        = (_pre["blur_kernel"], _pre["blur_kernel"])
ROI                = _cam.get("roi")        # [x, y, w, h]，None 表示全畫面
FRAME_SKIP         = _cam.get("frame_skip", 1)

CANNY_LOW          = _edge["canny_low"]
CANNY_HIGH         = _edge["canny_high"]

HOUGH_RHO          = _hough["rho"]
HOUGH_THETA        = np.pi / 180 * _hough["theta_deg"]
HOUGH_THRESHOLD    = _hough["threshold"]
MIN_LINE_LENGTH    = _hough["min_line_length"]
MAX_LINE_GAP       = _hough["max_line_gap"]

ANGLE_THRESHOLD    = _classify["angle_threshold"]

FOCAL_LENGTH_RATIO = _pose["focal_length_ratio"]
MIN_LINES_REQUIRED = _pose["min_lines_required"]


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
    """套用 ROI 遮罩，ROI 為 None 時直接回傳原圖。"""
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
    """Canny 邊緣偵測 → HoughLinesP 直線偵測"""
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


def classify_lines(lines):
    """將直線依傾斜角分成水平線、垂直線、斜線三類。"""
    horizontal, vertical, diagonal = [], [], []
    if lines is None:
        return horizontal, vertical, diagonal
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < ANGLE_THRESHOLD:
            horizontal.append((x1, y1, x2, y2))
        elif abs(abs(angle) - 90) < ANGLE_THRESHOLD:
            vertical.append((x1, y1, x2, y2))
        else:
            diagonal.append((x1, y1, x2, y2))
    return horizontal, vertical, diagonal


# ════════════════════════════════════════════════════
#  姿態估計
# ════════════════════════════════════════════════════
def find_vanishing_point(lines):
    """最小二乘法求消失點（齊次座標 + 超定方程組）。"""
    if len(lines) < 2:
        return None
    A, b_vec = [], []
    for x1, y1, x2, y2 in lines:
        p1 = np.array([x1, y1, 1.0])
        p2 = np.array([x2, y2, 1.0])
        l  = np.cross(p1, p2)
        A.append(l[:2])
        b_vec.append(-l[2])
    vp, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b_vec), rcond=None)
    return vp


def estimate_pose(h_lines, v_lines, img_shape):
    """由水平線群和垂直線群估計 roll / pitch / yaw。"""
    h, w         = img_shape[:2]
    cx, cy       = w / 2, h / 2
    focal_length = w * FOCAL_LENGTH_RATIO

    if len(h_lines) + len(v_lines) < MIN_LINES_REQUIRED:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    # Roll：水平線群的中位數傾斜角
    roll = 0.0
    if h_lines:
        angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for x1,y1,x2,y2 in h_lines]
        roll   = float(np.median(angles))

    # Pitch：垂直消失點的 y 偏移
    pitch = 0.0
    vp_v  = find_vanishing_point(v_lines)
    if vp_v is not None:
        pitch = float(np.degrees(np.arctan2(vp_v[1] - cy, focal_length)))

    # Yaw：水平消失點的 x 偏移
    yaw  = 0.0
    vp_h = find_vanishing_point(h_lines)
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
    """將偵測直線、正十字準星、角度數值疊加到影像上。"""
    result = img.copy()
    h, w   = result.shape[:2]
    cx, cy = w // 2, h // 2

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.line(result, (cx-30, cy), (cx+30, cy), (0, 0, 255), 2)
    cv2.line(result, (cx, cy-30), (cx, cy+30), (0, 0, 255), 2)

    text = f"Yaw: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Roll: {pose['roll']:.1f}"
    cv2.putText(result, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return result


# ════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════
def process_frame(frame, verbose=False):
    img, blurred        = preprocess(frame)
    lines               = detect_lines(blurred)
    h_lines, v_lines, _ = classify_lines(lines)
    pose                = estimate_pose(h_lines, v_lines, img.shape)

    if verbose:
        print(f"Yaw:   {pose['yaw']}°")
        print(f"Pitch: {pose['pitch']}°")
        print(f"Roll:  {pose['roll']}°")

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
        cv2.imwrite("result.jpg", result)
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
