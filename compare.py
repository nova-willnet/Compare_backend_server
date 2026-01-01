#!/usr/bin/env python3
# compare.py
# Updated: removed AI / Gemini summary logic. Only image differencing (SSIM + ORB) and heatmap generation remain.
#
# This file is based on the original compare.py you uploaded; AI summarization functions were removed.
# See original for reference. :contentReference[oaicite:4]{index=4}

import sys
import json
import os
import traceback
import base64
import uuid
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# optional face cascade filename (bundled with OpenCV or system). If not found we skip face detection.
DEFAULT_FACE_CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
]

def load_and_downscale(path, max_dim=1200):
    # Read file robustly (handles Windows path etc.)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def orb_align(A_color, B_color, max_features=1000):
    A_gray = cv2.cvtColor(A_color, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_color, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(A_gray, None)
    kp2, des2 = orb.detectAndCompute(B_gray, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None  # alignment not possible

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = matcher.match(des1, des2)
    except Exception:
        return None

    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 8:
        return None

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    warped = cv2.warpPerspective(B_color, H, (A_color.shape[1], A_color.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

def compute_ssim_map(A_color, B_aligned):
    A_gray = cv2.cvtColor(A_color, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_aligned, cv2.COLOR_BGR2GRAY)
    if A_gray.shape != B_gray.shape:
        B_gray = cv2.resize(B_gray, (A_gray.shape[1], A_gray.shape[0]), interpolation=cv2.INTER_AREA)

    score, diff = ssim(A_gray, B_gray, full=True)
    # convert to difference map where higher means larger difference
    diff = (1.0 - diff)
    diff = diff - diff.min()
    if diff.max() > 0:
        diff = diff / diff.max()
    diff_uint8 = np.uint8(diff * 255)
    return float(score), diff_uint8

def detect_regions(diff_uint8, min_area=150, thresh_val=60):
    _, binm = cv2.threshold(diff_uint8, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regs = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area:
            continue
        roi = diff_uint8[y:y+h, x:x+w]
        mean = float(np.mean(roi))
        sev = 'green'
        if mean > 170: sev = 'red'
        elif mean > 120: sev = 'yellow'
        regs.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'mean': mean, 'sev': sev})
    return regs

def render_heatmap_to_png(diff_uint8, base_color_img, alpha=0.5):
    colored = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_JET)
    if colored.shape[:2] != base_color_img.shape[:2]:
        colored = cv2.resize(colored, (base_color_img.shape[1], base_color_img.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(base_color_img, 1.0 - alpha, colored, alpha, 0)
    is_success, buffer = cv2.imencode(".png", overlay)
    if not is_success:
        raise RuntimeError("Failed to encode heatmap png")
    b64 = base64.b64encode(buffer).decode('utf-8')
    return 'data:image/png;base64,' + b64, buffer.tobytes(), overlay  # base64, raw bytes, overlay mat

def detect_faces_if_possible(img_color):
    # Try to load Haar cascade; return number of faces and bounding boxes
    for p in DEFAULT_FACE_CASCADE_PATHS:
        if p and os.path.exists(p):
            try:
                face_cascade = cv2.CascadeClassifier(p)
                gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
                return len(faces), [{'x': int(x),'y':int(y),'w':int(w),'h':int(h)} for (x,y,w,h) in faces]
            except Exception:
                continue
    return 0, []

def save_temp_png(buffer_bytes, prefix='heatmap_'):
    fname = f"{prefix}{uuid.uuid4().hex}.png"
    tmp_path = os.path.join('./tmp' if os.path.exists('./tmp') else '.', fname)
    with open(tmp_path, 'wb') as f:
        f.write(buffer_bytes)
    return tmp_path

def main():
    try:
        # expect CLI args: beforePath afterPath (this matches how server.js spawns it)
        if len(sys.argv) < 3:
            print(json.dumps({ "success": False, "error": "Usage: compare.py <beforePath> <afterPath>" }))
            return

        beforePath = sys.argv[1]
        afterPath = sys.argv[2]

        A_color, scaleA = load_and_downscale(beforePath)
        B_color, scaleB = load_and_downscale(afterPath)

        # attempt ORB alignment; if alignment fails, we will use B as-is
        B_aligned = orb_align(A_color, B_color)
        if B_aligned is None:
            B_aligned = B_color.copy()

        score, diff_uint8 = compute_ssim_map(A_color, B_aligned)
        regions = detect_regions(diff_uint8, min_area=150, thresh_val=60)

        # optional face detection
        faces_count, faces_boxes = detect_faces_if_possible(A_color)

        # render heatmap image (overlay of base + colored diff)
        heatmap_base64, heatmap_bytes, overlay_mat = render_heatmap_to_png(diff_uint8, A_color, alpha=0.5)

        # save local heatmap PNG to temp file (server will upload to Cloudinary if configured)
        heatmap_file = save_temp_png(heatmap_bytes, prefix='compare_heatmap_')

        # Prepare compareJson for frontend
        comp_json = {
            "regions": regions,
            "counts": {
                "num_regions": len(regions),
                "faces_count": faces_count
            },
            "meta": {
                "A_shape": [int(A_color.shape[0]), int(A_color.shape[1])],
                "B_shape": [int(B_color.shape[0]), int(B_color.shape[1])],
                "scaleA": float(scaleA),
                "scaleB": float(scaleB)
            },
            "ssim_score": float(score)
        }

        result = {
            "success": True,
            "compareJson": comp_json,
            "heatmap_png_base64": heatmap_base64,
            "heatmap_file": heatmap_file,
            "score": float(score)
        }

        print(json.dumps(result))
        return

    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write("compare.py exception:\n" + tb + "\n")
        print(json.dumps({ "success": False, "error": str(e), "traceback": tb }))
        return

if __name__ == '__main__':
    main()
