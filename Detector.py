"""
╔══════════════════════════════════════════════════════════╗
║  AURA — Module 1: YOLOv8 Real-Time Traffic Detector     ║
║  Dataset: Highway Traffic Videos / Kaggle / Roboflow    ║
╚══════════════════════════════════════════════════════════╝

SUPPORTED REAL DATASETS (drop any of these in and it works):
─────────────────────────────────────────────────────────────
1. Kaggle — Highway Traffic Videos Dataset
   https://www.kaggle.com/datasets/aryashah2k/highway-traffic-videos-dataset
   → Download → extract → set VIDEO_PATH = "path/to/video.mp4"

2. Kaggle — Traffic Video Dataset (Arshad Rahimanziban)
   https://www.kaggle.com/datasets/arshadrahmanziban/traffic-video-dataset
   → Multiple MP4 clips, set VIDEO_PATH = "path/to/clip.mp4"

3. Roboflow Universe — Traffic Road Object Detection
   https://universe.roboflow.com/yolo-iuil0/traffic-4zbse
   → Download YOLO format, pass image folder via IMAGE_DIR

4. TRANCOS Dataset (vehicle counting in congestion)
   http://agamenon.tsc.uah.es/Investigacion/gram/publications/eccv2016-onoro/

5. Live CCTV / Webcam  →  set VIDEO_PATH = 0

6. Generated synthetic frames (included in this project)
   → data/sample_frames/  (run utils/generate_dataset.py first)

HOW TO USE:
   pip install ultralytics opencv-python
   python module1_detection/detector.py --source data/sample_frames
   python module1_detection/detector.py --source /path/to/video.mp4
   python module1_detection/detector.py --source 0          # webcam
"""

import argparse
import csv
import json
import time
import datetime
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── YOLOv8 import (graceful fallback) ──────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Constants ───────────────────────────────────────────────────────────────
COCO_VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Indian traffic — add autorickshaw/tuk-tuk if using custom-trained model
CUSTOM_CLASSES = {
    0: "car", 1: "truck", 2: "bus",
    3: "motorcycle", 4: "autorickshaw"
}

CONF_THRESHOLD   = 0.40
IOU_THRESHOLD    = 0.50
MODEL_WEIGHTS    = "yolov8n.pt"    # nano=fastest; yolov8m.pt for better accuracy
OUTPUT_CSV       = "outputs/tdi_results.csv"
OUTPUT_VIDEO     = "outputs/annotated_output.mp4"

# TDI = vehicles per normalised frame (relative to 1280×720 baseline)
def compute_tdi(n_vehicles: int, frame_w: int, frame_h: int) -> float:
    norm = (frame_w * frame_h) / (1280 * 720)
    return round(n_vehicles / max(norm, 0.01), 2)

# AQI category lookup
def aqi_category(pm25):
    if pm25 <= 50:   return "Good",        (0, 230, 118)
    if pm25 <= 100:  return "Satisfactory",(168, 230, 99)
    if pm25 <= 200:  return "Moderate",    (255, 170,  0)
    if pm25 <= 300:  return "Poor",        (255, 107, 53)
    if pm25 <= 400:  return "Very Poor",   (255,  68, 68)
    return               "Severe",         (139,   0,  0)

# Simple TDI→PM2.5 correlation (baseline before RF model kicks in)
def quick_pm25_estimate(tdi, hour):
    base = 160 if (7<=hour<=10 or 17<=hour<=20) else 110
    return min(400, base + tdi * 1.1)


# ══════════════════════════════════════════════════════════════════════════
# DRAWING UTILITIES
# ══════════════════════════════════════════════════════════════════════════
COLOURS = {
    "car":         (0, 229, 255),
    "motorcycle":  (127, 255, 110),
    "bus":         (255, 107,  53),
    "truck":       (255,  68,  68),
    "autorickshaw":(255, 200,   0),
}

def draw_box(frame, x1, y1, x2, y2, label, conf, vtype):
    c = COLOURS.get(vtype, (200, 200, 200))
    cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
    tag = f"{label} {conf:.2f}"
    tw, th = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), c, -1)
    cv2.putText(frame, tag, (x1+2, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

def draw_hud(frame, tdi, counts, pm25, frame_idx, fps):
    H, W = frame.shape[:2]
    cat, col = aqi_category(pm25)
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # Semi-transparent HUD panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (300, 160), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        (f"AURA — Traffic Monitor",       (0,229,255), 0.55, 2),
        (f"Frame: {frame_idx:05d}  {ts}", (180,180,180), 0.45, 1),
        (f"TDI : {tdi:.1f} veh/frame",   (0,229,255), 0.50, 1),
        (f"Cars : {counts.get('car',0)}  Bikes: {counts.get('motorcycle',0)}", (200,200,200), 0.45, 1),
        (f"Bus  : {counts.get('bus',0)}  Trucks: {counts.get('truck',0)}",    (200,200,200), 0.45, 1),
        (f"PM2.5≈{pm25:.0f} µg/m³ [{cat}]", col, 0.50, 1),
        (f"FPS  : {fps:.1f}",             (150,150,150), 0.40, 1),
    ]
    for i, (txt, color, scale, thick) in enumerate(lines):
        cv2.putText(frame, txt, (8, 20 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    # AQI bar (bottom)
    bar_w, bar_h = W - 40, 12
    bx, by = 20, H - 30
    cv2.rectangle(frame, (bx, by), (bx+bar_w, by+bar_h), (40,40,40), -1)
    grad = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
    for px in range(bar_w):
        r = int(255 * px / bar_w)
        g = int(255 * (1 - px/bar_w))
        grad[:, px] = (0, g, r)   # BGR: green→red
    frame[by:by+bar_h, bx:bx+bar_w] = grad

    needle_x = bx + int(min(pm25, 400) / 400 * bar_w)
    cv2.line(frame, (needle_x, by-5), (needle_x, by+bar_h+5), (255,255,255), 2)
    cv2.putText(frame, "AQI", (bx, by-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
    cv2.putText(frame, "0", (bx, by+bar_h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100,220,100), 1)
    cv2.putText(frame, "400", (bx+bar_w-22, by+bar_h+14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,220), 1)


# ══════════════════════════════════════════════════════════════════════════
# REAL YOLOV8 DETECTION (needs ultralytics installed)
# ══════════════════════════════════════════════════════════════════════════
def run_yolo_on_video(source, max_frames=None, display=True, save_video=False):
    """
    Full YOLOv8 pipeline on real video file or webcam.
    source: str path to .mp4 / int for webcam
    """
    model = YOLO(MODEL_WEIGHTS)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return []

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if save_video:
        Path("outputs").mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, orig_fps, (W,H))

    Path("outputs").mkdir(exist_ok=True)
    csv_file = open(OUTPUT_CSV, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "frame","timestamp","tdi","total_vehicles",
        "cars","motorcycles","buses","trucks","pm25_estimate"
    ])
    csv_writer.writeheader()

    records, frame_idx = [], 0
    t_prev = time.time()
    print(f"\n{'='*55}")
    print(f"  AURA Module 1 — YOLOv8 Detection")
    print(f"  Source : {source}  |  {W}×{H}  {orig_fps:.0f}fps")
    print(f"  Model  : {MODEL_WEIGHTS}")
    print(f"  Press Q to quit")
    print(f"{'='*55}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        results = model(frame,
                        classes=list(COCO_VEHICLE_CLASSES.keys()),
                        conf=CONF_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        verbose=False)

        counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_VEHICLE_CLASSES:
                continue
            vtype = COCO_VEHICLE_CLASSES[cls_id]
            counts[vtype] = counts.get(vtype, 0) + 1
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            draw_box(frame, x1,y1,x2,y2, vtype, float(box.conf[0]), vtype)

        total = sum(counts.values())
        tdi = compute_tdi(total, W, H)
        hour = datetime.datetime.now().hour
        pm25 = quick_pm25_estimate(tdi, hour)

        fps = 1.0 / max(time.time()-t_prev, 1e-6)
        t_prev = time.time()
        draw_hud(frame, tdi, counts, pm25, frame_idx, fps)

        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = {"frame": frame_idx, "timestamp": ts, "tdi": tdi,
               "total_vehicles": total,
               "cars": counts.get("car",0),
               "motorcycles": counts.get("motorcycle",0),
               "buses": counts.get("bus",0),
               "trucks": counts.get("truck",0),
               "pm25_estimate": round(pm25,1)}
        csv_writer.writerow(row)
        records.append(row)

        if writer:
            writer.write(frame)
        if display:
            cv2.imshow("AURA — Traffic Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx:05d} | TDI={tdi:.1f} | Vehicles={total} | PM2.5≈{pm25:.0f}")
        frame_idx += 1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\n✅ Processed {frame_idx} frames  →  {OUTPUT_CSV}")
    return records


# ══════════════════════════════════════════════════════════════════════════
# FRAME-FOLDER MODE (works without a video file — uses our generated PNGs)
# ══════════════════════════════════════════════════════════════════════════
def run_on_frame_folder(folder, annotation_mode="json"):
    """
    Process pre-extracted frames (JPG/PNG) from:
    - Our synthetic generator (data/sample_frames/)
    - Roboflow downloads (images/)
    - Extracted Kaggle video frames

    annotation_mode:
      "json"  — reads our generated JSON annotations (no YOLO needed)
      "yolo"  — runs real YOLOv8 on each frame image
    """
    folder = Path(folder)
    frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not frames:
        print(f"[ERROR] No images found in {folder}")
        return []

    Path("outputs").mkdir(exist_ok=True)
    csv_path = "outputs/tdi_results.csv"
    records = []

    print(f"\n{'='*55}")
    print(f"  AURA Module 1 — Frame Folder Processing")
    print(f"  Folder : {folder}")
    print(f"  Frames : {len(frames)}")
    print(f"  Mode   : {annotation_mode.upper()}")
    print(f"{'='*55}\n")

    model = None
    if annotation_mode == "yolo":
        if not YOLO_AVAILABLE:
            print("[WARN] ultralytics not installed → falling back to JSON mode")
            annotation_mode = "json"
        else:
            model = YOLO(MODEL_WEIGHTS)

    for i, fpath in enumerate(frames):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        H, W = frame.shape[:2]

        # ── JSON annotation mode (pre-computed or our synthetic) ──
        if annotation_mode == "json":
            ann_path = fpath.with_suffix(".json")
            if ann_path.exists():
                with open(ann_path) as f:
                    data = json.load(f)
                annots = data.get("annotations", [])
                hour = data.get("hour", datetime.datetime.now().hour)
                counts = {}
                for a in annots:
                    counts[a["class"]] = counts.get(a["class"], 0) + 1
                    x,y,w,h = a["bbox"]
                    vtype = a["class"]
                    cv2.rectangle(frame,(x,y),(x+w,y+h), COLOURS.get(vtype,(200,200,200)), 2)
                    cv2.putText(frame, vtype, (x,y-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOURS.get(vtype,(200,200,200)), 1)
            else:
                counts = {}; hour = 8

        # ── Real YOLOv8 mode ──
        else:
            results = model(frame, classes=list(COCO_VEHICLE_CLASSES.keys()),
                            conf=CONF_THRESHOLD, verbose=False)
            counts = {}
            hour = datetime.datetime.now().hour
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                vtype = COCO_VEHICLE_CLASSES.get(cls_id, "unknown")
                counts[vtype] = counts.get(vtype, 0) + 1
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                draw_box(frame, x1,y1,x2,y2, vtype, float(box.conf[0]), vtype)

        total = sum(counts.values())
        tdi   = compute_tdi(total, W, H)
        pm25  = quick_pm25_estimate(tdi, hour)

        draw_hud(frame, tdi, counts, pm25, i, 0)
        out_path = f"outputs/annotated_{fpath.name}"
        cv2.imwrite(out_path, frame)

        ts = datetime.datetime.now().isoformat(timespec="seconds")
        row = {
            "frame": i, "source_file": fpath.name, "hour": hour,
            "tdi": tdi, "total_vehicles": total,
            "cars": counts.get("car",0), "motorcycles": counts.get("motorcycle",0),
            "buses": counts.get("bus",0), "trucks": counts.get("truck",0),
            "autorickshaws": counts.get("autorickshaw",0),
            "pm25_estimate": round(pm25,1),
            "timestamp": ts
        }
        records.append(row)

        if i % 50 == 0:
            print(f"  [{i:3d}/{len(frames)}] {fpath.name} | "
                  f"TDI={tdi:.1f} | Vehicles={total} | PM2.5≈{pm25:.0f}")

    # Save CSV
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader(); w.writerows(records)

    print(f"\n✅ Processed {len(records)} frames  →  {csv_path}")
    print(f"   Annotated images saved to: outputs/")
    return records


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA Traffic Detector")
    parser.add_argument("--source", default="videos",
        help="Video file (.mp4), webcam (0), or folder of frames")
    parser.add_argument("--mode", default="json",
        choices=["json","yolo"],
        help="json=use annotation files, yolo=run YOLOv8 (needs ultralytics)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    src = args.source
    # Numeric string → webcam index
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    if isinstance(src, int) or (isinstance(src, str) and Path(src).suffix.lower() in [".mp4",".avi",".mkv",".mov"]):
        # VIDEO or WEBCAM mode
        if not YOLO_AVAILABLE:
            print("[ERROR] Video mode requires: pip install ultralytics")
            sys.exit(1)
        records = run_yolo_on_video(src, args.max_frames,
                                    display=not args.no_display,
                                    save_video=args.save_video)
    else:
        # FRAME FOLDER mode
        records = run_on_frame_folder(src, annotation_mode=args.mode)

    if records:
        tdis = [r["tdi"] for r in records]
        pm25s = [r["pm25_estimate"] for r in records]
        print(f"\n{'─'*45}")
        print(f"  Summary:")
        print(f"  Frames processed : {len(records)}")
        print(f"  Avg TDI          : {sum(tdis)/len(tdis):.2f}")
        print(f"  Peak TDI         : {max(tdis):.2f}")
        print(f"  Avg PM2.5 est.   : {sum(pm25s)/len(pm25s):.1f} µg/m³")
        print(f"  Peak PM2.5 est.  : {max(pm25s):.1f} µg/m³")
        print(f"{'─'*45}\n")