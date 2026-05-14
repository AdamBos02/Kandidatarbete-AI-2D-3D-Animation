import json
import cv2
import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmdet.apis import inference_detector, init_detector
import os

# --- Konfiguration ---
VIDEO_PATH = "/Users/adambostrom/Desktop/Qualisys mätningar/Indianhopp_viktor/IMG_5478_sync.mp4"
OUTPUT_VIDEO = "vis_results/övning3_pose.mp4"
OUTPUT_JSON = "vis_results/övning3_keypoints.json"

DETECTOR_CONFIG = "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
DETECTOR_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

POSE_CONFIG = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py"
POSE_CHECKPOINT = "checkpoints/vitpose_large_coco_aic_mpii.pth"

DEVICE = "cpu"
DETECTION_THRESHOLD = 0.1

# --- Hjälpfunktion: beräkna IoU mellan två bboxar ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Hjälpfunktion: beräkna area av bbox ---
def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# --- Initiera modeller ---
print("Laddar detektorn...")
detector = init_detector(DETECTOR_CONFIG, DETECTOR_CHECKPOINT, device=DEVICE)

print("Laddar VITPose...")
pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)

# --- Öppna video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")

# --- Hitta personen med högst konfidens i första framen ---
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Kunde inte läsa videon.")

mmdet_results = inference_detector(detector, first_frame)
person_bboxes = [bbox for bbox in mmdet_results[0] if bbox[4] > DETECTION_THRESHOLD]

if len(person_bboxes) == 0:
    raise RuntimeError("Ingen person hittades i första framen!")

best_person = max(person_bboxes, key=lambda x: x[4])
tracking_bbox = best_person[:4].tolist()
print(f"Vald person i första framen med konfidens {best_person[4]:.2f}: {tracking_bbox}")

# Återställ videon till början
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Skapa output-video ---
os.makedirs("vis_results", exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# --- Kör pose estimation per frame ---
all_keypoints = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Bearbetar frame {frame_idx + 1}/{total_frames}...", end="\r")

    # Detektera personer
    mmdet_results = inference_detector(detector, frame)
    person_bboxes = [bbox for bbox in mmdet_results[0] if bbox[4] > DETECTION_THRESHOLD]

    # Försök hitta personen via IoU
    best_person = None
    best_iou = 0.0
    for bbox in person_bboxes:
        iou = compute_iou(tracking_bbox, bbox[:4].tolist())
        if iou > best_iou:
            best_iou = iou
            best_person = bbox

    # Om ingen IoU-match (t.ex. under hopp) – välj den med störst area
    if best_iou == 0.0 and len(person_bboxes) > 0:
        best_person = max(person_bboxes, key=lambda x: bbox_area(x[:4]))
        print(f"\nFrame {frame_idx}: Ingen IoU-match, väljer störst person (area={bbox_area(best_person[:4]):.0f})")

    # Uppdatera tracking-bbox och kör pose estimation
    frame_data = {"frame": frame_idx, "persons": []}
    if best_person is not None:
        tracking_bbox = best_person[:4].tolist()
        person_results = [{"bbox": best_person}]
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame,
            person_results,
            bbox_thr=DETECTION_THRESHOLD,
            format="xyxy",
            dataset=pose_model.cfg.data.test.type,
        )
        for person in pose_results:
            keypoints = person["keypoints"].tolist()
            frame_data["persons"].append({"keypoints": keypoints})

        vis_frame = vis_pose_result(pose_model, frame, pose_results)
    else:
        print(f"\nVarning: Ingen person hittades i frame {frame_idx}, använder original.")
        vis_frame = frame

    all_keypoints.append(frame_data)
    writer.write(vis_frame)
    frame_idx += 1

cap.release()
writer.release()

# --- Spara JSON ---
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_keypoints, f, indent=2)

print(f"\nKlart!")
print(f"Video sparad: {OUTPUT_VIDEO}")
print(f"Keypoints sparade: {OUTPUT_JSON}")