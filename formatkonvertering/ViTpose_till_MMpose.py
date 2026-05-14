"""
VITPose_2_MMPose.py
────────────────────────────
Konverterar VITPose JSON-output till samma namngivna format som MMPose
producerar. Detta möjliggör att samma nedströms-scripts för formatkonvertering
(coco_to_motionbert.py, coco_to_motionagformer.py) kan användas oavsett
om 2D-estimeringen gjorts med MMPose eller ViTPose.

VITPose-format (in):
  [{"frame": 0, "persons": [{"keypoints": [[x,y,score], ...17st...]}]}, ...]

MMPose-format (ut):
  [{"frame": 0, "persons": [{"nose": {"x":..,"y":..,"score":..}, ...}]}, ...]

Keypoints som saknas i COCO-17 (tår, hälar, handcentra) sätts till null
så att nedströms-scripts hanterar dem som saknade värden.
"""

import json
import numpy as np

# ========= SÖKVÄGAR =========
INPUT_JSON  = "vis_results/övning3_keypoints.json"        # Output från ViTPose
OUTPUT_JSON = "vis_results/övning3_keypoints_named.json"  # MMPose-kompatibelt format
# ============================

# ========= COCO-17 NAMNORDNING =========
# ViTPose genererar keypoints i COCO-17-ordning (index 0–16).
# Dessa mappas till läsbara namn för att matcha MMPose-formatet.
COCO17_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Keypoints som finns i MMPose-formatet men saknas i COCO-17.
# Sätts till null här och hanteras i efterföljande konverteringssteg.
MISSING_IN_COCO17 = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "head_center",
    "left_hand_center", "right_hand_center",
]

# Huvud-keypoints används för att beräkna head_center
HEAD_IDXS = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear
SCORE_THR = 0.20              # Konfidenströskel för head_center-beräkning


def convert_person(kpts_raw: list) -> dict:
    """Konverterar en persons 17 COCO-keypoints från listformat till
    MMPose-kompatibelt namngivet dict-format.

    Tre steg utförs:
    1. De 17 COCO-keypoints namnges enligt COCO17_NAMES.
    2. head_center beräknas som medelvärde av huvud-keypoints
       med tillräcklig konfidens (>SCORE_THR).
    3. Keypoints som saknas i COCO-17 sätts till null för att
       hanteras i efterföljande konverteringssteg.

    Args:
        kpts_raw: Lista av [x, y, score] med 17 element (COCO-17-ordning).
    Returns:
        Namngivet dict i MMPose-format.
    """
    person_dict = {}

    # Steg 1: Namnge de 17 COCO-keypoints
    for i, name in enumerate(COCO17_NAMES):
        if i < len(kpts_raw):
            x, y, score = kpts_raw[i]
            person_dict[name] = {"x": float(x), "y": float(y), "score": float(score)}
        else:
            person_dict[name] = None

    # Steg 2: Beräkna head_center från huvud-keypoints
    # Används som approximation för huvudets position då ViTPose
    # inte genererar en explicit head_center som MMPose gör
    head_pts = []
    head_scores = []
    for i in HEAD_IDXS:
        kp = person_dict.get(COCO17_NAMES[i])
        if kp and kp["score"] > SCORE_THR:
            head_pts.append([kp["x"], kp["y"]])
            head_scores.append(kp["score"])

    if head_pts:
        c = np.mean(head_pts, axis=0)
        person_dict["head_center"] = {
            "x": float(c[0]),
            "y": float(c[1]),
            "score": float(min(head_scores)),
        }
    else:
        person_dict["head_center"] = None

    # Steg 3: Sätt saknade keypoints till null
    # Tår och hälar finns inte i COCO-17; dessa sätts till null
    # och hanteras i coco_to_motionbert.py respektive coco_to_motionagformer.py
    for name in MISSING_IN_COCO17:
        if name not in person_dict:
            person_dict[name] = None

    return person_dict


def convert(input_path: str, output_path: str):
    """Läser ViTPose JSON-fil och skriver om den i MMPose-kompatibelt format."""
    with open(input_path, "r", encoding="utf-8") as f:
        frames = json.load(f)

    out_frames = []
    for fr in frames:
        new_persons = []
        for person in fr.get("persons", []):
            kpts_raw = person.get("keypoints", [])
            new_persons.append(convert_person(kpts_raw))

        out_frames.append({"frame": fr["frame"], "persons": new_persons})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_frames, f, ensure_ascii=False, indent=2)

    print(f"Konverterade {len(out_frames)} frames")
    print(f"In:  {input_path}")
    print(f"Ut:  {output_path}")


if __name__ == "__main__":
    convert(INPUT_JSON, OUTPUT_JSON)
