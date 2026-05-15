from mmpose.apis import MMPoseInferencer
import cv2
import subprocess
import json
import numpy as np

# ========= SÖKVÄGAR =========
# Sökväg till indatavideo samt utdatafiler
video_in = ""
out_video = ""

# Två separata JSON-filer sparas:
# 1) Namngivna keypoints (kropp + fot) för vidare analys
# 2) Alla 133 keypoints i COCO WholeBody-format
out_json_named = ""
out_json_all   = ""
# ============================

# Initierar MMPose inferencer med modellvarianten RTMw-x
# Modellen är tränad på COCO WholeBody och genererar 133 keypoints
# Inkluderar kropp (17), fötter (6), ansikte (68) och händer (42)
inferencer = MMPoseInferencer(pose2d="rtmw-x_8xb320-270e_cocktail14-384x288", device="cpu")

writer = None
all_frames_named = []
all_frames_all   = []

# ========= COCO WHOLEBODY INDEXINTERVALL =========
# Definierar vilket index-intervall varje kroppsdel täcker
# enligt COCO WholeBody-standarden (totalt 133 keypoints)
BODY_RANGE  = range(0, 17)      # Kropp:  index 0–16
FOOT_RANGE  = range(17, 23)     # Fötter: index 17–22
FACE_RANGE  = range(23, 91)     # Ansikte: index 23–90
LHAND_RANGE = range(91, 112)    # Vänster hand: index 91–111
RHAND_RANGE = range(112, 133)   # Höger hand: index 112–132

# För den namngivna JSON-filen behålls endast kropp och fot
KEEP_IDXS = set(list(BODY_RANGE) + list(FOOT_RANGE))

# Huvud-keypoints används för att beräkna ett centrumvärde för huvudet
HEAD_IDXS = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear

# Konfidenspoäng-tröskel: keypoints under detta värde räknas som osäkra
SCORE_THR = 0.20

# ========= NAMNMAPPNING FÖR KROPP + FOT (index 0–22) =========
# Mappar COCO WholeBody-index till läsbara namn
IDX_TO_NAME = {
    0:  "nose",
    1:  "left_eye",
    2:  "right_eye",
    3:  "left_ear",
    4:  "right_ear",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    17: "left_big_toe",
    18: "left_small_toe",
    19: "left_heel",
    20: "right_big_toe",
    21: "right_small_toe",
    22: "right_heel",
}

def idx_to_name_full(idx: int) -> str:
    """Returnerar ett läsbart namn för alla 133 keypoints.
    Kropp och fot får specifika namn, ansikte och händer får generiska."""
    if idx in IDX_TO_NAME:
        return IDX_TO_NAME[idx]
    if idx in FACE_RANGE:
        return f"face_{idx-23}"           # face_0 .. face_67
    if idx in LHAND_RANGE:
        return f"left_hand_{idx-91}"      # left_hand_0 .. left_hand_20
    if idx in RHAND_RANGE:
        return f"right_hand_{idx-112}"    # right_hand_0 .. right_hand_20
    return f"kp_{idx}"


def get_vis(result: dict):
    """Extraherar visualiseringsbilden (annoterad frame) från MMPose-resultatet."""
    vis = result.get("visualization", None) or result.get("vis", None)
    if vis is None:
        raise RuntimeError(f"Hittar ingen vis i result. Keys: {list(result.keys())}")
    if isinstance(vis, list):
        if len(vis) == 0:
            return None
        vis = vis[0]
    return vis


def get_preds(result: dict):
    """Extraherar prediktionerna (keypoints per person) från MMPose-resultatet."""
    for k in ("predictions", "prediction", "preds", "pred"):
        if k in result:
            return result[k]
    raise RuntimeError(f"Hittar inga predictions i result. Keys: {list(result.keys())}")


def normalize_person(person):
    """Normaliserar personobjektet till (keypoints, scores) oavsett vilket
    format MMPose returnerar. Hanterar både dict- och list-format."""
    if isinstance(person, dict):
        kpts = person.get("keypoints")
        scr = person.get("keypoint_scores") or person.get("keypoints_score") or person.get("scores")
        if kpts is None or scr is None:
            raise RuntimeError(f"Saknar keypoints/scores i dict-person. Keys: {list(person.keys())}")
        return kpts, scr

    if isinstance(person, (list, tuple)):
        if len(person) > 0 and isinstance(person[0], dict):
            return normalize_person(person[0])
        if len(person) >= 2 and isinstance(person[0], (list, tuple, np.ndarray)) and isinstance(person[1], (list, tuple, np.ndarray)):
            return person[0], person[1]

    raise RuntimeError(f"Okänt person-format: {type(person)}  Exempel: {str(person)[:200]}")


def add_center(person_dict, name, idxs, kpts, scr):
    """Beräknar och lägger till ett centrumvärde för en grupp keypoints
    (t.ex. huvud eller hand) som medelvärdet av punkter över konfidenströskeln."""
    pts = []
    scores = []
    for i in idxs:
        if i < len(kpts) and float(scr[i]) > SCORE_THR:
            pts.append(kpts[i, :2])
            scores.append(float(scr[i]))
    if pts:
        c = np.mean(np.stack(pts, axis=0), axis=0)
        person_dict[name] = {
            "x": float(c[0]),
            "y": float(c[1]),
            "score": float(min(scores)),
        }


# ========= HUVUDLOOP: BEARBETA VIDEO FRAME FÖR FRAME =========
# MMPose inferencer itererar över varje bildruta i videon
for frame_idx, result in enumerate(inferencer(video_in, show=False, return_vis=True)):

    # --- Skriv annoterad frame till utdatavideo ---
    vis = get_vis(result)
    if vis is None:
        continue

    # Initierar VideoWriter vid första frame med korrekt FPS och upplösning
    if writer is None:
        cap = cv2.VideoCapture(video_in)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or fps <= 0:
            fps = 30

        h, w = vis.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # --- Extrahera och spara keypoints ---
    preds = get_preds(result)

    frame_entry_named = {"frame": frame_idx, "persons": []}
    frame_entry_all   = {"frame": frame_idx, "persons": []}

    for person in preds:
        kpts, scr = normalize_person(person)
        kpts = np.asarray(kpts)
        scr = np.asarray(scr)

        if kpts.ndim != 2 or kpts.shape[1] < 2:
            raise RuntimeError(f"Keypoints har oväntad shape: {kpts.shape}")

        # --- Fil 1: Namngivna keypoints (kropp + fot, index 0–22) ---
        # Används som indata till 3D-lyftningsmodellerna (MotionBERT/MotionAGFormer)
        person_dict_named = {}
        for idx, ((x, y), s) in enumerate(zip(kpts[:, :2], scr)):
            if idx in KEEP_IDXS:
                name = IDX_TO_NAME.get(idx, f"kp_{idx}")
                person_dict_named[name] = {"x": float(x), "y": float(y), "score": float(s)}

        # Lägger till centrumpunkter för huvud och händer
        add_center(person_dict_named, "head_center", HEAD_IDXS, kpts, scr)
        add_center(person_dict_named, "left_hand_center", list(LHAND_RANGE), kpts, scr)
        add_center(person_dict_named, "right_hand_center", list(RHAND_RANGE), kpts, scr)
        frame_entry_named["persons"].append(person_dict_named)

        # --- Fil 2: Alla 133 keypoints i COCO WholeBody-format ---
        # Sparas för eventuell framtida analys av ansikte och händer
        person_dict_all = {}
        for idx, ((x, y), s) in enumerate(zip(kpts[:, :2], scr)):
            name = idx_to_name_full(idx)
            person_dict_all[name] = {"x": float(x), "y": float(y), "score": float(s)}
        frame_entry_all["persons"].append(person_dict_all)

    all_frames_named.append(frame_entry_named)
    all_frames_all.append(frame_entry_all)

# ========= SPARA UTDATA =========
if writer is not None:
    writer.release()

with open(out_json_named, "w", encoding="utf-8") as f:
    json.dump(all_frames_named, f, ensure_ascii=False, indent=2)

with open(out_json_all, "w", encoding="utf-8") as f:
    json.dump(all_frames_all, f, ensure_ascii=False, indent=2)

print("KLAR ✅ Sparade video:", out_video)
print("KLAR ✅ Sparade keypoints (named):", out_json_named)
print("KLAR ✅ Sparade keypoints (all):", out_json_all)

subprocess.run(["open", out_video])
