import json
import numpy as np

# ========= SÖKVÄGAR =========
# Välj indata: kommentera ut den som inte används
#JSON_NAMED = "/Users/adambostrom/Desktop/pose_keypoints_named.json"          # MMPose
JSON_NAMED = "/Users/adambostrom/Desktop/VITpose/ViTPose/vis_results/övning3_keypoints_named.json"  # ViTPose
OUT_JSON   = "/Users/adambostrom/Desktop/Kandidatarbete/alphapose_halpe26.json"
# ============================

# ========= HALPE-26 KEYPOINT-ORDNING =========
# Definierar ordningen på de 26 keypoints i HALPE-26-formatet
# som MotionBERT kräver som indata.
# Index 0–16 motsvarar COCO-17, index 17–19 är virtuella punkter
# beräknade som medelvärden, index 20–25 är tå- och hälpunkter.
HALPE26 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",          # 0–4
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",    # 5–8
    "left_wrist", "right_wrist", "left_hip", "right_hip",              # 9–12
    "left_knee", "right_knee", "left_ankle", "right_ankle",            # 13–16
    "head",             # 17 – beräknas som medelvärde av huvud-keypoints
    "neck",             # 18 – beräknas som medelvärde av axlarna
    "hip",              # 19 – beräknas som medelvärde av höfterna
    "left_big_toe", "right_big_toe",                                    # 20–21
    "left_small_toe", "right_small_toe",                                # 22–23
    "left_heel", "right_heel",                                          # 24–25
]

def load_frames(path):
    """Läser in JSON-fil med keypoints per frame från MMPose eller ViTPose."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_xy_score(person, name):
    """Hämtar x, y och konfidenspoäng för en namngiven keypoint.
    Returnerar None-värden om punkten saknas."""
    v = person.get(name)
    if isinstance(v, dict) and v.get("x") is not None and v.get("y") is not None:
        return float(v["x"]), float(v["y"]), float(v.get("score", 0.0))
    return None, None, 0.0

def midpoint(a, b):
    """Beräknar mittpunkten mellan två keypoints.
    Används för att skapa de virtuella punkterna neck och hip."""
    ax, ay, ascore = a
    bx, by, bscore = b
    if ax is None or bx is None:
        return None, None, 0.0
    score = min(ascore, bscore)
    return ((ax + bx) * 0.5, (ay + by) * 0.5, score)

def build_halpe26(person):
    """Konverterar en persons keypoints från COCO-format till HALPE-26-format.
    
    De tre virtuella punkterna beräknas enligt avsnitt 3.2.5 i rapporten:
    - head:  hämtas från head_center (medelvärde av huvud-keypoints från MMPose)
    - neck:  medelvärde av vänster och höger axel
    - hip:   medelvärde av vänster och höger höft
    
    Returnerar en platt lista med [x, y, score] för varje av de 26 punkterna,
    totalt 78 värden (26 × 3).
    """
    d = {}
    for n in [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
        "left_heel", "right_heel", "head_center"
    ]:
        d[n] = get_xy_score(person, n)

    # Beräkna virtuella punkter som saknas i COCO-formatet
    head = d["head_center"]
    neck = midpoint(d["left_shoulder"], d["right_shoulder"])
    hip  = midpoint(d["left_hip"], d["right_hip"])

    # Bygg den platta listan i HALPE-26-ordning
    out = []
    for name in HALPE26:
        if name == "head":
            x, y, s = head
        elif name == "neck":
            x, y, s = neck
        elif name == "hip":
            x, y, s = hip
        else:
            x, y, s = d.get(name, (None, None, 0.0))

        out.extend([x if x is not None else np.nan,
                    y if y is not None else np.nan,
                    s])
    return out

def bbox_from_kp(kp_flat):
    """Beräknar en bounding box runt alla giltiga keypoints i en frame.
    Bounding box krävs av MotionBERT som metadata per frame."""
    arr = np.array(kp_flat, dtype=float).reshape(26, 3)
    xs = arr[:, 0]; ys = arr[:, 1]; ss = arr[:, 2]
    m = (ss > 0) & ~np.isnan(xs) & ~np.isnan(ys)
    if not np.any(m):
        return [0.0, 0.0, 0.0, 0.0], 0.0
    x1 = float(xs[m].min()); y1 = float(ys[m].min())
    x2 = float(xs[m].max()); y2 = float(ys[m].max())
    box = [x1, y1, float(x2 - x1), float(y2 - y1)]
    score = float(ss[m].mean())
    return box, score

def main():
    frames = load_frames(JSON_NAMED)

    all_kp = []
    empty_frames = []

    # ========= KONVERTERA VARJE FRAME =========
    # Itererar över alla frames och konverterar keypoints till HALPE-26-format.
    # Frames utan detekterade personer fylls med NaN och flaggas.
    for i, fr in enumerate(frames):
        persons = fr.get("persons", []) or []
        if not persons:
            all_kp.append([np.nan] * (26 * 3))
            empty_frames.append(i)
        else:
            # Endast första detekterade personen används per frame
            kp = build_halpe26(persons[0])
            all_kp.append(kp)

    print(f"Antal frames: {len(all_kp)}")
    print(f"Tomma frames (ingen detektion): {len(empty_frames)}")

    arr = np.array(all_kp, dtype=float)

    # ========= BYGG MOTIONBERT-KOMPATIBEL JSON =========
    # Skapar en lista av frame-objekt med keypoints, bounding box och score
    # i det format som MotionBERT förväntar sig som indata.
    results = []
    for i in range(len(frames)):
        kp = arr[i].tolist()
        kp = [0.0 if np.isnan(v) else v for v in kp]   # Ersätter NaN med 0
        box, score = bbox_from_kp(kp)
        results.append({
            "image_id": i,      # Framenummer
            "idx": 0,           # Personindex (alltid 0 då vi tar första personen)
            "keypoints": kp,    # 78 värden: [x, y, score] × 26 keypoints
            "box": box,         # Bounding box: [x, y, bredd, höjd]
            "score": score,     # Genomsnittlig konfidenspoäng för framen
        })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print("Sparade:", OUT_JSON, "| Antal frames:", len(results))

if __name__ == "__main__":
    main()