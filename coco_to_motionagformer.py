import json
import numpy as np

# ========= SÖKVÄGAR =========
# Välj indata: kommentera ut den som inte används
#JSON_NAMED = "/Users/adambostrom/Desktop/pose_keypoints_named.json"          # MMPose
JSON_NAMED = "/Users/adambostrom/Desktop/VITpose/ViTPose/vis_results/övning3_keypoints_named.json"  # ViTPose
OUT_NPZ    = "/Users/adambostrom/Desktop/Kandidatarbete/MotionAGformer2D.json"
# ============================

# ========= HUMAN3.6M KEYPOINT-ORDNING =========
# Definierar de 17 keypoints i Human3.6M-formatet som MotionAGFormer
# kräver som indata. Ordningen följer standarden i figur 2.2 i rapporten.
# Tre virtuella punkter beräknas som medelvärden enligt avsnitt 3.2.5:
# hip_center (0), spine (7) och neck (8).
H36M_JOINTS = [
    "hip_center",     # 0  – medelvärde av vänster och höger höft
    "right_hip",      # 1
    "right_knee",     # 2
    "right_ankle",    # 3
    "left_hip",       # 4
    "left_knee",      # 5
    "left_ankle",     # 6
    "spine",          # 7  – medelvärde av hip_center och neck
    "neck",           # 8  – medelvärde av vänster och höger axel
    "nose",           # 9
    "head",           # 10
    "left_shoulder",  # 11
    "left_elbow",     # 12
    "left_wrist",     # 13
    "right_shoulder", # 14
    "right_elbow",    # 15
    "right_wrist",    # 16
]


def get_xy_score(person, name):
    """Hämtar x, y och konfidenspoäng för en namngiven keypoint.
    Returnerar None om punkten saknas."""
    v = person.get(name)
    if isinstance(v, dict) and v.get("x") is not None:
        return np.array([float(v["x"]), float(v["y"])]), float(v.get("score", 0.0))
    return None, 0.0

def midpoint(a, b):
    """Beräknar mittpunkten mellan två keypoints som ett aritmetiskt medelvärde.
    Används för att skapa de virtuella punkterna hip_center, neck och spine."""
    return (a + b) * 0.5

def build_h36m(person):
    """Konverterar en persons keypoints från COCO-format till Human3.6M-format.
    
    De tre virtuella punkterna beräknas enligt avsnitt 3.2.5 i rapporten:
    - hip_center: medelvärde av vänster och höger höft
    - neck:       medelvärde av vänster och höger axel  
    - spine:      medelvärde av hip_center och neck
    
    Returnerar två arrayer:
    - kp: keypoint-koordinater med shape (17, 2)
    - sc: konfidenspoäng med shape (17,)
    """
    kp = np.zeros((17, 2), dtype=np.float32)
    sc = np.zeros(17, dtype=np.float32)

    # Hämta de punkter som behövs för att beräkna virtuella joints
    ls, ls_s = get_xy_score(person, "left_shoulder")
    rs, rs_s = get_xy_score(person, "right_shoulder")
    lh, lh_s = get_xy_score(person, "left_hip")
    rh, rh_s = get_xy_score(person, "right_hip")

    # Beräkna virtuella punkter som saknas i COCO-formatet
    hip_center = midpoint(lh, rh)   if lh is not None and rh is not None else None
    neck       = midpoint(ls, rs)   if ls is not None and rs is not None else None
    spine      = midpoint(hip_center, neck) if hip_center is not None and neck is not None else None

    # Bygg upp ett dictionary med alla 17 joints och deras konfidenspoäng
    joints = {
        "hip_center":     (hip_center, min(lh_s, rh_s) if lh is not None and rh is not None else 0.0),
        "right_hip":      get_xy_score(person, "right_hip"),
        "right_knee":     get_xy_score(person, "right_knee"),
        "right_ankle":    get_xy_score(person, "right_ankle"),
        "left_hip":       get_xy_score(person, "left_hip"),
        "left_knee":      get_xy_score(person, "left_knee"),
        "left_ankle":     get_xy_score(person, "left_ankle"),
        "spine":          (spine, min(ls_s, rs_s, lh_s, rh_s) if spine is not None else 0.0),
        "neck":           (neck, min(ls_s, rs_s) if neck is not None else 0.0),
        "nose":           get_xy_score(person, "nose"),
        "head":           get_xy_score(person, "head_center"),
        "left_shoulder":  get_xy_score(person, "left_shoulder"),
        "left_elbow":     get_xy_score(person, "left_elbow"),
        "left_wrist":     get_xy_score(person, "left_wrist"),
        "right_shoulder": get_xy_score(person, "right_shoulder"),
        "right_elbow":    get_xy_score(person, "right_elbow"),
        "right_wrist":    get_xy_score(person, "right_wrist"),
    }

    # Fyll arrayerna i Human3.6M-ordning; saknade punkter sätts till NaN
    for i, name in enumerate(H36M_JOINTS):
        xy, score = joints[name]
        if xy is not None:
            kp[i] = xy
            sc[i] = score
        else:
            kp[i] = np.array([np.nan, np.nan])
            sc[i] = 0.0

    return kp, sc


def main():
    with open(JSON_NAMED, "r", encoding="utf-8") as f:
        frames = json.load(f)

    all_kp = []
    all_sc = []

    # ========= KONVERTERA VARJE FRAME =========
    # Itererar över alla frames och konverterar keypoints till Human3.6M-format.
    # Frames utan detekterade personer fylls med NaN.
    for fr in frames:
        persons = fr.get("persons", [])
        if not persons:
            all_kp.append(np.full((17, 2), np.nan, dtype=np.float32))
            all_sc.append(np.zeros(17, dtype=np.float32))
        else:
            # Endast första detekterade personen används per frame
            kp, sc = build_h36m(persons[0])
            all_kp.append(kp)
            all_sc.append(sc)

    # Stapla till arrayer med shape (N_frames, 17, 2) respektive (N_frames, 17)
    keypoints = np.stack(all_kp)
    scores    = np.stack(all_sc)

    print(f"Antal frames: {keypoints.shape[0]}")
    print(f"Saknade leder (NaN): {np.isnan(keypoints).sum()}")

    # ========= SPARA UTDATA =========
    # Sparar keypoints och scores som en komprimerad NumPy-fil (.npz)
    # i det format som MotionAGFormer förväntar sig som indata.
    np.savez(OUT_NPZ, keypoints=keypoints, scores=scores)
    print(f"Sparade: {OUT_NPZ}  shape: {keypoints.shape}")

if __name__ == "__main__":
    main()