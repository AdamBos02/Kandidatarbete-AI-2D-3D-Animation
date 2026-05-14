"""
Kör_motionbert.py
─────────────────
Kör MotionBERT-inferens på ett videoklipp med tillhörande 2D-keypoints
i HALPE-26-format (genererade av coco_to_motionbert.py) och sparar:
  - X3D.npy – 3D-koordinater i Human3.6M-format, shape (N_frames, 17, 3)
  - X3D.mp4 – renderad video med 3D-skelett

Detta script motsvarar det tredje steget i arbetsflödet:
  2D-estimering (MMPose/ViTPose) → formatkonvertering → 3D-lyftning (MotionBERT)

Kör så här i terminalen:
  python Kör_motionbert.py
"""

import os
import sys
import subprocess
import numpy as np
import imageio

# Lägg till MotionBERT-katalogen i Python-sökvägen
MOTIONBERT_PATH = '/Users/adambostrom/Desktop/MotionBERT'
sys.path.append(MOTIONBERT_PATH)
from lib.utils.vismo import render_and_save

# ========= INSTÄLLNINGAR =========
# Sökväg till originalvideo (används för FPS och renderad output)
VIDEO_PATH = "/Users/adambostrom/Desktop/Qualisys mätningar/Indianhopp_viktor/IMG_5478_sync.mp4"

# Sökväg till 2D-keypoints i HALPE-26-format från coco_to_motionbert.py
JSON_PATH  = "/Users/adambostrom/Desktop/Kandidatarbete/alphapose_halpe26.json"

# Katalog där utdatafilerna sparas
OUT_PATH   = "/Users/adambostrom/Desktop/Qualisys mätningar"

# Modellkonfiguration och vikter för MotionBERT-varianten FT_MB_ft_h36m
# Tränad på Human3.6M-datasetet enligt avsnitt 3.2.4 i rapporten
CONFIG     = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
CHECKPOINT = "checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch_stripped.bin"
# =================================


def get_fps(video_path):
    """Läser FPS från videofilens metadata för att bevara
    korrekt uppspelningshastighet i den renderade outputvideon."""
    vid = imageio.get_reader(video_path, 'ffmpeg')
    fps = vid.get_meta_data()['fps']
    vid.close()
    return fps


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    fps = get_fps(VIDEO_PATH)

    # ========= STEG 1: KÖR MOTIONBERT-INFERENS =========
    # Anropar MotionBERTs infer_wild.py med videon och 2D-keypoints som indata.
    # Modellen lyfter de 2D-keypoints till 3D-koordinater i Human3.6M-format
    # och sparar resultatet som X3D.npy i OUT_PATH.
    print("Kör MotionBERT inferens...")
    cmd = [
        "python", "infer_wild.py",
        "--config", CONFIG,
        "--vid_path", VIDEO_PATH,
        "--json_path", JSON_PATH,
        "--out_path", OUT_PATH,
        "-e", CHECKPOINT,
    ]
    subprocess.run(cmd, check=True)
    print("Inferens klar!")

    # ========= STEG 2: LADDA 3D-KOORDINATER =========
    # X3D.npy innehåller de estimerade 3D-koordinaterna med
    # shape (N_frames, 17, 3) i Human3.6M-format
    npy_path = os.path.join(OUT_PATH, "X3D.npy")
    poses = np.load(npy_path)
    print(f"Laddade 3D-koordinater: shape={poses.shape}")

    # ========= STEG 3: RENDERA VIDEO =========
    # Genererar en video med det animerade 3D-skelettet
    # för visuell inspektion av resultatet
    print("Renderar video...")
    render_and_save(
        poses,
        os.path.join(OUT_PATH, "X3D.mp4"),
        keep_imgs=False,
        fps=fps
    )

    print(f"\nKlart! Filer sparade i: {OUT_PATH}")
    print("  X3D.npy – 3D-koordinater, shape (N_frames, 17, 3)")
    print("  X3D.mp4 – renderad video med 3D-skelett")


if __name__ == "__main__":
    main()