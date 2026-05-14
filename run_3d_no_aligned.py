"""
run_3d_no_aligned.py
─────────────────────
Kör MotionAGFormer-inferens på ett videoklipp med tillhörande 2D-keypoints
i Human3.6M-format (genererade av coco_to_motionagformer.py) och sparar:
  - X3D.npy – 3D-koordinater i Human3.6M-format, shape (N_frames, 17, 3)
  - X3D.mp4 – renderad video med 3D-skelett

Detta script motsvarar det tredje steget i arbetsflödet:
  2D-estimering (MMPose/ViTPose) → formatkonvertering → 3D-lyftning (MotionAGFormer)

Modellvarianten motionagformer-s-ap3d är tränad på AthletePose3D-datasetet,
ett dataset specifikt anpassat för idrottsrörelser (se avsnitt 3.2.4 i rapporten).

Kör så här i terminalen:
  python run_3d_no_aligned.py
"""

import os
import sys
import copy
import glob
import io
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import cv2
from tqdm import tqdm

# Sätt arbetskatalog till MotionAGFormer och lägg till i sökvägen
os.chdir('/Users/adambostrom/Desktop/MotionAGFormer/MotionAGFormer')
sys.path.append(os.getcwd())

from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

plt.switch_backend('agg')

# ========= INSTÄLLNINGAR =========
# Sökväg till originalvideo (används för FPS och bildstorlek)
VIDEO_PATH  = '/Users/adambostrom/Desktop/Qualisys mätningar/Indianhopp_viktor/IMG_5478_sync.mp4'

# Sökväg till 2D-keypoints i Human3.6M-format från coco_to_motionagformer.py
NPZ_PATH    = '/Users/adambostrom/Desktop/Kandidatarbete/MotionAGformer2D.json.npz'

# Katalog där utdatafilerna sparas
OUT_PATH    = '/Users/adambostrom/Desktop/Qualisys mätningar'

# Modellvikter för MotionAGFormer tränad på AthletePose3D
MODEL_PATH  = 'checkpoint/motionagformer-s-ap3d.pth.tr'
# =================================


def show3Dpose(vals, ax, fixed_limits=None):
    """Ritar 3D-skelettet i ett matplotlib-axelobjekt.
    
    Skelettet definieras av ben (I→J) med sidofärgning (LR):
    blå = vänster sida, röd = höger sida.
    Fasta axelgränser (fixed_limits) används för konsekvent
    skalning mellan frames i den renderade videon.
    """
    ax.view_init(elev=15., azim=70)
    lcolor = (0, 0, 1)  # Blå = vänster
    rcolor = (1, 0, 0)  # Röd = höger

    # Skelettstruktur: par av joint-index som definierar ben
    I  = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12,  8,  9])
    J  = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13,  9, 10])
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0,  0,  1,  0,  0,  1,  1,  0,  0], dtype=bool)

    for i in range(len(I)):
        xs, ys, zs = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(-xs, -zs, -ys, lw=2, color=lcolor if LR[i] else rcolor)

    if fixed_limits is not None:
        ax.set_xlim3d(fixed_limits[0])
        ax.set_ylim3d(fixed_limits[1])
        ax.set_zlim3d(fixed_limits[2])
    else:
        RADIUS, RADIUS_Z = 0.72, 0.7
        xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
        ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
        ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
        ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])

    ax.set_aspect('auto')
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def resample(n_frames):
    """Samplar om ett klipp till exakt 81 frames med jämn fördelning.
    MotionAGFormer kräver sekvenser av fast längd (81 frames) som indata."""
    even = np.linspace(0, n_frames, num=81, endpoint=False)
    result = np.floor(even)
    return np.clip(result, 0, n_frames - 1).astype(np.uint32)


def turn_into_clips(keypoints):
    """Delar upp keypoint-sekvensen i klipp om 81 frames.
    Kortare sekvenser samplas om till 81 frames via resample().
    Returnerar klippen och ett downsample-index för sista klippet."""
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 81:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 81):
            clip = keypoints[:, start_idx:start_idx + 81, ...]
            if clip.shape[1] != 81:
                new_indices = resample(clip.shape[1])
                clips.append(clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(clip)
                downsample = np.arange(81)
    return clips, downsample


def flip_data(data, left_joints=[1,2,3,14,15,16], right_joints=[4,5,6,11,12,13]):
    """Speglar keypoints horisontellt och byter vänster/höger-joints.
    Används för test-time augmentation: modellen körs på både originaldata
    och speglad data, och resultaten medelvärdesbildas för ökad robusthet."""
    flipped = copy.deepcopy(data)
    flipped[..., 0] *= -1
    flipped[..., left_joints + right_joints, :] = flipped[..., right_joints + left_joints, :]
    return flipped


def main():
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(OUT_PATH + 'input_2D/', exist_ok=True)

    # ========= STEG 1: LADDA OCH KONVERTERA KEYPOINTS =========
    # Läser in 2D-keypoints från coco_to_motionagformer.py och lägger
    # till konfidenspoäng som en tredje koordinat per keypoint.
    # Sparar i det format som MotionAGFormer förväntar sig: (1, N_frames, 17, 3)
    raw = np.load(NPZ_PATH)
    keypoints_raw = raw['keypoints']
    scores_raw    = raw['scores']
    keypoints_with_conf = np.concatenate([keypoints_raw, scores_raw[..., None]], axis=-1)
    keypoints_batched   = keypoints_with_conf[None, ...]
    np.savez_compressed(OUT_PATH + 'input_2D/keypoints.npz', reconstruction=keypoints_batched)
    print(f"Keypoints konverterade! Shape: {keypoints_batched.shape}")

    # ========= STEG 2: LADDA MODELLEN =========
    # Initierar MotionAGFormer med de hyperparametrar som
    # används av varianten motionagformer-s-ap3d
    args = dict(
        n_layers=26, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3,
        mlp_ratio=4, act_layer=nn.GELU,
        attn_drop=0.0, drop=0.0, drop_path=0.0,
        use_layer_scale=True, layer_scale_init_value=0.00001,
        use_adaptive_fusion=True, num_heads=8, qkv_bias=False, qkv_scale=None,
        hierarchical=False, use_temporal_similarity=True, neighbour_num=2,
        temporal_connection_len=1, use_tcn=False, graph_only=False, n_frames=81
    )
    model = nn.DataParallel(MotionAGFormer(**args))
    pre_dict = torch.load(sorted(glob.glob(MODEL_PATH))[0], map_location='cpu')
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    # ========= STEG 3: LADDA KEYPOINTS OCH VIDEOINFO =========
    keypoints = np.load(OUT_PATH + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    clips, downsample = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(VIDEO_PATH)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, img = cap.read()
    img_size = img.shape
    cap.release()

    # ========= STEG 4: KÖR INFERENS =========
    # Kör modellen med test-time augmentation: varje klipp processas
    # både i originalriktning och speglat, och resultaten medelvärdesbildas.
    # Koordinaterna normaliseras till [-1, 1] baserat på bildstorlek
    # innan de skickas till modellen.
    print("\nKör MotionAGFormer inferens...")
    all_3d_poses = []

    with torch.no_grad():
        for idx, clip in enumerate(tqdm(clips)):
            input_2D     = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
            input_2D_aug = flip_data(input_2D)
            input_2D     = torch.from_numpy(input_2D.astype('float32'))
            input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))

            out_non_flip = model(input_2D)
            out_flip     = flip_data(model(input_2D_aug))
            output_3D    = (out_non_flip + out_flip) / 2  # Medelvärde av original och speglat

            if idx == len(clips) - 1:
                output_3D = output_3D[:, downsample]

            for post_out in output_3D[0].cpu().detach().numpy():
                all_3d_poses.append(post_out.copy())

    # Spara 3D-koordinater: shape (N_frames, 17, 3) i Human3.6M-format
    poses = np.array(all_3d_poses)
    np.save(os.path.join(OUT_PATH, 'X3D.npy'), poses)
    print(f"Inferens klar! Shape: {poses.shape}")

    # ========= STEG 5: BERÄKNA FASTA AXELGRÄNSER =========
    # Beräknar gemensamma axelgränser för hela sekvensen så att
    # skalan är konsekvent mellan frames i den renderade videon
    pad = 0.1
    fixed_limits = [
        [(-poses[:, :, 0].max()) - pad, (-poses[:, :, 0].min()) + pad],
        [(-poses[:, :, 2].max()) - pad, (-poses[:, :, 2].min()) + pad],
        [(-poses[:, :, 1].max()) - pad, (-poses[:, :, 1].min()) + pad],
    ]

    # ========= STEG 6: RENDERA VIDEO =========
    # Genererar en video frame för frame genom att rita 3D-skelettet
    # med matplotlib och enkoda varje frame direkt till videoströmmen
    print("\nGenererar video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = None

    for i, post_out in enumerate(tqdm(poses)):
        fig = plt.figure(figsize=(9.6, 5.4))
        gs  = gridspec.GridSpec(1, 1)
        ax  = plt.subplot(gs[0], projection='3d')
        show3Dpose(post_out, ax, fixed_limits=fixed_limits)

        buf = io.BytesIO()
        plt.savefig(buf, dpi=100, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        buf.close()

        if video_out is None:
            h, w = frame.shape[:2]
            video_out = cv2.VideoWriter(os.path.join(OUT_PATH, 'X3D.mp4'), fourcc, fps, (w, h))
        video_out.write(frame)

    video_out.release()
    print(f"\nKlart! Filer sparade i: {OUT_PATH}")
    print("  X3D.npy – 3D-koordinater, shape (N_frames, 17, 3)")
    print("  X3D.mp4 – renderad video med 3D-skelett")


if __name__ == "__main__":
    main()