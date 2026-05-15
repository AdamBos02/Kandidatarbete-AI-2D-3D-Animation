import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── COCO17 ──────────────────────────────────────────────────────────────────
# Index → namn (för referens):
#  0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
#  5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
#  9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle

COCO_SKELETON = [
    (0, 1), (0, 2),           # näsa → ögon
    (1, 3), (2, 4),           # ögon → öron
    (5, 6),                   # axlar
    (5, 7), (7, 9),           # vänster arm
    (6, 8), (8, 10),          # höger arm
    (5, 11), (6, 12),         # axlar → höfter
    (11, 12),                 # höfter
    (11, 13), (13, 15),       # vänster ben
    (12, 14), (14, 16),       # höger ben
]

COCO_POS = np.array([
    [0.50, 0.12],  # 0  nose
    [0.46, 0.10],  # 1  left_eye
    [0.54, 0.10],  # 2  right_eye
    [0.42, 0.10],  # 3  left_ear
    [0.58, 0.10],  # 4  right_ear
    [0.40, 0.27],  # 5  left_shoulder
    [0.60, 0.27],  # 6  right_shoulder
    [0.33, 0.40],  # 7  left_elbow
    [0.67, 0.40],  # 8  right_elbow
    [0.27, 0.52],  # 9  left_wrist
    [0.73, 0.52],  # 10 right_wrist
    [0.42, 0.54],  # 11 left_hip
    [0.58, 0.54],  # 12 right_hip
    [0.40, 0.72],  # 13 left_knee
    [0.60, 0.72],  # 14 right_knee
    [0.41, 0.90],  # 15 left_ankle
    [0.59, 0.90],  # 16 right_ankle
])

# ── Human3.6M ────────────────────────────────────────────────────────────────
# Index → namn (för referens):
#  0=hip_center, 1=R_hip,  2=R_knee,    3=R_ankle,
#  4=L_hip,      5=L_knee, 6=L_ankle,   7=spine,
#  8=thorax,     9=neck,  10=head,
# 11=L_shoulder,12=L_elbow,13=L_wrist,
# 14=R_shoulder,15=R_elbow,16=R_wrist

H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),           # höger ben
    (0, 4), (4, 5), (5, 6),           # vänster ben
    (0, 7), (7, 8), (8, 9), (9, 10),  # ryggrad → huvud
    (8, 11), (11, 12), (12, 13),      # vänster arm
    (8, 14), (14, 15), (15, 16),      # höger arm
]

H36M_POS = np.array([
    [0.50, 0.54],  # 0  hip_center
    [0.58, 0.57],  # 1  R_hip
    [0.60, 0.73],  # 2  R_knee
    [0.59, 0.90],  # 3  R_ankle
    [0.42, 0.57],  # 4  L_hip
    [0.40, 0.73],  # 5  L_knee
    [0.41, 0.90],  # 6  L_ankle
    [0.50, 0.44],  # 7  spine
    [0.50, 0.27],  # 8  thorax
    [0.50, 0.20],  # 9  neck
    [0.50, 0.15],  # 10 head
    [0.40, 0.27],  # 11 L_shoulder
    [0.33, 0.44],  # 12 L_elbow
    [0.27, 0.54],  # 13 L_wrist
    [0.60, 0.27],  # 14 R_shoulder
    [0.67, 0.44],  # 15 R_elbow
    [0.73, 0.54],  # 16 R_wrist
])

# ── Ritfunktion ──────────────────────────────────────────────────────────────

def draw_skeleton(ax, pos, skeleton, color, title):
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(1.0, 0.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Ben
    for i, j in skeleton:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color=color, lw=2.0, zorder=1, solid_capstyle="round",
        )

    # Punkter
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=80, color=color, zorder=3,
        edgecolors="white", linewidths=1.2,
    )

    # Siffror bredvid varje punkt  ← ERSÄTT DETTA BLOCK
    for idx, (x, y) in enumerate(pos):
        ax.text(
            x + 0.0062, y - 0.02,
            str(idx),
            fontsize=15,
            color="#222222",
            zorder=5,
            ha="left", va="center",
        )

# ── Plotta ───────────────────────────────────────────────────────────────────

# Bild 1: COCO17
fig1, ax1 = plt.subplots(figsize=(5, 7))
fig1.patch.set_facecolor("#F8F7F4")
ax1.set_facecolor("#F8F7F4")
draw_skeleton(ax1, COCO_POS, COCO_SKELETON, "#534AB7", "")
plt.tight_layout(pad=2.0)
plt.savefig("coco17_keypoints.png", dpi=150, bbox_inches="tight",
            facecolor=fig1.get_facecolor())
plt.show()
 
# Bild 2: Human3.6M
fig2, ax2 = plt.subplots(figsize=(5, 7))
fig2.patch.set_facecolor("#F8F7F4")
ax2.set_facecolor("#F8F7F4")
draw_skeleton(ax2, H36M_POS, H36M_SKELETON, "#0F6E56", "")
plt.tight_layout(pad=2.0)
plt.savefig("human36m_keypoints.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.show()