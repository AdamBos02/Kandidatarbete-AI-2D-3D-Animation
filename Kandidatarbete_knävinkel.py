from pathlib import Path
import csv
import os
import struct
import zlib

import numpy as np
import matplotlib.pyplot as plt



QUALISYS_PATH   = Path(r"C:\Users\tjede\Downloads\indianhopp.fbx")
QUICKMAGIC_PATH = Path(r"C:\Users\tjede\Downloads\Indianhopp_fram_Mixamo.fbx")
MIMEM_PATH      = Path(r"C:\Users\tjede\Downloads\indianhopp_mimem_Unity.fbx")
OPENMOCAP_CSV   = Path(r"C:\Users\tjede\Downloads\Indianhopp_blender.csv")


POSE_ESTIMATORS = [
    ("MMPose+MotionBERT",      Path(r"C:\Users\tjede\Downloads\Mmpose_motionbert_indianhopp_adam.npy"),    60, "#2ca02c", "-."),
    ("MMPose+MotionAGFormer",  Path(r"C:\Users\tjede\Downloads\Mmpose_AGformer_indianhopp_adam.npy"),      60, "#9467bd", ":"),
    ("ViTPose+MotionBERT",     Path(r"C:\Users\tjede\Downloads\VITpose_Motionbert_indianhopp_adam.npy"),   60, "#ff7f0e", "-."),
    ("ViTPose+MotionAGFormer", Path(r"C:\Users\tjede\Downloads\VITPose_AGFormer_indianhopp_adam.npy"),     60, "#e377c2", ":"),
]

QUALISYS_FPS    = 120
QUICKMAGIC_FPS  = 60
MIMEM_FPS       = 60
COMMON_FPS      = 120


STATS_INTERVAL  = (6.3, 7.4)

#qualisys markörer
QUALISYS_MARKERS = {
    "L": {
        "fot":      "New 0001",
        "underben": "New 0018",
        "kna":      "New 0015",
        "lar":      "New 0019",
    },
    "R": {
        "fot":      "New 0011",
        "underben": "New 0006",
        "kna":      "New 0012",
        "lar":      "New 0010",
    },
}

# quickmagic — mixamo-rigg 
QUICKMAGIC_BONES = {
    "L": {"hip": "mixamorig:LeftUpLeg",  "knee": "mixamorig:LeftLeg",  "ankle": "mixamorig:LeftFoot"},
    "R": {"hip": "mixamorig:RightUpLeg", "knee": "mixamorig:RightLeg", "ankle": "mixamorig:RightFoot"},
}

# mimem —  auto-rig pro-rigg

MIMEM_BONES = {
    "L": {"hip": "thigh_stretch.l", "knee": "leg_stretch.l", "ankle": "foot.l"},
    "R": {"hip": "thigh_stretch.r", "knee": "leg_stretch.r", "ankle": "foot.r"},
}

# human3.6m för .npy-filer
H36M_INDICES = {
    "L": {"hip": 4, "knee": 5, "ankle": 6},
    "R": {"hip": 1, "knee": 2, "ankle": 3},
}

# open mocap mediaPipe
OPENMOCAP_INDICES = {
    "L": {"hip": "23", "knee": "25", "ankle": "27"},
    "R": {"hip": "24", "knee": "26", "ankle": "28"},
}


def angle_between(v1, v2): #beräknar vinkeln mellan vektorerna v1 och v2 i grader med skalärprodukt och normering
    dot = np.einsum("ij,ij->i", v1, v2)
    n1  = np.linalg.norm(v1, axis=1)
    n2  = np.linalg.norm(v2, axis=1)
    return np.degrees(np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0)))


def resample_to_fps(signal, fps_in, target_t, shift=0): #resampla en signal som ursprungligen är i fps_in till en ny tidsbas med fps_out, returnerar den resamplade signalen och dess nya tidsvektor
    n_in = len(signal)
    t_in = np.arange(n_in) / fps_in + shift
    return np.interp(target_t, t_in, signal, left=np.nan, right=np.nan)
#

class FBXReader: #laddar fbx-filer och extraherar marker-positioner
    def __init__(self, path):
        self.f    = open(path, "rb")
        self.size = os.path.getsize(path)
        self.f.read(23)
        self.version = struct.unpack("<I", self.f.read(4))[0]
        self.is64    = self.version >= 7500

    def _hdr(self): # markerar starten av en nod och hur lång den är, samt hur många egenskaper den har
        if self.is64: 
            end = struct.unpack("<Q", self.f.read(8))[0]
            n   = struct.unpack("<Q", self.f.read(8))[0]; self.f.read(8)
        else:
            end = struct.unpack("<I", self.f.read(4))[0]
            n   = struct.unpack("<I", self.f.read(4))[0]; self.f.read(4)
        return end, n, self.f.read(1)[0]

    def _prop(self): #läser en egenskap, returnerar dess värde
        c = self.f.read(1).decode("ascii")
        if c == "Y": return struct.unpack("<h", self.f.read(2))[0]
        if c == "C": return struct.unpack("<?", self.f.read(1))[0]
        if c == "I": return struct.unpack("<i", self.f.read(4))[0]
        if c == "F": return struct.unpack("<f", self.f.read(4))[0]
        if c == "D": return struct.unpack("<d", self.f.read(8))[0]
        if c == "L": return struct.unpack("<q", self.f.read(8))[0]
        if c in "fdlib":
            length = struct.unpack("<I", self.f.read(4))[0]
            enc    = struct.unpack("<I", self.f.read(4))[0]
            clen   = struct.unpack("<I", self.f.read(4))[0]
            data   = self.f.read(clen)
            if enc == 1: data = zlib.decompress(data)
            fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "?"}[c]
            return list(struct.unpack("<" + fmt * length, data))
        if c in ("S", "R"):
            length = struct.unpack("<I", self.f.read(4))[0]
            d = self.f.read(length)
            return d.decode("utf-8", errors="replace") if c == "S" else d
        raise ValueError(f"Okand FBX-typ: {c!r}")

    def _node(self): #läs en nod och dess barn, returnerar en dict
        end, np_, nl = self._hdr()
        if end == 0: return None
        name  = self.f.read(nl).decode("utf-8", errors="replace")
        props = [self._prop() for _ in range(np_)]
        children = []
        while self.f.tell() < end:
            ch = self._node()
            if ch is None: break
            children.append(ch)
        self.f.seek(end)
        return {"name": name, "props": props, "children": children} 

    def parse(self): #läs hela filen och returnera en lista av noder
        nodes = []
        while self.f.tell() < self.size - 100:
            try:
                n = self._node()
                if n is None: break
                nodes.append(n)
            except Exception: break
        return nodes


def _find(nodes, name): #sök efter en nod med givet namn i en lista av noder, returnerar den 
    return next((n for n in nodes if n["name"] == name), None)


# qualisys-laddning
def load_qualisys(fbx_path): #laddar en Qualisys .fbx-fil och extraherar marker-positioner över tid
    nodes      = FBXReader(fbx_path).parse()
    objects    = _find(nodes, "Objects") 
    connections = _find(nodes, "Connections") 
    id_to_obj  = {c["props"][0]: c for c in objects["children"] if c["props"]}
    conns      = [c["props"] for c in connections["children"] if c["name"] == "C"]

    model_to_cn = {}
    for c in conns: 
        if len(c) >= 4 and c[0] == "OP":
            s, d, p = c[1], c[2], c[3]
            if d in id_to_obj and id_to_obj[d]["name"] == "Model" and p == "Lcl Translation":
                model_to_cn.setdefault(d, []).append(s)
    cn_to_curves = {}
    for c in conns: 
        if len(c) >= 4 and c[0] == "OP":
            s, d, p = c[1], c[2], c[3]
            if d in id_to_obj and id_to_obj[d]["name"] == "AnimationCurveNode":
                cn_to_curves.setdefault(d, []).append((s, p))

    def curve(nid): #läser en animation curve, dvs en sekvens av keyframes, och returnerar dess keyframe-tider och värden 
        if nid not in id_to_obj: return None, None
        t = v = None
        for c in id_to_obj[nid]["children"]:
            if c["name"] == "KeyTime":         t = c["props"][0] if c["props"] else []
            elif c["name"] == "KeyValueFloat": v = c["props"][0] if c["props"] else []
        return t, v

    markers_by_name = {}
    for oid, obj in id_to_obj.items():
        if (obj["name"] == "Model" and len(obj["props"]) > 2
                and obj["props"][2] == "OpticalMarker"):
            markers_by_name[obj["props"][1].split("\x00")[0]] = oid

    all_times = set(); raw = {}
    for name, oid in markers_by_name.items():
        cns = model_to_cn.get(oid, [])
        if not cns: raw[name] = None; continue
        ch = {"d|X": None, "d|Y": None, "d|Z": None}
        for cid, prop in cn_to_curves.get(cns[0], []):
            if prop in ch: ch[prop] = cid
        tx, xs = curve(ch["d|X"]); _, ys = curve(ch["d|Y"]); _, zs = curve(ch["d|Z"])
        raw[name] = (tx, xs, ys, zs)
        if tx: all_times.update(tx)

    times = sorted(all_times); t2i = {t: i for i, t in enumerate(times)}; N = len(times)
    positions = {}
    for name, r in raw.items():
        arr = np.full((N, 3), np.nan)
        if r is not None and r[0]:
            for t, x, y, z in zip(*r):
                i = t2i.get(t)
                if i is not None: arr[i] = (x, y, z)
        positions[name] = arr
    return positions, N


def qualisys_knee_angles(markers): #beräknar knävinkeln över tid baserat på marker-positioner från Qualisys, returnerar en dict med vinklar för vänster och höger knä
    out = {}
    for side, m in QUALISYS_MARKERS.items():
        lar   = markers[m["lar"]]
        kna   = markers[m["kna"]]
        knee  = markers[m["underben"]]
        foot  = markers[m["fot"]]
        out[side] = 180.0 - angle_between(kna - lar, foot - knee)
    return out


# riggat skellett för QuickMagic och Mimem  
def _euler_xyz_to_matrix(e_deg): #omvandlar euler-vinklar i grader till en 3x3 rotationsmatris, returnerar en array av matriser
    e  = np.deg2rad(e_deg); ex, ey, ez = e[:,0], e[:,1], e[:,2]
    c, s = np.cos, np.sin; N = e.shape[0]
    Rx = np.zeros((N,3,3)); Rx[:,0,0]=1; Rx[:,1,1]=c(ex); Rx[:,1,2]=-s(ex); Rx[:,2,1]=s(ex); Rx[:,2,2]=c(ex)
    Ry = np.zeros((N,3,3)); Ry[:,0,0]=c(ey); Ry[:,0,2]=s(ey); Ry[:,1,1]=1; Ry[:,2,0]=-s(ey); Ry[:,2,2]=c(ey)
    Rz = np.zeros((N,3,3)); Rz[:,0,0]=c(ez); Rz[:,0,1]=-s(ez); Rz[:,1,0]=s(ez); Rz[:,1,1]=c(ez); Rz[:,2,2]=1
    return np.einsum("nij,njk->nik", np.einsum("nij,njk->nik", Rz, Ry), Rx)


def load_rigged_joint_positions(fbx_path, target_bone_names): #laddar en riggad .fbx-fil från QuickMagic eller Mimem och extraherar position över tid för specifika ben, returnerar en dict med ben-namn som nycklar och positioner som värden
    nodes      = FBXReader(fbx_path).parse()
    objects    = _find(nodes, "Objects")
    connections = _find(nodes, "Connections")
    id_to_obj  = {c["props"][0]: c for c in objects["children"] if c["props"]}
    parent_of  = {}; cn_connects = []; curve_connects = []

    for conn in connections["children"]:
        if conn["name"] != "C": continue
        p = conn["props"]
        if p[0] == "OO" and p[1] in id_to_obj and p[2] in id_to_obj:
            if id_to_obj[p[1]]["name"] == "Model" and id_to_obj[p[2]]["name"] == "Model":
                parent_of[p[1]] = p[2]
        elif p[0] == "OP":
            s, d, pr = p[1], p[2], p[3]
            if d in id_to_obj:
                if   id_to_obj[d]["name"] == "Model":              cn_connects.append((s, d, pr))
                elif id_to_obj[d]["name"] == "AnimationCurveNode": curve_connects.append((s, d, pr))

    bones = {}
    for oid in id_to_obj:
        if id_to_obj[oid]["name"] != "Model": continue
        obj  = id_to_obj[oid]
        name = obj["props"][1].split("\x00")[0]
        t = [0.0, 0.0, 0.0]; r = [0.0, 0.0, 0.0]
        for ch in obj["children"]:
            if ch["name"] == "Properties70":
                for p in ch["children"]:
                    if p["name"] != "P": continue
                    pp = p["props"]
                    if   pp[0] == "Lcl Translation": t = [pp[4], pp[5], pp[6]]
                    elif pp[0] == "Lcl Rotation":    r = [pp[4], pp[5], pp[6]]
        bones[oid] = {"name": name, "lcl_t": t, "lcl_r": r, "parent": parent_of.get(oid)}

    bone_cn   = {}; [bone_cn.setdefault(d, {}).__setitem__(pr, s) for s, d, pr in cn_connects]
    cn_curves = {}; [cn_curves.setdefault(d, {}).__setitem__(pr, s) for s, d, pr in curve_connects]

    def curve_kv(cid): #läser en animation curve, dvs en sekvens av keyframes, och returnerar dess keyframe-tider och värden
        if cid is None or cid not in id_to_obj: return None, None
        t = v = None
        for c in id_to_obj[cid]["children"]:
            if c["name"] == "KeyTime":         t = c["props"][0] if c["props"] else []
            elif c["name"] == "KeyValueFloat": v = c["props"][0] if c["props"] else []
        return np.array(t, dtype=np.int64), np.array(v, dtype=np.float64)

    all_times = set()
    for bid, cns in bone_cn.items():
        for pr, cnid in cns.items():
            for ch, cid in cn_curves.get(cnid, {}).items():
                t, _ = curve_kv(cid)
                if t is not None: all_times.update(t.tolist())
    times = sorted(all_times); N = len(times); t2i = {t: i for i, t in enumerate(times)}

    def bone_channel_arr(bid, prop): #skapar en array av positioner eller rotationer över tid för ett givet ben och egenskap (translation eller rotation)
        rest = np.array(bones[bid]["lcl_t" if prop == "Lcl Translation" else "lcl_r"])

        arr  = np.tile(rest, (N, 1))
        cnid = bone_cn.get(bid, {}).get(prop)
        if cnid is None: return arr
        for ch, axis in [("d|X", 0), ("d|Y", 1), ("d|Z", 2)]:
            cid = cn_curves.get(cnid, {}).get(ch)
            if cid is None: continue
            t, v = curve_kv(cid)
            if t is None: continue
            for ti, vi in zip(t, v):
                i = t2i.get(int(ti))
                if i is not None: arr[i, axis] = vi
        return arr

    #framåt kinematik:
    local_T = {}
    for bid in bones:
        t = bone_channel_arr(bid, "Lcl Translation")
        r = bone_channel_arr(bid, "Lcl Rotation")
        R = _euler_xyz_to_matrix(r)
        M = np.zeros((N, 4, 4)); M[:, :3, :3] = R; M[:, :3, 3] = t; M[:, 3, 3] = 1
        local_T[bid] = M

    world_T = {}
    def world_of(bid):
        if bid in world_T: return world_T[bid]
        parent = bones[bid]["parent"]
        world_T[bid] = (local_T[bid] if parent is None or parent not in local_T
                        else np.einsum("nij,njk->nik", world_of(parent), local_T[bid]))
        return world_T[bid]

    name_to_id = {b["name"]: oid for oid, b in bones.items()}
    positions  = {}
    for bname in target_bone_names:
        if bname not in name_to_id:
            avail = list(name_to_id.keys())
            raise RuntimeError(
                f"Ben '{bname}' saknas i {fbx_path.name}.\n"
                f"Tillgangliga ben (forsta 20): {avail[:20]}")
        W = world_of(name_to_id[bname])
        positions[bname] = W[:, :3, 3]
    return positions, N


def rigged_knee_angles(joints, bones_def): #beräknar knävinkeln över tid baserat på led-positioner från en riggad källa (QuickMagic eller Mimem), returnerar en dict med vinklar för vänster och höger knä
    out = {}
    for side, b in bones_def.items():
        out[side] = angle_between(
            joints[b["hip"]] - joints[b["knee"]],
            joints[b["ankle"]] - joints[b["knee"]])
    return out


# mmpose / vitpose — human3.6m .npy-format 
def load_h36m_knee_angles(npy_path): #laddar en .npy-fil i human3.6m-format och beräknar knävinkeln över tid, returnerar en dict med vinklar för vänster och höger knä och antal frames
    data = np.load(npy_path)
    if data.ndim != 3 or data.shape[1] < 7 or data.shape[2] != 3:
        raise RuntimeError(f"Forvantar (N,17,3), fick {data.shape}")
    out = {}
    for side, idxs in H36M_INDICES.items():
        hip   = data[:, idxs["hip"]]
        knee  = data[:, idxs["knee"]]
        ankle = data[:, idxs["ankle"]]
        out[side] = angle_between(hip - knee, ankle - knee)
    return out, data.shape[0]


# open mocap, csv
def load_open_mocap_csv(csv_path): #laddar en csv-fil exporterad från blender med open mocap-data och extraherar marker-positioner över tid, returnerar en dict med positioner
    fps = None
    with open(csv_path, "r") as f:
        first = f.readline().strip()
        if first.startswith("#") and "fps=" in first:
            fps = float(first.split("fps=")[1].split(",")[0])
        reader = csv.reader(f)
        header = next(reader)
        rows   = [r for r in reader if r]

    col_idx = {}
    for i, col in enumerate(header[1:], start=1):
        if col.endswith("_x"):
            col_idx[col[:-2]] = (i, i + 1, i + 2)

    positions = {}
    for name, (ix, iy, iz) in col_idx.items():
        positions[name] = np.array(
            [[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows])

    if fps is None:
        fps = 30.0
        print(f"  Varning: kunde inte lasa fps, antar {fps}.")
    return positions, fps


def open_mocap_knee_angles(positions): #beräknar knävinkeln över tid baserat på marker-positioner från open mocap, returnerar en dict med vinklar för vänster och höger knä
    out = {}
    for side, idxs in OPENMOCAP_INDICES.items():
        hip   = positions[idxs["hip"]]
        knee  = positions[idxs["knee"]]
        ankle = positions[idxs["ankle"]]
        out[side] = angle_between(hip - knee, ankle - knee)
    return out


# huvudprogram
def main(): #
    # qualisys
    qmarkers, n_q = load_qualisys(QUALISYS_PATH)
    qang = qualisys_knee_angles(qmarkers)

    # quickmagic
    qm_bone_names = [b for sides in QUICKMAGIC_BONES.values() for b in sides.values()]
    qm_joints, n_qm = load_rigged_joint_positions(QUICKMAGIC_PATH, qm_bone_names)
    qm_angles = rigged_knee_angles(qm_joints, QUICKMAGIC_BONES)

    # mimem
    mm_bone_names = [b for sides in MIMEM_BONES.values() for b in sides.values()]
    mm_joints, n_mm = load_rigged_joint_positions(MIMEM_PATH, mm_bone_names)
    mm_angles = rigged_knee_angles(mm_joints, MIMEM_BONES)

    # mediapipe
    op_pos, op_fps = load_open_mocap_csv(OPENMOCAP_CSV)
    op_angles = open_mocap_knee_angles(op_pos)
    n_op = next(iter(op_pos.values())).shape[0]


    
    pose_data = {}
    for label, path, fps, color, linestyle in POSE_ESTIMATORS:
        print(f"Laser {label} ...")
        angles, n = load_h36m_knee_angles(path)
        print(f"  {n} frames @ {fps} fps  ({n/fps:.2f} s)")
        pose_data[label] = {"angles": angles, "fps": fps, "color": color,
                            "linestyle": linestyle, "n": n}

   #interpolera alla system till Qualisys-tidsbasen
    t_q = np.arange(n_q) / QUALISYS_FPS

    shift_qm = 183 / QUALISYS_FPS
    shift_op = 183 / QUALISYS_FPS
    shift_mm = 183 / QUALISYS_FPS
    shifts = {label: 183 / QUALISYS_FPS for label in pose_data}

    aligned = {"L": {}, "R": {}}
    for side in ("L", "R"):
        aligned[side]["QuickMagic"] = resample_to_fps(qm_angles[side], QUICKMAGIC_FPS, t_q, shift_qm)
        aligned[side]["Mimem"]      = resample_to_fps(mm_angles[side], MIMEM_FPS,      t_q, shift_mm)
        aligned[side]["MediaPipe"]  = resample_to_fps(op_angles[side], op_fps,         t_q, shift_op)
        for label, info in pose_data.items():
            aligned[side][label] = resample_to_fps(info["angles"][side], info["fps"], t_q, shifts[label])
    if STATS_INTERVAL is not None:
        t_lo, t_hi    = STATS_INTERVAL
        interval_mask = (t_q >= t_lo) & (t_q <= t_hi)
    else:
        interval_mask = np.ones(n_q, dtype=bool)

    def stats(ref, test, mask):
        m = mask & ~np.isnan(test) & ~np.isnan(ref)
        d = ref[m] - test[m]
        if d.size == 0:
            return {"n": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan}
        return {"n": int(m.sum()), "bias": d.mean(),
                "mae": np.abs(d).mean(), "rmse": np.sqrt((d**2).mean())}

    SYSTEM_NAMES = ["QuickMagic", "Mimem", "MediaPipe"] + [l for l, *_ in POSE_ESTIMATORS]

    all_stats = {}
    for side in ("R", "L"):
        all_stats[side] = {}
        for sys_name in SYSTEM_NAMES:
            s = stats(qang[side], aligned[side][sys_name], interval_mask)
            all_stats[side][sys_name] = s

    MARKER_EVERY = 25

    STYLES = {
        "QuickMagic": dict(color="#648FFF", ls="--",  lw=1.5, marker="o", ms=5, mew=1.2, label="QuickMagic"),
        "Mimem":      dict(color="#DC267F", ls="-.",  lw=1.5, marker="s", ms=5, mew=1.2, label="Mimem"),
        "MediaPipe":              dict(color="#FE6100", ls="--",  lw=1.3, marker="^", ms=5,  mew=1.0, label="MediaPipe"),
        "MMPose+MotionBERT":      dict(color="#7B2D8B", ls="-.",  lw=1.3, marker="D", ms=4,  mew=1.0, label="MMPose+MotionBERT"),
        "MMPose+MotionAGFormer":  dict(color="#009988", ls=":",   lw=1.3, marker="d", ms=4,  mew=1.0, label="MMPose+MotionAGFormer"),
        "ViTPose+MotionBERT":     dict(color="#CC3311", ls="-.",  lw=1.3, marker="v", ms=5,  mew=1.0, label="ViTPose+MotionBERT"),
        "ViTPose+MotionAGFormer": dict(color="#0077BB", ls="--",   lw=1.3, marker="P", ms=5,  mew=1.0, label="ViTPose+MotionAGFormer"),
    }
    QUALISYS_STYLE = dict(color="#111111", ls="-", lw=1.8, alpha=0.8,
                          label="Qualisys", zorder=1)

    if STATS_INTERVAL is not None:
        x_lo, x_hi = STATS_INTERVAL
    else:
        video_duration = len(list(pose_data.values())[0]["angles"]["R"]) / list(pose_data.values())[0]["fps"]
        x_lo = shift_qm - 0.15
        x_hi = shift_qm + video_duration + 0.15

    def plot_with_markers(ax, t, y, style, every=MARKER_EVERY):
        st = dict(style)
        marker = st.pop("marker", None); ms = st.pop("ms", 5)
        mew = st.pop("mew", 1.0); label = st.pop("label", "")
        ax.plot(t, y, **st, zorder=5, label="_nolegend_")
        mask = np.arange(len(t)) % every == 0
        t_m = t[mask]; y_m = np.where(np.isnan(y[mask]), np.nan, y[mask])
        ax.plot(t_m, y_m, linestyle="none", marker=marker, markersize=ms,
                color=st["color"], markeredgewidth=mew, markerfacecolor=st["color"],
                alpha=0.85, zorder=6, label=label)

    for side in ("R", "L"):
        side_label = "Höger" if side == "R" else "Vänster"
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        mask_x = (t_q >= x_lo) & (t_q <= x_hi)
        t_clip = t_q[mask_x]

        def clip(arr):
            return arr[mask_x]

        ax = axes[0]
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 185)
        ax.set_autoscale_on(False)
        ax.plot(t_clip, clip(qang[side]), **QUALISYS_STYLE)
        for sys in ("Mimem", "QuickMagic"):
            plot_with_markers(ax, t_clip, clip(aligned[side][sys]), STYLES[sys])
        ax.axhline(180, color="k", lw=0.5, alpha=0.25, ls=":")
        ax.set_ylabel("Knävinkel (grader)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", fontsize=13, framealpha=0.5, ncol=3)
        ax.set_title("Kommersiella system", fontsize=15)

        ax = axes[1]
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0, 185)
        ax.set_autoscale_on(False)
        ax.plot(t_clip, clip(qang[side]), **QUALISYS_STYLE)
        open_systems = ["MediaPipe", "MMPose+MotionBERT", "MMPose+MotionAGFormer",
                        "ViTPose+MotionBERT", "ViTPose+MotionAGFormer"]
        for sys in open_systems:
            plot_with_markers(ax, t_clip, clip(aligned[side][sys]), STYLES[sys])
        ax.axhline(180, color="k", lw=0.5, alpha=0.25, ls=":")
        ax.set_ylabel("Knävinkel (grader)", fontsize=15)
        ax.set_xlabel("Tid (s)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower center", fontsize=13, framealpha=0.5, ncol=3)
        ax.set_title("Öppen källkod-system", fontsize=15)

        import matplotlib.ticker as mticker
        for ax in axes:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0, 185)
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda val, pos: f"{val - x_lo:.1f}"))

        out = Path(__file__).parent / f"indianhopp_{side_label.lower()}_kna.png"
        plt.savefig(out, dpi=120)

    plt.show()


main()