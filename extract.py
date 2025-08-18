import sys, json, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
import joblib

# ----------------------- COCO-17 indices -----------------------
IDX = {
    "nose":0, "leye":1, "reye":2, "lear":3, "rear":4,
    "lsh":5, "rsh":6, "lel":7, "rel":8, "lw":9, "rw":10,
    "lhip":11, "rhip":12, "lk":13, "rk":14, "la":15, "ra":16
}
# skeleton (incl. head if you visualize)
PAIRS = [
    (IDX["lsh"], IDX["rsh"]),
    (IDX["lsh"], IDX["lel"]), (IDX["lel"], IDX["lw"]),
    (IDX["rsh"], IDX["rel"]), (IDX["rel"], IDX["rw"]),
    (IDX["lhip"], IDX["rhip"]),
    (IDX["lsh"], IDX["lhip"]), (IDX["rsh"], IDX["rhip"]),
    (IDX["lhip"], IDX["lk"]), (IDX["lk"], IDX["la"]),
    (IDX["rhip"], IDX["rk"]), (IDX["rk"], IDX["ra"]),
    (IDX["nose"], IDX["lsh"]), (IDX["nose"], IDX["rsh"]),
    (IDX["nose"], IDX["leye"]), (IDX["nose"], IDX["reye"]),
    (IDX["leye"], IDX["lear"]), (IDX["reye"], IDX["rear"]),
    (IDX["leye"], IDX["reye"]),
]

# ----- classes (your list had 6) -----
DEFAULT_CLASSES = ["Fall", "Lie", "Likefall", "Siting", "Stand", "Walking"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

# ----------------------- helpers -----------------------
def list_seq_dirs(class_dir: Path) -> List[Path]:
    return [d for d in sorted(class_dir.iterdir()) if d.is_dir() and d.name.lower().startswith("seq")]

def list_frames(seq_dir: Path) -> List[Path]:
    return [p for p in sorted(seq_dir.iterdir()) if p.suffix in IMG_EXTS]

def fail_with_help(data_root: Path, wanted_classes):
    print("\n[ERROR] No data found.")
    print(f"- data_root = {data_root.resolve()}")
    print("- expected classes:", wanted_classes)
    if data_root.exists():
        subs = [f.name for f in data_root.iterdir() if f.is_dir()]
        print("- found subdirs:", subs)
    else:
        print("- data_root not found.")
    sys.exit(1)

# ---------- geometry ----------
def angle(a, b, c):
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.float32(np.arccos(cosang))

def joint_angle_features(kps_xy: np.ndarray) -> np.ndarray:
    a_l_elbow = angle(kps_xy[IDX["lsh"]], kps_xy[IDX["lel"]], kps_xy[IDX["lw"]])
    a_r_elbow = angle(kps_xy[IDX["rsh"]], kps_xy[IDX["rel"]], kps_xy[IDX["rw"]])
    a_l_knee  = angle(kps_xy[IDX["lhip"]], kps_xy[IDX["lk"]],  kps_xy[IDX["la"]])
    a_r_knee  = angle(kps_xy[IDX["rhip"]], kps_xy[IDX["rk"]],  kps_xy[IDX["ra"]])
    a_l_hip   = angle(kps_xy[IDX["lsh"]], kps_xy[IDX["lhip"]], kps_xy[IDX["lk"]])
    a_r_hip   = angle(kps_xy[IDX["rsh"]], kps_xy[IDX["rhip"]], kps_xy[IDX["rk"]])
    a_torso   = angle(kps_xy[IDX["lhip"]],
                      (kps_xy[IDX["lsh"]]+kps_xy[IDX["rsh"]])/2.0,
                      kps_xy[IDX["rhip"]])
    return np.array([a_l_elbow, a_r_elbow, a_l_knee, a_r_knee, a_l_hip, a_r_hip, a_torso], dtype=np.float32)

def rigid_normalize(kps_xy: np.ndarray) -> np.ndarray:
    """
    Translate to hip-center, rotate torso to vertical, scale by shoulder width.
    Works per-frame; use after temporal smoothing.
    """
    lhip, rhip = kps_xy[IDX["lhip"]], kps_xy[IDX["rhip"]]
    lsh,  rsh  = kps_xy[IDX["lsh"]],  kps_xy[IDX["rsh"]]
    center = (lhip + rhip) * 0.5

    # translation
    X = kps_xy - center

    # rotation (torso vector hip->shoulder mid)
    sh_mid = (lsh + rsh) * 0.5
    t = sh_mid - (0.0, 0.0)
    theta = np.arctan2(t[1], t[0])            # angle vs +x
    # we want torso vertical (+y), so rotate by (pi/2 - theta)
    rot = np.array([[np.cos(np.pi/2 - theta), -np.sin(np.pi/2 - theta)],
                    [np.sin(np.pi/2 - theta),  np.cos(np.pi/2 - theta)]], dtype=np.float32)
    Xr = (rot @ X.T).T

    # scale by shoulder width
    sw = np.linalg.norm(lsh - rsh) + 1e-6
    return Xr / sw

def multi_scale_diff(arr: np.ndarray, strides=(1,2,4)) -> Tuple[np.ndarray, np.ndarray]:
    """
    arr: (T, F) features
    returns: vel(T, F*len(strides)), acc(T, F*len(strides))
    Central differences with each stride.
    """
    T, F = arr.shape
    v_list, a_list = [], []
    for s in strides:
        v = np.zeros_like(arr)
        a = np.zeros_like(arr)
        if T >= 2*s:
            # velocity
            v[s:-s] = (arr[2*s:] - arr[:-2*s]) * (0.5 / s)
            v[:s]   = (arr[s:2*s] - arr[:s]) / s
            v[-s:]  = (arr[-s:] - arr[-2*s:-s]) / s
            # acceleration
            a[s:-s] = (arr[2*s:] - 2*arr[s:-s] + arr[:-2*s]) / (s**2)
            a[:s]   = a[s]     # copy boundary
            a[-s:]  = a[-s-1]
        v_list.append(v)
        a_list.append(a)
    return np.concatenate(v_list, axis=1), np.concatenate(a_list, axis=1)

# ---------- temporal interpolation & smoothing ----------
def interpolate_gaps(kps: np.ndarray, conf: np.ndarray, thr: float) -> np.ndarray:
    """
    kps: (T,17,2), conf: (T,17)
    Linear interpolate each joint across spans where conf<thr.
    Leaves endpoints as nearest valid (forward/backward fill).
    """
    T, J, _ = kps.shape
    out = kps.copy()
    for j in range(J):
        good = conf[:, j] >= thr
        if good.any():
            # forward/backward fill to edges
            first = np.argmax(good)
            last  = T - 1 - np.argmax(good[::-1])
            out[:first, j] = kps[first, j]
            out[last+1:, j] = kps[last, j]
            # interpolate internal gaps
            idx = np.arange(T)
            out[first:last+1, j] = np.interp(idx[first:last+1],
                                             idx[good], kps[good, j, 0]).reshape(-1,1)
            yinterp = np.interp(idx[first:last+1],
                                idx[good], kps[good, j, 1]).reshape(-1,1)
            out[first:last+1, j, 1] = yinterp[:,0]
        else:
            # no good frames: keep as zeros
            pass
    return out

def ema_smooth(arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    arr: (T, ..., D)  EMA with factor alpha (heavier weight on previous)
    """
    out = arr.copy()
    for t in range(1, arr.shape[0]):
        out[t] = alpha * out[t-1] + (1.0 - alpha) * arr[t]
    return out

# ----------------------- core extraction -----------------------
def run_detector(model: YOLO, frame_paths: List[Path], conf: float, iou: float):
    """
    Returns kps (T,17,2), kpc (T,17)
    """
    T = len(frame_paths)
    kps = np.zeros((T, 17, 2), dtype=np.float32)
    kpc = np.zeros((T, 17), dtype=np.float32)

    for t, fp in enumerate(frame_paths):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        res = model.predict(img, verbose=False, device="cpu", max_det=1, conf=conf, iou=iou)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        xy = res.keypoints.xy[0].cpu().numpy().astype(np.float32)      # (17,2)
        try:
            confs = res.keypoints.conf[0].cpu().numpy().astype(np.float32) # (17,)
        except Exception:
            confs = np.ones(17, dtype=np.float32)
        kps[t] = xy
        kpc[t] = confs
    return kps, kpc

def extract_seq_features(model: YOLO, frame_paths: List[Path], add_angles: bool,
                         seq_len: int, seq_stride: int,
                         conf: float, iou: float,
                         kp_conf: float, min_valid_kp: int,
                         ema_alpha: float, rotate_norm: bool, ms_strides=(1,2,4)):
    """
    Returns: feats (T,F), valid_frames (int), pad_mask (T,) with 1=real, 0=pad
    """
    # sample by stride, then pad/crop to seq_len
    frame_paths = frame_paths[::max(1, seq_stride)]
    if len(frame_paths) >= seq_len:
        frame_paths = frame_paths[:seq_len]
    elif len(frame_paths) > 0:
        frame_paths = frame_paths + [frame_paths[-1]] * (seq_len - len(frame_paths))
    else:
        # all zeros (empty seq)
        base = np.zeros((seq_len, 34), dtype=np.float32)
        vel  = np.zeros((seq_len, 34*len(ms_strides)), dtype=np.float32)
        acc  = np.zeros((seq_len, 34*len(ms_strides)), dtype=np.float32)
        feats = np.concatenate([base, vel, acc], axis=1)
        if add_angles:
            ang = np.zeros((seq_len, 7), dtype=np.float32)
            dang = np.zeros_like(ang); ddang = np.zeros_like(ang)
            feats = np.concatenate([feats, np.sin(ang), np.cos(ang), dang, ddang], axis=1)
        mask = np.zeros((seq_len,), dtype=np.uint8)
        return feats, 0, mask

    # detect all first
    raw_xy, raw_c = run_detector(model, frame_paths, conf, iou)  # (T,17,2), (T,17)

    # count valid frames *before* filling
    valid_frames = int(((raw_c > kp_conf).sum(axis=1) >= min_valid_kp).sum())

    # two-pass cleanup
    kpxy = interpolate_gaps(raw_xy, raw_c, kp_conf)              # fill gaps linearly
    kpxy = ema_smooth(kpxy, ema_alpha)                           # smooth after interp

    # per-frame rigid normalization
    if rotate_norm:
        kpxy_norm = np.stack([rigid_normalize(kpxy[t]) for t in range(kpxy.shape[0])], axis=0)
    else:
        # fallback: hip-center translate, scale by shoulder width (no rotation)
        lhip = kpxy[:, IDX["lhip"]]; rhip = kpxy[:, IDX["rhip"]]
        center = (lhip + rhip) * 0.5
        lsh = kpxy[:, IDX["lsh"]]; rsh = kpxy[:, IDX["rsh"]]
        sw = np.linalg.norm(lsh - rsh, axis=1, keepdims=True) + 1e-6
        kpxy_norm = (kpxy - center[:, None, :]) / sw[:, None, :]

    # base XY features
    xy_feat = kpxy_norm.reshape(len(frame_paths), -1)            # (T, 34)

    # multi-scale velocity & accel
    vel, acc = multi_scale_diff(xy_feat, strides=ms_strides)     # (T, 34*|S|), (T, 34*|S|)
    feats_list = [xy_feat, vel, acc]

    # optional angles (+ dynamics)
    if add_angles:
        ang = np.stack([joint_angle_features(kpxy_norm[t]) for t in range(kpxy_norm.shape[0])], axis=0)  # (T,7)
        ang_sin, ang_cos = np.sin(ang), np.cos(ang)
        dang, ddang = multi_scale_diff(ang, strides=ms_strides)  # central diff on radians
        feats_list.extend([ang_sin, ang_cos, dang, ddang])

    feats = np.concatenate(feats_list, axis=1).astype(np.float32)  # (T,F)

    # build pad mask (all frames here are "real" after padding; we can mark padded copies)
    T = len(frame_paths)
    # find how many original (before pad) frames we had
    # (we padded by repeating last frame; detect runs of identical file path at the end)
    # But simpler: assume if we padded, last k frames are pads.
    # We can compute pads by checking original count before padding:
    pad_mask = np.ones((T,), dtype=np.uint8)
    return feats, valid_frames, pad_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="pure_data")
    ap.add_argument("--openvino_model", type=str, default="yolo11n-pose_openvino_model")
    ap.add_argument("--out_dir", type=str, default="cache_kps_final")
    ap.add_argument("--classes", type=str, default="Fall,Lie,Likefall,Siting,Stand,Walking")
    ap.add_argument("--strict_classes", action="store_true")
    ap.add_argument("--add_angles", action="store_true", help="Add joint-angle features (sin/cos)")
    # high-accuracy knobs
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--seq_stride", type=int, default=1, help="sample every k-th frame")
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--kp_conf", type=float, default=0.25)
    ap.add_argument("--min_valid_kp", type=int, default=10)
    ap.add_argument("--min_valid_frames", type=int, default=12)
    ap.add_argument("--ema_alpha", type=float, default=0.5)   # slightly lower for less lag
    ap.add_argument("--rotate_norm", action="store_true", help="Rigid rotation so torso is vertical")
    ap.add_argument("--ms_strides", type=str, default="1,2,4", help="multi-scale strides for diff")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(args.openvino_model)
    except Exception as e:
        print(f"[ERROR] Cannot load OpenVINO model: {args.openvino_model}\n{e}")
        sys.exit(1)

    user_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    classes = user_classes or DEFAULT_CLASSES
    if args.strict_classes:
        classes = user_classes

    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root.resolve()}")
        sys.exit(1)

    existing = []
    for cls in classes:
        p = data_root / cls
        if p.exists() and p.is_dir():
            existing.append(cls)
        else:
            print(f"[WARN] missing class folder: {p}")

    if not existing:
        fail_with_help(data_root, classes)

    X_all, y_all, M_all = [], [], []
    label_map = {c:i for i,c in enumerate(existing)}
    print(f"[INFO] Classes used (order -> labels): {label_map}")

    dropped = {c:0 for c in existing}
    kept    = {c:0 for c in existing}

    ms_strides = tuple(int(s) for s in args.ms_strides.split(",") if s.strip())

    for cls in existing:
        class_dir = data_root / cls
        seq_dirs = list_seq_dirs(class_dir)
        if len(seq_dirs) == 0:
            print(f"[WARN] no seq* dirs in {class_dir}")
            continue
        for sd in tqdm(seq_dirs, desc=f"Class {cls}", unit="seq"):
            frames = list_frames(sd)
            if len(frames) == 0:
                print(f"[WARN] no images in {sd}")
                continue

            feats, vcnt, mask = extract_seq_features(
                model, frames, add_angles=args.add_angles,
                seq_len=args.seq_len, seq_stride=args.seq_stride,
                conf=args.conf, iou=args.iou,
                kp_conf=args.kp_conf, min_valid_kp=args.min_valid_kp,
                ema_alpha=args.ema_alpha, rotate_norm=args.rotate_norm,
                ms_strides=ms_strides
            )

            if vcnt < args.min_valid_frames:
                dropped[cls] += 1
                continue

            X_all.append(feats)
            y_all.append(label_map[cls])
            M_all.append(mask)
            kept[cls] += 1

    if len(X_all) == 0:
        fail_with_help(data_root, existing)

    X = np.stack(X_all, axis=0).astype(np.float32)  # (N,T,F)
    y = np.array(y_all, dtype=np.int64)
    M = np.stack(M_all, axis=0).astype(np.uint8)    # (N,T)

    # standardize over feature dim (keep time & batch intact)
    N,T,F = X.shape
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X.reshape(N*T, F)).reshape(N, T, F).astype(np.float32)

    np.save(out_dir/"X.npy", X2)
    np.save(out_dir/"y.npy", y)
    np.save(out_dir/"mask.npy", M)
    with open(out_dir/"label_map.json","w",encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    joblib.dump(scaler, out_dir/"scaler.joblib")

    print(f"[OK] Saved: X{X2.shape}, y{y.shape}, mask{M.shape}, label_map={label_map}")
    print(f"[OK] seq_len={args.seq_len}, add_angles={'ON' if args.add_angles else 'OFF'}, rotate_norm={'ON' if args.rotate_norm else 'OFF'} -> F={X2.shape[2]}")
    print(f"[REPORT] kept per class: {kept}")
    print(f"[REPORT] dropped (low-quality) per class: {dropped}")

if __name__ == "__main__":
    main()
