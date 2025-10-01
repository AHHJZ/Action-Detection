import time, argparse
from pathlib import Path
from collections import deque
import numpy as np
import cv2
import torch
from torch import nn
from ultralytics import YOLO
import joblib

# ---------------- COCO indices & pairs ----------------
IDX = {
    "nose":0, "leye":1, "reye":2, "lear":3, "rear":4,
    "lsh":5, "rsh":6, "lel":7, "rel":8, "lw":9, "rw":10,
    "lhip":11, "rhip":12, "lk":13, "rk":14, "la":15, "ra":16
}
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

def normalize_by_torso(kps_xy: np.ndarray) -> np.ndarray:
    lhip, rhip = kps_xy[IDX["lhip"]], kps_xy[IDX["rhip"]]
    lsh, rsh  = kps_xy[IDX["lsh"]],  kps_xy[IDX["rsh"]]
    center = (lhip + rhip) / 2.0
    scale  = np.linalg.norm(lsh - rsh) + 1e-6
    return (kps_xy - center) / scale

def angles_from_kps(kps_xy: np.ndarray) -> np.ndarray:
    def angle(a,b,c):
        v1=a-b; v2=c-b
        n1=np.linalg.norm(v1)+1e-9; n2=np.linalg.norm(v2)+1e-9
        cosang = np.dot(v1,v2)/(n1*n2); cosang = np.clip(cosang, -1.0, 1.0)
        return np.float32(np.arccos(cosang))
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

def _multi_scale_diff(arr: np.ndarray, strides=(1,2,4)):
    T, F = arr.shape
    v_list, a_list = [], []
    for s in strides:
        v = np.zeros_like(arr); a = np.zeros_like(arr)
        if T >= 2*s:
            v[s:-s] = (arr[2*s:] - arr[:-2*s]) * (0.5 / s)
            v[:s]   = (arr[s:2*s] - arr[:s]) / s
            v[-s:]  = (arr[-s:] - arr[-2*s:-s]) / s
            a[s:-s] = (arr[2*s:] - 2*arr[s:-s] + arr[:-2*s]) / (s**2)
            a[:s]   = a[s]; a[-s:]  = a[-s-1]
        v_list.append(v); a_list.append(a)
    return np.concatenate(v_list, axis=1), np.concatenate(a_list, axis=1)

def build_features_from_buffers(buf_xy, buf_ang_sin, buf_ang_cos, use_angles: bool, ms_strides=(1,2,4)):
    xy = np.stack(list(buf_xy), axis=0)  # (t, 34)
    vel_xy, acc_xy = _multi_scale_diff(xy, strides=ms_strides)
    parts = [xy, vel_xy, acc_xy]
    if use_angles:
        ang_sin = np.stack(list(buf_ang_sin), axis=0)    # (t,7)
        ang_cos = np.stack(list(buf_ang_cos), axis=0)    # (t,7)
        ang = np.arctan2(ang_sin, ang_cos)               # [-pi,pi], (t,7)
        d_ang, dd_ang = _multi_scale_diff(ang, strides=ms_strides)
        parts.extend([ang_sin, ang_cos, d_ang, dd_ang])
    feats = np.concatenate(parts, axis=1).astype(np.float32)
    return feats

# ---------------- viz ----------------
def draw_bbox_and_pose(img, bbox_xyxy, kps_xy, kps_conf=None, kp_thresh=0.5, label_txt=None):
    x1,y1,x2,y2 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    for i, (x,y) in enumerate(kps_xy.astype(int)):
        if (kps_conf is None) or (float(kps_conf[i]) >= kp_thresh):
            cv2.circle(img, (x,y), 3, (0,255,255), -1)
    for a,b in PAIRS:
        ok = True
        if kps_conf is not None:
            ok = (float(kps_conf[a]) >= kp_thresh) and (float(kps_conf[b]) >= kp_thresh)
        if ok:
            xa,ya = kps_xy[a].astype(int); xb,yb = kps_xy[b].astype(int)
            cv2.line(img, (xa,ya), (xb,yb), (255,0,0), 2)
    if label_txt:
        cv2.rectangle(img, (x1, max(0, y1-28)), (x1+max(120, len(label_txt)*9), y1), (0,255,0), -1)
        cv2.putText(img, label_txt, (x1+5, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10,10,10), 2, cv2.LINE_AA)
    return img

# ---------------- IoU tracker ----------------
def iou_xyxy(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def assign_detections_to_tracks(dets, tracks, iou_thresh=0.3):
    matches = []; unmatched_d = list(range(len(dets))); unmatched_t = list(range(len(tracks)))
    if len(dets)==0 or len(tracks)==0:
        return matches, unmatched_d, unmatched_t
    iou_mat = np.zeros((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            iou_mat[i,j] = iou_xyxy(d, t.bbox)
    while True:
        i,j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i,j] < iou_thresh: break
        matches.append((i,j)); iou_mat[i,:] = -1; iou_mat[:,j] = -1
        if i in unmatched_d: unmatched_d.remove(i)
        if j in unmatched_t: unmatched_t.remove(j)
    return matches, unmatched_d, unmatched_t

# ---------------- TCN ----------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size=int(chomp_size)
    def forward(self,x): return x[:,:, :-self.chomp_size].contiguous() if self.chomp_size>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size-1)*dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x): out=self.net(x); return out + self.down(x)

class TCN(nn.Module):
    def __init__(self, in_feat, channels, num_classes, kernel_size=5, dropout=0.2):
        super().__init__()
        layers=[]; prev=in_feat
        for i,ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation=2**i, dropout=dropout)); prev=ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout), nn.Linear(channels[-1], num_classes)
        )
    def forward(self, x):  # x: (B, T, F)
        x=x.transpose(1,2)      # (B,F,T)
        feats=self.tcn(x)       # (B,C,T)
        return self.head(feats) # (B,num_classes)

def build_classifier(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_map = ckpt["label_map"]; inv_label = {v:k for k,v in label_map.items()}
    model = TCN(in_feat=ckpt["in_feat"], channels=ckpt["channels"],
                num_classes=len(label_map),
                kernel_size=ckpt.get("kernel_size",5),
                dropout=ckpt.get("dropout",0.2))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    scaler = joblib.load(ckpt["scaler_path"])
    return model, scaler, inv_label

def joint_from_flat(x34, idx):
    xy = x34.reshape(17,2); return xy[idx]

def gait_osc_score(buf_xy, win=10):
    if len(buf_xy) < 3: return 0.0
    xs = list(buf_xy)[-win:]
    L = np.array([joint_from_flat(x, IDX["la"]) for x in xs])
    R = np.array([joint_from_flat(x, IDX["ra"]) for x in xs])
    d = np.linalg.norm(L - R, axis=1)
    if d.mean() <= 1e-6: return 0.0
    return float(np.clip(d.std() / (d.mean()+1e-6), 0.0, 1.0))

def knee_flexion_score(kps_norm):
    def ang(a,b,c):
        v1=a-b; v2=c-b
        n1=np.linalg.norm(v1)+1e-9; n2=np.linalg.norm(v2)+1e-9
        cosang = np.dot(v1,v2)/(n1*n2); cosang = np.clip(cosang,-1,1)
        return np.arccos(cosang)
    lk = ang(kps_norm[IDX["lhip"]], kps_norm[IDX["lk"]], kps_norm[IDX["la"]])
    rk = ang(kps_norm[IDX["rhip"]], kps_norm[IDX["rk"]], kps_norm[IDX["ra"]])
    s = (np.pi - (lk+rk)/2)/np.pi
    return float(np.clip(s, 0, 1))

def hip_height_score(buf_hipy_norm, win=6):
    if len(buf_hipy_norm)==0: return 0.5
    vals = np.array(list(buf_hipy_norm)[-win:])
    v = -vals.mean()
    return float(np.clip(0.5 + v*0.5, 0.0, 1.0))

def head_speed(buf_xy, win=6):
    if len(buf_xy) < 2: return 0.0
    xs = list(buf_xy)[-win:]
    sp=[]
    for a,b in zip(xs[:-1], xs[1:]):
        ha = joint_from_flat(a, IDX["nose"]); hb = joint_from_flat(b, IDX["nose"])
        sp.append(np.linalg.norm(hb - ha))
    return float(np.mean(sp)) if sp else 0.0

# ---------------- Track ----------------
class Track:
    _next_id = 1
    def __init__(self, bbox, kpxy, kpc, scaler, T, use_angles, ema_kps, kp_conf):
        self.id = Track._next_id; Track._next_id += 1
        self.bbox = bbox.astype(np.float32); self.last_seen = 0

        self.buf_xy = deque(maxlen=T)            # (34,)
        self.buf_ang_sin = deque(maxlen=T) if use_angles else None
        self.buf_ang_cos = deque(maxlen=T) if use_angles else None

        self.prev_ema = None; self.use_angles = use_angles
        self.ema_kps = ema_kps; self.kp_conf = kp_conf

        self.buf_hipy_norm = deque(maxlen=T)
        self.buf_tilt = deque(maxlen=T)

        self.scaler = scaler
        self.label = "…"; self.conf = 0.0
        self.ema_prob = None; self.ema_alpha_prob = 0.0
        self.fall_sticky = 0

        self._append_from_kps(kpxy, kpc)

    def _append_from_kps(self, kpxy, kpc):
        mask = (kpc >= self.kp_conf) if kpc is not None else np.ones(17, dtype=bool)
        if self.prev_ema is not None:
            kpxy_filled = np.where(mask[:, None], kpxy, self.prev_ema)
        else:
            center = (kpxy[IDX["lhip"]] + kpxy[IDX["rhip"]]) / 2.0
            kpxy_filled = np.where(mask[:, None], kpxy, center[None, :])
        ema_kps = kpxy_filled if self.prev_ema is None else self.ema_kps * self.prev_ema + (1.0 - self.ema_kps) * kpxy_filled
        self.prev_ema = ema_kps

        kpxy_norm = normalize_by_torso(ema_kps)
        self.buf_xy.append(kpxy_norm.reshape(-1).astype(np.float32))

        if self.use_angles:
            ang = angles_from_kps(kpxy_norm)
            self.buf_ang_sin.append(np.sin(ang).astype(np.float32))
            self.buf_ang_cos.append(np.cos(ang).astype(np.float32))

        lsh, rsh = ema_kps[IDX["lsh"]], ema_kps[IDX["rsh"]]
        lhip, rhip = ema_kps[IDX["lhip"]], ema_kps[IDX["rhip"]]
        sw = np.linalg.norm(lsh - rsh) + 1e-6
        hip_mid = (lhip + rhip) / 2.0
        sh_mid  = (lsh + rsh) / 2.0
        self.buf_hipy_norm.append(float(hip_mid[1] / sw))
        v = sh_mid - hip_mid
        tilt = np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)  # 0=vertical
        self.buf_tilt.append(float(tilt))

    def update(self, bbox, kpxy, kpc):
        self.bbox = bbox.astype(np.float32)
        self._append_from_kps(kpxy, kpc)

    def smooth_prob(self, prob, adaptive=False):
        if self.ema_alpha_prob <= 0: return prob
        if self.ema_prob is None:
            self.ema_prob = prob
        else:
            alpha = self.ema_alpha_prob
            if adaptive:
                change = float(np.max(np.abs(prob - self.ema_prob)))
                alpha = np.clip(alpha - 0.5*change, 0.1, alpha)
            self.ema_prob = alpha * self.ema_prob + (1 - alpha) * prob
        return self.ema_prob

    def recent_motion_energy(self, win=5):
        if len(self.buf_xy) < 2: return 0.0
        xs = list(self.buf_xy)[-win:]
        diffs=[]
        for a,b in zip(xs[:-1], xs[1:]):
            da = a.reshape(17,2); db = b.reshape(17,2)
            diffs.append(np.mean(np.linalg.norm(db - da, axis=1)))
        return float(np.mean(diffs)) if diffs else 0.0

    def fall_trigger_sum(self, drop_thr, tilt_thr, win, sticky_max):
        if len(self.buf_hipy_norm) < 2: return False
        hip = np.array(list(self.buf_hipy_norm)[-win:], dtype=np.float32)
        tilt = np.array(list(self.buf_tilt)[-win:], dtype=np.float32)
        drop = np.maximum(np.diff(hip), 0).sum()
        tilt_up = np.maximum(np.diff(tilt), 0).sum()
        if (drop >= drop_thr) and (tilt_up >= tilt_thr):
            self.fall_sticky = sticky_max; return True
        return False

# ---- Fall confidence (combo) ----
def fall_confidence(track: Track, win=6):
    if len(track.buf_hipy_norm) < 2: return 0.0
    hip = np.array(list(track.buf_hipy_norm)[-win:], dtype=np.float32)
    tilt = np.array(list(track.buf_tilt)[-win:], dtype=np.float32)
    drop = np.maximum(np.diff(hip), 0).sum()
    tilt_up = np.maximum(np.diff(tilt), 0).sum()
    hsp = head_speed(track.buf_xy, win=win)
    s_drop = np.clip(drop/0.50, 0.0, 1.0)
    s_tilt = np.clip(tilt_up/0.50, 0.0, 1.0)
    s_head = np.clip(hsp/0.08, 0.0, 1.0)
    return 0.45*s_drop + 0.35*s_tilt + 0.20*s_head

# ---- Ensure features match the scaler ----
def ensure_feat_layout(feats_noang, feats_withang, scaler):
    if hasattr(scaler, "n_features_in_"):
        need = int(scaler.n_features_in_)
    elif hasattr(scaler, "mean_"):
        need = int(scaler.mean_.shape[0])
    else:
        need = feats_withang.shape[1]
    cand = {feats_noang.shape[1]: feats_noang,
            feats_withang.shape[1]: feats_withang}
    if need in cand: return cand[need]
    key = min(cand.keys(), key=lambda k: abs(k-need))
    return cand[key]

def safe_predict(clf, x_seq):
    try:
        with torch.no_grad():
            logits = clf(torch.from_numpy(x_seq))
            if logits is None: return None
            if not torch.isfinite(logits).all(): return None
            prob_t = torch.softmax(logits, dim=1)
            if not torch.isfinite(prob_t).all(): return None
            prob = prob_t.cpu().numpy()[0]
            if not np.isfinite(prob).all(): return None
            return prob
    except Exception:
        return None

# -------- SALUTE (improved, both hands + side view) --------
def is_salute_pose(kps_norm):
    angs = angles_from_kps(kps_norm)
    l_el, r_el = angs[0], angs[1]
    nose = kps_norm[IDX["nose"]]
    leye, reye = kps_norm[IDX["leye"]], kps_norm[IDX["reye"]]
    lear, rear = kps_norm[IDX["lear"]], kps_norm[IDX["rear"]]
    lsh, rsh = kps_norm[IDX["lsh"]], kps_norm[IDX["rsh"]]
    lw, rw   = kps_norm[IDX["lw"]],  kps_norm[IDX["rw"]]
    lel, rel = kps_norm[IDX["lel"]], kps_norm[IDX["rel"]]

    sh_w = np.linalg.norm(lsh - rsh) + 1e-6

    def have(pt): return np.isfinite(pt).all()

    
    if have(leye) and have(reye):
        forehead = (leye + reye) / 2.0
    else:
        forehead = nose + np.array([0.0, -0.15])

    
    temple_L = leye if have(leye) else (lear if have(lear) else forehead)
    temple_R = reye if have(reye) else (rear if have(rear) else forehead)

    
    MIN_ELBOW_DEG = 130.0
    Y_SLACK = 0.22
    D_FACE_MAIN = 0.95
    D_FACE_FALL = 1.15
    ANG_MIN, ANG_MAX = 10.0, 100.0

    dL = D_FACE_MAIN if (have(leye) or have(lear)) else D_FACE_FALL
    dR = D_FACE_MAIN if (have(reye) or have(rear)) else D_FACE_FALL

    def side_ok(sh, el, wr, elbow_angle, target, dface):
        if float(np.degrees(elbow_angle)) < MIN_ELBOW_DEG:
            return False
        if wr[1] > sh[1] + Y_SLACK:
            return False
        if np.linalg.norm(wr - target) > dface * sh_w:
            return False
        fore  = wr - el
        upper = el - sh
        if np.linalg.norm(fore) < 0.22 or np.linalg.norm(upper) < 0.22:
            return False
        ang = abs(float(np.degrees(np.arctan2(fore[1], fore[0]))))
        if not (ANG_MIN <= ang <= ANG_MAX):
            return False
        if el[1] > sh[1] + 0.12:
            return False
        return True

    ok_left  = side_ok(lsh, lel, lw, l_el, temple_L, dL)
    ok_right = side_ok(rsh, rel, rw, r_el, temple_R, dR)
    return ok_left or ok_right

def overlay_rgba(img_bgr, overlay_rgba, x, y):
    oh, ow = overlay_rgba.shape[:2]
    h, w = img_bgr.shape[:2]
    if x >= w or y >= h: return img_bgr
    x2 = min(x+ow, w); y2 = min(y+oh, h)
    ow = x2 - x; oh = y2 - y
    if ow <= 0 or oh <= 0: return img_bgr
    roi = img_bgr[y:y+oh, x:x+ow]
    if overlay_rgba.shape[2] == 4:  # PNG
        overlay = overlay_rgba[:oh,:ow,:3]
        alpha = overlay_rgba[:oh,:ow,3:4] / 255.0
        roi[:] = (alpha*overlay + (1-alpha)*roi).astype(np.uint8)
    else:
        roi[:] = overlay_rgba[:oh,:ow,:3]
    return img_bgr

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openvino_model", type=str, default="yolo11n-pose_openvino_model")
    ap.add_argument("--ckpt", type=str, default="runs_maxacc_final/tcn_best_final.pt")
    ap.add_argument("--cam_index", type=int, default=0)
    ap.add_argument("--video", type=str, default="sample.mp4")
    ap.add_argument("--webcam", action="store_true")

    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--min_seq_len", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--kp_conf", type=float, default=0.25)
    ap.add_argument("--vis_kp_conf", type=float, default=0.5)

    ap.add_argument("--draw_pose", action="store_true")
    ap.add_argument("--use_angles", action="store_true")
    ap.add_argument("--ema_kps", type=float, default=0.35)
    ap.add_argument("--ema", type=float, default=0.45)
    ap.add_argument("--adaptive_ema", action="store_true")
    ap.add_argument("--reject_thres", type=float, default=0.55)
    ap.add_argument("--max_det", type=int, default=6)
    ap.add_argument("--iou_thresh", type=float, default=0.4)
    ap.add_argument("--max_age", type=int, default=12)
    ap.add_argument("--ms_strides", type=str, default="1,2,4")

    ap.add_argument("--imgsz", type=int, default=640, help="input size for YOLO predict")
    ap.add_argument("--resize_width", type=int,default=832)
    ap.add_argument("--no_gui", action="store_true")

    # gait & separation thresholds
    ap.add_argument("--walk_speed_thr", type=float, default=0.018)
    ap.add_argument("--walk_speed_win", type=int, default=6)
    ap.add_argument("--osc_thr_walk_lo", type=float, default=0.10)

    # Sit vs Stand
    ap.add_argument("--knee_sit_thr", type=float, default=0.42)
    ap.add_argument("--hip_up_stand_thr", type=float, default=0.43)
    ap.add_argument("--sit_motion_guard", type=float, default=0.95)

    # Fall triggers
    ap.add_argument("--fall_drop_thr", type=float, default=0.28)
    ap.add_argument("--fall_tilt_thr", type=float, default=0.28)
    ap.add_argument("--fall_window", type=int, default=5)
    ap.add_argument("--fall_sticky", type=int, default=24)
    ap.add_argument("--fall_conf_thr", type=float, default=0.72)



    args = ap.parse_args()
    ms_strides = tuple(int(s) for s in args.ms_strides.split(",") if s.strip())

    # Source
    if args.webcam:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERR] Cannot open source."); return

    # Models
    pose_model = YOLO(args.openvino_model)
    clf, scaler, inv_label = build_classifier(Path(args.ckpt))
    stand_name = next((v for v in inv_label.values() if v.lower().startswith("stand")), "Stand")

    # salute image (optional)
    salute_rgba = None
    if args.salute_img:
        tmp = cv2.imread(args.salute_img, cv2.IMREAD_UNCHANGED)
        if tmp is not None:
            h, w = tmp.shape[:2]
            new_w = int(args.salute_size)
            new_h = int(h * (new_w / float(w))) if w>0 else h
            salute_rgba = cv2.resize(tmp, (new_w, new_h), interpolation=cv2.INTER_AREA)

    T = args.seq_len
    tracks = []

    # writer (lazy init)
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0; t_total = 0.0
    gui_enabled = not args.no_gui

    while True:
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()

        if args.resize_width and frame.shape[1] != args.resize_width:
            new_w = int(args.resize_width); new_h = int(frame.shape[0]*new_w/frame.shape[1])
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        res = pose_model.predict(frame, verbose=False, device="cpu",
                                 max_det=args.max_det, conf=args.conf, iou=args.iou,
                                 imgsz=args.imgsz)[0]

        vis = frame.copy()
        det_boxes, det_kps, det_kpc = [], [], []

        if res.keypoints is not None and len(res.keypoints) > 0:
            kps = res.keypoints.xy.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            try:
                kpc = res.keypoints.conf.cpu().numpy()
            except Exception:
                kpc = np.ones((len(boxes), 17), dtype=np.float32)
            for i in range(len(boxes)):
                if (kpc[i] > args.kp_conf).sum() >= 1:
                    det_boxes.append(boxes[i]); det_kps.append(kps[i]); det_kpc.append(kpc[i])

        if writer is None:
            hh, ww = vis.shape[:2]
            writer = cv2.VideoWriter("out_sample.mp4", fourcc, fps_in, (ww, hh))

        # track match
        matches, unmatched_det, unmatched_trk = assign_detections_to_tracks(det_boxes, tracks, iou_thresh=args.iou_thresh)
        for t_idx in unmatched_trk: tracks[t_idx].last_seen += 1
        for d_idx, t_idx in matches:
            tracks[t_idx].last_seen = 0
            tracks[t_idx].update(det_boxes[d_idx], det_kps[d_idx], det_kpc[d_idx])
        for d_idx in unmatched_det:
            tracks.append(Track(det_boxes[d_idx], det_kps[d_idx], det_kpc[d_idx],
                                scaler, T, args.use_angles, args.ema_kps, args.kp_conf))
        tracks = [t for t in tracks if t.last_seen <= args.max_age]

        # --- SALUTE flag for this frame ---
        salute_any = False

        # classify + logic
        for t in tracks:
            # fast fall
            if t.fall_sticky > 0:
                t.fall_sticky -= 1; t.label = "Fall"; t.conf = 1.0
            else:
                if t.fall_trigger_sum(args.fall_drop_thr, args.fall_tilt_thr, args.fall_window, args.fall_sticky):
                    t.label = "Fall"; t.conf = 1.0
                fscore = fall_confidence(t, win=args.fall_window)
                if fscore >= args.fall_conf_thr:
                    t.fall_sticky = args.fall_sticky; t.label = "Fall"; t.conf = max(t.conf, 0.99)

            if len(t.buf_xy) >= max(2, args.min_seq_len):
                bx = list(t.buf_xy)[-T:]
                if t.use_angles:
                    bs = list(t.buf_ang_sin)[-T:] if t.buf_ang_sin is not None else None
                    bc = list(t.buf_ang_cos)[-T:] if t.buf_ang_cos is not None else None
                else:
                    bs = bc = None

                feats_noang   = build_features_from_buffers(bx, None, None, use_angles=False, ms_strides=ms_strides)
                feats_withang = build_features_from_buffers(bx, bs, bc, use_angles=True,  ms_strides=ms_strides)
                feats = ensure_feat_layout(feats_noang, feats_withang, scaler)

                try:
                    feats_std = scaler.transform(feats).astype(np.float32)
                except Exception:
                    feats = feats_withang if feats is feats_noang else feats_noang
                    feats_std = scaler.transform(feats).astype(np.float32)

                x_seq = feats_std[None, :, :]
                prob = safe_predict(clf, x_seq)
                if prob is None:
                    continue

                t.ema_alpha_prob = args.ema
                prob = t.smooth_prob(prob, adaptive=args.adaptive_ema)
                pred_id = int(np.argmax(prob)); maxp = float(prob[pred_id]); lbl = inv_label[pred_id]

                # motion & gait cues 
                me  = t.recent_motion_energy(win=args.walk_speed_win)
                osc = gait_osc_score(t.buf_xy, win=max(args.walk_speed_win, 8))
                kps_last = np.array(t.buf_xy[-1]).reshape(17,2)
                knee_s   = knee_flexion_score(kps_last)
                hip_up   = hip_height_score(t.buf_hipy_norm, win=6)

                # --- SALUTE detection (new) ---
                try:
                    if is_salute_pose(kps_last):
                        salute_any = True
                except Exception:
                    pass

                # --- walking logic  ---
                if lbl.lower().startswith("walk"):
                    if (me < args.walk_speed_thr) and (osc < args.osc_thr_walk_lo):
                        lbl = stand_name if stand_name else "UNCERTAIN"
                        maxp = min(maxp, 0.60)

                elif lbl.lower().startswith("stand"):
                    if (knee_s > args.knee_sit_thr) and (hip_up < args.hip_up_stand_thr) \
                       and (me < args.walk_speed_thr * args.sit_motion_guard):
                        lbl = "Siting" if "Siting" in inv_label.values() else lbl

                elif lbl.lower().startswith("sit") or "siting" in lbl.lower():
                    if (me >= args.walk_speed_thr*1.25) and (osc >= max(0.22, args.osc_thr_walk_lo+0.08)):
                        lbl = "Walking"

                if maxp < args.reject_thres and t.fall_sticky == 0:
                    lbl = "UNCERTAIN"

                if t.fall_sticky == 0 or lbl == "Fall":
                    t.label = lbl; t.conf = maxp

            # draw
            if len(det_boxes)>0 and args.draw_pose and t.last_seen==0:
                ious = [iou_xyxy(np.array(det_boxes[i]), t.bbox) for i in range(len(det_boxes))]
                j = int(np.argmax(ious)) if len(ious)>0 else -1
                if j >= 0:
                    lbl_txt = f"{t.label} ({t.conf*100:.1f}%)" if len(t.buf_xy)>=args.min_seq_len else "…"
                    vis = draw_bbox_and_pose(vis, det_boxes[j], det_kps[j], det_kpc[j], args.vis_kp_conf, lbl_txt)
            else:
                x1,y1,x2,y2 = [int(v) for v in t.bbox]
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                lbl_txt = f"{t.label} ({t.conf*100:.1f}%)" if len(t.buf_xy)>=args.min_seq_len else "…"
                cv2.putText(vis, lbl_txt, (x1, max(0, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        if len(det_boxes) == 0:
            cv2.putText(vis, "NO PERSON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 2, cv2.LINE_AA)

        # --- SALUTE overlay (new) ---
        if salute_any and salute_rgba is not None:
            hh, ww = vis.shape[:2]
            pad = 10
            oh, ow = salute_rgba.shape[:2]
            corner = args.salute_corner.lower()
            if corner == "tr":
                x, y = ww - ow - pad, pad
            elif corner == "tl":
                x, y = pad, pad
            elif corner == "br":
                x, y = ww - ow - pad, hh - oh - pad
            else:  # "bl"
                x, y = pad, hh - oh - pad
            vis = overlay_rgba(vis, salute_rgba, x, y)

        # HUD
        dt_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        hh, ww = vis.shape[:2]
        cv2.putText(vis, f"{dt_ms:.1f} ms ({fps:.1f} FPS)", (20, hh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)

        # output
        writer.write(vis)

        if gui_enabled:
            try:
                cv2.imshow("Multi-Person Action Recognition (TCN + YOLO-Pose)", vis)
                if cv2.waitKey(1) & 0xFF == 27: break
            except Exception:
                gui_enabled = False

        t_total += (time.perf_counter() - t0); frame_idx += 1

    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()
    if frame_idx>0:
        ms_per_frame = (t_total/frame_idx)*1000.0
        fps_avg = 1000.0/ms_per_frame if ms_per_frame>0 else 0.0
        print(f"[INFER PER FRAME] ~ {ms_per_frame:.2f} ms/frame  (~{fps_avg:.1f} FPS)")

if __name__ == "__main__":
    main()

