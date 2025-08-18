import json
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): 
        super().__init__(); self.chomp_size=int(chomp_size)
    def forward(self, x): 
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size>0 else x

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
    def forward(self, x): 
        out=self.net(x); return out + self.down(x)

class TCN(nn.Module):
    def __init__(self, in_feat, channels, num_classes, kernel_size=5, dropout=0.2):
        super().__init__()
        layers=[]; prev=in_feat
        for i,ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation=2**i, dropout=dropout))
            prev=ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout), nn.Linear(channels[-1], num_classes),
        )
    def forward(self, x): 
        x=x.transpose(1,2)  # (B,T,F) -> (B,F,T)
        feats=self.tcn(x); 
        return self.head(feats)

class SeqDataset(Dataset):
    def __init__(self, X, y, time_jitter=0.0, time_mask=0.0, seed=42):
        self.X, self.y = X, y
        self.time_jitter = float(time_jitter); self.time_mask = float(time_mask)
        self.rng = np.random.RandomState(seed)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].copy().astype(np.float32); y = int(self.y[i])
        if self.time_jitter > 0:
            noise = self.rng.normal(0, self.time_jitter, size=x.shape).astype(np.float32)
            x += noise
        if self.time_mask > 0:
            T = x.shape[0]; seg = max(1, int(self.time_mask * T))
            start = self.rng.randint(0, max(1, T-seg))
            x[start:start+seg] = x[start-1 if start>0 else 0]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class WeightedSmoothCE(nn.Module):
    def __init__(self, num_classes, weights=None, eps=0.0):
        super().__init__(); self.num_classes = num_classes; self.eps = float(eps)
        if weights is None: self.register_buffer("weights", None)
        else: self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
    def forward(self, logits, target):
        if self.eps <= 0 and self.weights is None:
            return nn.functional.cross_entropy(logits, target)
        logp = nn.functional.log_softmax(logits, dim=1)
        B, C = logp.shape
        with torch.no_grad():
            true_dist = torch.zeros_like(logp)
            true_dist.fill_(self.eps / (C - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        loss = (-true_dist * logp).sum(dim=1)
        if self.weights is not None:
            w = self.weights.index_select(0, target); loss = loss * w
        return loss.mean()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct=0; total=0
    for xb,yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1); correct += (pred==yb).sum().item(); total += yb.size(0)
    return correct / max(1,total)

def train_one_epoch(model, loader, opt, crit, device, clip=1.0):
    model.train(); total_loss=0.0; total_samp=0
    for xb,yb in loader:
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad(); logits = model(xb); loss = crit(logits, yb); loss.backward()
        if clip and clip>0: nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); total_loss += loss.item()*yb.size(0); total_samp += yb.size(0)
    return total_loss/max(1,total_samp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="cache_kps_final")
    ap.add_argument("--out_dir", type=str, default="runs_maxacc_final")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--channels", type=str, default="64,128,256")
    ap.add_argument("--kernel_size", type=int, default=5)
    ap.add_argument("--class_weight", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--time_jitter", type=float, default=0.0)
    ap.add_argument("--time_mask", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    cache_dir = Path(args.cache_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    Xp, yp, lmp, sp = cache_dir/"X.npy", cache_dir/"y.npy", cache_dir/"label_map.json", cache_dir/"scaler.joblib"
    assert Xp.exists() and yp.exists() and lmp.exists() and sp.exists(), "Run extract_01.py first."

    label_map = json.load(open(lmp,"r",encoding="utf-8"))
    X = np.load(Xp); y = np.load(yp)
    N,T,F = X.shape; num_classes = len(label_map)
    print(f"[INFO] Data X{X.shape}, y{y.shape}, classes={num_classes}  (labels: {label_map})")

    dataset = SeqDataset(X, y, time_jitter=args.time_jitter, time_mask=args.time_mask, seed=args.seed)
    n_val = int(args.val_split * len(dataset)); n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    channels = [int(c.strip()) for c in args.channels.split(",") if c.strip()]
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    model = TCN(in_feat=F, channels=channels, num_classes=num_classes,
                kernel_size=args.kernel_size, dropout=args.dropout).to(device)

    weights = None
    if args.class_weight:
        cls_w = compute_class_weight(class_weight="balanced",
                                     classes=np.arange(num_classes), y=y)
        weights = cls_w.astype(np.float32); print(f"[INFO] class weights: {cls_w}")
    crit = WeightedSmoothCE(num_classes=num_classes, weights=weights, eps=args.label_smoothing)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, mode="max")

    best_acc = 0.0; best_path = out_dir/"tcn_best_final.pt"
    history = {"train_loss": [], "val_acc": []}

    for ep in range(1,args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, crit, device, clip=1.0)
        val_acc = evaluate(model, val_loader, device)
        sched.step(val_acc)

        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "label_map": label_map,
                "scaler_path": str(sp.resolve()),
                "in_feat": F,
                "channels": channels,
                "kernel_size": args.kernel_size,
                "dropout": args.dropout
            }, best_path)
        lr_now = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:03d} | lr={lr_now:.2e} | train_loss={tr_loss:.4f} | val_acc={val_acc:.3f} | best={best_acc:.3f}")

    json.dump(history, open(out_dir/"history.json","w"))
    print(f"[OK] Best saved: {best_path} (val_acc={best_acc:.3f})")

if __name__ == "__main__":
    main()
