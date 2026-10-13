# %% [markdown]
#  # thesis_common.py : shared code for the LOOCV pipeline
# 
#  Imported (`from thesis_common import *`) by 01_C0.py, 02_C1.py, 03_C2.py, 04_C3.py and 05_evaluate.py.
# 
# 
# 
#  * Preprocessing: per-recording, per-channel **z-score** on the raw channels (as in super.py).
# 
#  * Evaluation: **stitched** overlapping windows with the 10 % margins dropped (as in super.py), on whole recordings.
# 
#  * Folds: leave-one-individual-out over the 14 individuals; 2 validation individuals per fold; splits saved to `fold_splits.json`.
# 
#  * Everything else (model, losses, sampler, epochs, patience, pseudo-label settings) is copied from final.py.
# 
# 
# 
#  Environment variables: THESIS_PROFILE=smoke|final, OUT_DIR, WORKER / N_WORKERS (fold parallelism, fold % N_WORKERS == WORKER),
# 
#  DEADLINE_H (hours before a clean stop, default 20). Create OUT/STOP to stop cleanly; rerun to resume.

# %%
import os, re, json, copy, time, pickle, random, math, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from scipy import signal, stats
from sklearn.metrics import confusion_matrix

try:
    from IPython.display import display
except Exception:
    display = lambda x: print(x)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
np.set_printoptions(precision=2, suppress=True, linewidth=50000)

WORKER, N_WORKERS = int(os.environ.get("WORKER", 0)), int(os.environ.get("N_WORKERS", 1))
n_threads = max(1, (os.cpu_count() or 2) // N_WORKERS)
for v_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[v_] = str(n_threads)
torch.set_num_threads(n_threads)

_wd = r"C:\Users\Bea Loubser\OneDrive\Desktop\Thesis\Vnet_seaturtles_semisupervised\notebooks"
if os.path.exists(_wd): os.chdir(_wd)

dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")

PROFILE = os.environ.get("THESIS_PROFILE", "final")
PROF = dict(
    final=dict(SEEDS=[0, 1, 2], N_FOLDS=None, EPOCHS=150, PAT=20, N_TRAIN=10000, N_VAL=8000, PS_ITERS=3, PS_EPOCHS=15,
               N_CAND=40000, PRE_STEPS=4000, PRE_BS=32),
    smoke=dict(SEEDS=[0], N_FOLDS=2, EPOCHS=2, PAT=2, N_TRAIN=640, N_VAL=320, PS_ITERS=2, PS_EPOCHS=1,
               N_CAND=1500, PRE_STEPS=100, PRE_BS=8),
)[PROFILE]
globals().update(PROF)

SEED = 42
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    if hasattr(torch, "xpu") and torch.xpu.is_available(): torch.xpu.manual_seed_all(s)

OUT = Path(os.environ.get("OUT_DIR", f"Model_results/LOOCV_zscore_stitched_{PROFILE}"))
for sub in ("units_C0", "units_C1", "units_C2", "units_C3", "models", "fig", "eval"): (OUT / sub).mkdir(parents=True, exist_ok=True)

# ---- classes: (name, behaviour codes, sampling_weight, loss_weight)
CL = [("Other", None, 1, 0.1), ("Breathing", [1, 2, 40], 1, 1), ("Feeding", [3,4,5,6,7,11,12,13,18,19,20,21], 2, 2),
      ("Gliding", [16, 17], 3, 1), ("Resting", [29,30,31,32], 0.5, 1), ("Scratching", [34,35,36], 2, 2),
      ("Swimming", [42,45,48,43,46,49,27,44,47,50,56,52,54,55], 0.5, 1)]
CLASS_NAMES = [c[0] for c in CL]; NC = len(CL)
TARGETS = ["Feeding", "Scratching", "Gliding"]; TARGET_IDS = [CLASS_NAMES.index(n) for n in TARGETS]
SAMPLING_W = np.array([c[2] for c in CL], float)
ponde = torch.tensor([c[3] for c in CL], dtype=torch.float32); ponde /= ponde.sum()

DESC_A, DESC_B_LAB, DESC_UNLAB_A, DESC_UNLAB_B = [1,2,3,4,5,6,25], [0,1,2,3,4,5,24], [0,1,2,3,4,5,9], [0,1,2,3,4,5]
COL_BEH, DT = 27, 0.05
FS = 1 / DT
W = int(40 / DT)            # 800
SUB = 50
DIR_IN, DIR_A, DIR_B = "Data", "Data/TV_processed_unlabelled", "Data/TV_Martinique_AGM_unlabelled"
PRESSURE_DIFF_S = 2.0

LR, BS, DEPTH, DROPOUT = 1e-4, 32, 32, 0.5
PSEUDO_LR, PSEUDO_W, PSEUDO_SMOOTH, PSEUDO_PASSES = 1e-4, 0.10, 0.10, 2.0
MAX_PER_CLASS, MIN_CONF = 1000, 5
TS_CAP_FLOOR, TS_CAP_RATIO = 20_000, 1.0
TARGET_PREC = {"Feeding": .9, "Scratching": .9, "Gliding": .9, "default": .95}
GATE_MARGIN, GATE_MACRO_TOL = 0.0, 0.01
FINAL = PS_ITERS


# %% [markdown]
#  ## Recordings, individuals, folds

# %%
NAMES_ALL = ['CC-07-48_18-02-2019','CC-07-48_06-10-2018','CC-07-48_04-10-2018','CC-07-115_13-05-2019','CC-07-115_14-05-2019',
 'CC-07-48_08-04-2019_1','CC-07-48_08-04-2019_2','CC-07-48_08-04-2019_3','CC-07-48_22-05-2018','CC-07-108_11-02-2019',
 'CC-07-59_20-05-2018','CC-07-47_14-02-2018','CC-07-48_10-10-2018','CC-07-48_08-10-2018','CC-07-48_26-05-2018',
 'CC-07-107_11-02-2019','CC-09-72_04-11-2022']
NAMES_UNLAB_A = ['20211130-203207-CC-09-29_TV_112021','20211114-120603-CC-09-28_TV_102021','20211122-104139-CC-09-28_TV_102021',
 '20220414-002025-CC-09-80','20211108-055826-CC-09-28_TV_102021','20211024-141045-CC-09-29_TV_102021',
 '20211101-081704-CC-09-28_TV_102021','20211102-115134-CC-09-29_TV_102021','20211113-213025-CC-09-29_TV_112021',
 '20211107-072758-CC-09-29_TV_102021','20220508-195728-CC-09-80','20220114-125839-CC-09-29_TV_012022',
 '20220421-211441-CC-09-80','20211121-203649-CC-09-29_TV_112021']
NAMES_UNLAB_B = ["AGM1_TV_AN_08122020","AGM1_TV_AN_15122020","AGM1_TV_AN_19012021","AGM1_TV_GA_07052021","AGM1_TV_GA_23122020",
 "AGM1_TV_GA_25052021","AGM1_TV_GA_30042021","AGM2_TV_AN_08122020_1","AGM2_TV_AN_15122020_2","AGM2_TV_AN_29012021",
 "AGM2_TV_GA_23122020","AGM4_TV_AN_08122020","AGM4_TV_AN_19012021_2","AGM4_TV_AN_29012021","AGM4_TV_GA_23122020",
 "AGM4_TV_GA_30042021","AGM5_TV_GA_22042021","AGM5_TV_GA_30042021","AGM6_TV_GA_07052021","AGM7_TV_AN_25052021_1"]
OVR = {'CC-07-115_13-05-2019': 'CC-07-115', 'CC-07-115_14-05-2019': 'CC-07-115'}
def individual_id_of(n): return OVR.get(n, re.sub(r'_\d+$', '', n))
IND2NAMES = defaultdict(list)
for n_ in NAMES_ALL: IND2NAMES[individual_id_of(n_)].append(n_)
INDIVIDUALS = sorted(IND2NAMES)
assert len(INDIVIDUALS) == 14, len(INDIVIDUALS)

def build_splits():
    out = {}
    for fi, ti in enumerate(INDIVIDUALS[:N_FOLDS]):
        rem = [i for i in INDIVIDUALS if i != ti]; rng = np.random.RandomState(SEED + fi); sh = [str(i) for i in rng.permutation(rem)]
        v, t = sh[:2], sorted(sh[2:]); f = lambda L: [n for i in L for n in IND2NAMES[i]]
        out[fi] = dict(test_ind=ti, test=list(IND2NAMES[ti]), val_inds=v, val=f(v), train_inds=t, train=f(t))
    return out

def save_splits(sp):
    json.dump({str(k): v for k, v in sp.items()}, open(OUT / "fold_splits.json", "w"), indent=1)

def load_splits():
    p = OUT / "fold_splits.json"
    assert p.exists(), "fold_splits.json not found: run 01_C0.py first"
    return {int(k): v for k, v in json.load(open(p)).items()}


# %% [markdown]
#  ## Data: loading + z-score preprocessing (filled in place by init_data)

# %%
matrices, labels, MATS = {}, {}, {}

def resample_pressure_diff(mes, diff_s=PRESSURE_DIFF_S, depth_col=9):
    mes = mes.copy()
    chg = np.where(np.diff(mes.iloc[:, depth_col]) != 0)[0]
    mes['Pressure_corr'] = np.nan; mes.iloc[chg, -1] = mes.iloc[chg, depth_col]
    mes['Pressure_corr'] = mes['Pressure_corr'].interpolate(method='linear').bfill()
    k = max(1, int(round(diff_s / (np.median(np.diff(chg)) * DT))))
    mes['Pressure_diff'] = np.nan; mes.iloc[chg, -1] = mes.iloc[chg, depth_col].diff(k)
    mes['Pressure_diff'] = mes['Pressure_diff'].interpolate(method='linear').bfill()
    return mes

def load_labelled(names):
    mats, labs = {}, {}
    for n in names:
        pa, pb = f"{DIR_IN}/out_A_09062025/{n}.npy", f"{DIR_IN}/out_B/{n}.npy"
        if os.path.exists(pa): f, cols, cb = pa, DESC_A, COL_BEH
        elif os.path.exists(pb): f, cols, cb = pb, DESC_B_LAB, COL_BEH - 1
        else: raise FileNotFoundError(n)
        full = np.load(f); pre = full[:, cb]; lab = np.zeros(len(pre), dtype=np.int8)
        for cid, (_, behs, _, _) in enumerate(CL):
            if behs:
                for b in behs: lab[pre == b] = cid
        mats[n] = full[:, cols].astype(np.float32); labs[n] = lab
    return mats, labs

def load_unlab_A(names):
    return {n: np.load(f"{DIR_A}/{n}.npy")[:, DESC_UNLAB_A].astype(np.float32) for n in names}

def load_unlab_B(names):
    out = {}
    for n in names:
        mes = pd.DataFrame(np.load(f"{DIR_B}/{n}.npy")); assert mes.shape[1] in (11, 21), (n, mes.shape)
        mes = resample_pressure_diff(mes)
        out[n] = mes.iloc[:, DESC_UNLAB_B + [mes.shape[1] - 1]].to_numpy(dtype=np.float32, copy=True)
    return out

def preprocess(m):
    """super.py preprocessing: per-recording, per-channel z-score on the raw channels."""
    x = m.astype(np.float64); sd = x.std(0); sd[sd < 1e-8] = 1.0
    return ((x - x.mean(0)) / sd).astype(np.float32)

def init_data(unlab=()):
    """Load labelled data (always) and the requested unlabelled pools ('A', 'B'). Safe to call repeatedly."""
    if not matrices:
        m, l = load_labelled(NAMES_ALL); matrices.update({n: preprocess(x) for n, x in m.items()}); labels.update(l)
        MATS["L"] = matrices
    if "A" in unlab and "A" not in MATS: MATS["A"] = {n: preprocess(x) for n, x in load_unlab_A(NAMES_UNLAB_A).items()}
    if "B" in unlab and "B" not in MATS: MATS["B"] = {n: preprocess(x) for n, x in load_unlab_B(NAMES_UNLAB_B).items()}
    for d in MATS.values(): assert all(np.isfinite(x).all() for x in d.values())
    hrs = lambda d: sum(len(x) for x in d.values()) * DT / 3600
    print("loaded (z-score):", " | ".join(f"{k} {hrs(v):.1f} h" for k, v in MATS.items()))


# %% [markdown]
#  ## Model, losses, labelled sampler

# %%
class SpatialDropout1D(nn.Module):
    def __init__(self, p): super().__init__(); self.d = nn.Dropout2d(p)
    def forward(self, x): return self.d(x.unsqueeze(3)).squeeze(3)

class VNet(nn.Module):
    def __init__(self, cin, cout, depth, dropout=0.5, k=5):
        super().__init__(); d = depth; self.k = k; self.drop = SpatialDropout1D(dropout)
        self.conv1 = self.dc(cin, d); self.down1 = nn.Conv1d(d, 2*d, 2, 2); self.conv2 = self.dc(2*d, 2*d)
        self.down2 = nn.Conv1d(2*d, 4*d, 2, 2); self.conv3 = self.dc(4*d, 4*d)
        self.down3 = nn.Conv1d(4*d, 8*d, 2, 2); self.conv4 = self.dc(8*d, 8*d)
        self.up1 = self.up(8*d, 4*d); self.conv5 = self.dc(4*d, 4*d)
        self.up2 = self.up(4*d, 2*d); self.conv6 = self.dc(2*d, 2*d)
        self.up3 = self.up(2*d, d); self.conv7 = self.dc(d, d)
        self.classifier = nn.Conv1d(d, cout, 1)
    def dc(self, i, o):
        return nn.Sequential(nn.Conv1d(i, o, self.k, padding=2), nn.BatchNorm1d(o), nn.ReLU(True),
                             nn.Conv1d(o, o, self.k, padding=2, bias=False), nn.BatchNorm1d(o), nn.ReLU(True))
    def up(self, i, o): return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv1d(i, o, 3, padding=1), nn.BatchNorm1d(o), nn.ReLU(True))
    def forward(self, x):
        c1 = self.conv1(x); c1d = self.drop(c1); d1 = F.relu(self.down1(c1d))
        c2 = self.conv2(d1); c2d = self.drop(c2); d2 = F.relu(self.down2(d1 + c2d))
        c3d = self.drop(self.conv3(d2)); d3 = F.relu(self.down3(d2 + c3d))
        c4d = self.drop(self.conv4(d3))
        u1 = self.up1(c4d); c5 = self.drop(self.conv5(c3d + u1)) + u1
        u2 = self.up2(c5); c6 = self.drop(self.conv6(c2 + u2)) + u2
        u3 = self.up3(c6); c7 = self.drop(self.conv7(c1 + u3)) + u3
        return F.softmax(self.classifier(c7), dim=1)

def new_model(): return VNet(7, NC, DEPTH, DROPOUT).to(dev)
def cpu_sd(sd): return {k: v.detach().cpu().clone() for k, v in sd.items()}

def save_state(model, stage, fi, seed, tag):
    p = OUT / "models" / stage; p.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_sd(model.state_dict()), p / f"f{fi+1:02d}_s{seed}_{tag}.pth")

def load_model(path):
    m = new_model(); m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); m.eval(); return m

def dice_ce(p, t, w=None):
    w = (ponde if w is None else w).to(p.device)
    dice = (2 * (t * p).sum(2) + 1e-8) / (t.sum(2) + p.sum(2) + 1e-8)
    ce = -(t * torch.log(p.clamp_min(1e-8)) * w.view(1, -1, 1)).sum(1).mean()
    return 0.5 * (1 - (dice * w.view(1, -1)).sum(1).mean()) + 0.5 * ce

def masked_ce(p, y, m):
    lp = torch.log(p.clamp_min(1e-8)); ys = torch.where(m, y, torch.zeros_like(y))
    nll = -lp.gather(1, ys.unsqueeze(1)).squeeze(1)
    per = (1 - PSEUDO_SMOOTH) * nll + PSEUDO_SMOOTH * (-lp.mean(1))
    w = ponde.to(p.device)[ys] * m.float()
    return (per * w).sum() / w.sum().clamp_min(1e-8) if m.any() else p.sum() * 0.0

class LabelledSampler:
    """Class-balanced window sampler (class ~ sampling weight, recording ~ class occurrence, 10% uniform windows)."""
    def __init__(self, names, seed, noise_sigma=None):
        self.rng = np.random.RandomState(seed); self.names = list(names); self.ns = noise_sigma
        self.idx = {c: [(n, np.flatnonzero(labels[n] == c)) for n in names if (labels[n] == c).any()] for c in range(NC)}
        self.cw = np.array([SAMPLING_W[c] if self.idx[c] else 0 for c in range(NC)], float); self.cw /= self.cw.sum()
        self.rw = {c: np.array([len(i) for _, i in self.idx[c]], float) for c in range(NC)}
        for c in self.rw:
            if len(self.rw[c]): self.rw[c] /= self.rw[c].sum()
    def one(self):
        r = self.rng
        if r.rand() < 0.1:
            n = self.names[r.randint(len(self.names))]; t = r.randint(0, len(labels[n]) - W); return n, t
        c = r.choice(NC, p=self.cw); j = r.choice(len(self.idx[c]), p=self.rw[c]); n, ix = self.idx[c][j]
        t = int(np.clip(ix[r.randint(len(ix))] - r.randint(W), 0, len(labels[n]) - W)); return n, t
    def batch(self, size):
        X = np.empty((size, W, 7), np.float32); Y = np.empty((size, W), np.int64)
        for i in range(size):
            n, t = self.one(); X[i] = matrices[n][t:t + W]; Y[i] = labels[n][t:t + W]
        if self.ns is not None:
            m = self.rng.rand(size) < 0.5
            X[m] += (self.rng.randn(int(m.sum()), W, 7) * self.ns).astype(np.float32)
        return X, Y

def cut_labelled(names):
    """Non-overlapping 40 s windows (used for pseudo-label quality diagnostics and the calibration/centroid fits)."""
    Xs, Ys = [], []
    for n in names:
        k = len(matrices[n]) // W
        if k: Xs.append(matrices[n][:k * W].reshape(k, W, 7)); Ys.append(labels[n][:k * W].reshape(k, W).astype(np.int64))
    return {"X": np.concatenate(Xs), "Y": np.concatenate(Ys)}


# %% [markdown]
#  ## Stitched evaluation (as in super.py) and metrics

# %%
def cut_full_slice(matrix, label):
    margin = int(0.1 * W); stride = W - 2 * margin; size = len(matrix) // stride - 1
    if size < 1: return None, None
    X = np.empty([size, W, matrix.shape[1]], np.float32); Y = np.empty([size, W], dtype=int)
    for i in range(size):
        t = i * stride; X[i] = matrix[t:t + W]; Y[i] = label[t:t + W]
    return X, Y

def stick(Y):
    ws = Y.shape[1]; margin = int(0.1 * ws); res = [Y[0, :ws - margin]]
    for i in range(1, len(Y)): res.append(Y[i, margin:-margin])
    return np.concatenate(res, axis=0)

@torch.no_grad()
def stitched_cm(model, names, bs=64):
    """Confusion matrix (rows = true, cols = predicted) over whole recordings, overlapping windows stitched, margins dropped."""
    model.eval(); cm = np.zeros((NC, NC), np.int64)
    for n in names:
        X, Y = cut_full_slice(matrices[n], labels[n])
        if X is None: continue
        preds = [model(torch.from_numpy(X[s:s + bs]).float().permute(0, 2, 1).to(dev)).argmax(1).cpu().numpy() for s in range(0, len(X), bs)]
        cm += confusion_matrix(stick(Y), stick(np.concatenate(preds)), labels=range(NC))
    return cm

def metrics_cm(cm):
    tp = np.diag(cm).astype(float); sup = cm.sum(1); fn = sup - tp; fp = cm.sum(0) - tp
    f1 = np.where(sup > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(f1=f1, macro=np.nanmean(f1), macro_noOther=np.nanmean(f1[1:]),
                    target=np.nanmean(f1[TARGET_IDS]) if np.isfinite(f1[TARGET_IDS]).any() else np.nan)

def make_row(meta, cmv, cmt):
    mv, mt = metrics_cm(cmv), metrics_cm(cmt)
    r = {**meta, "val_target": mv["target"], "val_macro": mv["macro"], "val_macroNoOther": mv["macro_noOther"],
         "test_target": mt["target"], "test_macro": mt["macro"], "test_macroNoOther": mt["macro_noOther"], "cm": cmt}
    r.update({f"test_F1_{n}": mt["f1"][i] for i, n in enumerate(CLASS_NAMES)}); return r

def row_of(meta, model, ctx):
    """Result row for a model: stitched confusion matrices on the fold's validation and test recordings."""
    return make_row(meta, stitched_cm(model, ctx["names_val"]), stitched_cm(model, ctx["names_test"]))


# %% [markdown]
#  ## Training (supervised, fine-tuning with pseudo-labels)

# %%
class PseudoSet:
    def __init__(self, X, Y, M): self.X, self.Y, self.M = X, Y, M
    def batch(self, n):
        i = np.random.randint(0, len(self.X), n)
        return (torch.from_numpy(self.X[i]).float().permute(0, 2, 1).to(dev), torch.from_numpy(self.Y[i]).long().to(dev),
                torch.from_numpy(self.M[i]).bool().to(dev))

def run_epoch(model, opt, X, Y, pseudo=None, every=1, step=0):
    model.train(); idx = np.random.permutation(len(X))
    for s in range(0, len(X), BS):
        b = idx[s:s + BS]
        xb = torch.from_numpy(X[b]).permute(0, 2, 1).to(dev)
        yb = F.one_hot(torch.from_numpy(Y[b]), NC).permute(0, 2, 1).float().to(dev)
        loss = dice_ce(model(xb), yb)
        if pseudo is not None and step % every == 0:
            xp, yp, mp = pseudo.batch(BS); loss = loss + PSEUDO_W * masked_ce(model(xp), yp, mp)
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
    return step

@torch.no_grad()
def val_loss(model, Xv, Yv):
    model.eval(); tot, n = 0.0, 0
    for s in range(0, len(Xv), 128):
        xb = torch.from_numpy(Xv[s:s + 128]).permute(0, 2, 1).to(dev)
        yb = F.one_hot(torch.from_numpy(Yv[s:s + 128]), NC).permute(0, 2, 1).float().to(dev)
        tot += dice_ce(model(xb), yb).item(); n += 1
    return tot / n

def train_supervised(names_train, Xv, Yv, seed, noise_sigma=None, init_state=None):
    set_seed(seed); m = new_model()
    if init_state is not None: print("init from pretrained:", m.load_state_dict(init_state, strict=False))
    opt = optim.Adam(m.parameters(), lr=LR); sm = LabelledSampler(names_train, seed, noise_sigma)
    best, bad, best_sd, ep_run = 1e9, 0, None, 0
    for ep in range(EPOCHS):
        X, Y = sm.batch(N_TRAIN); run_epoch(m, opt, X, Y); vl = val_loss(m, Xv, Yv); ep_run = ep + 1
        if vl < best: best, bad, best_sd = vl, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= PAT: break
    m.load_state_dict(best_sd); m.info = dict(epochs_run=ep_run, best_val_loss=best); return m

def finetune(base, names_train, seed, pseudo=None, noise_sigma=None):
    m = copy.deepcopy(base).to(dev); opt = optim.Adam(m.parameters(), lr=PSEUDO_LR); set_seed(seed)
    sm = LabelledSampler(names_train, seed, noise_sigma); every = 1
    if pseudo is not None and PSEUDO_PASSES:
        total = math.ceil(N_TRAIN / BS) * PS_EPOCHS; every = max(1, total // max(1, math.ceil(PSEUDO_PASSES * len(pseudo.X) / BS)))
    step = 0
    for ep in range(PS_EPOCHS):
        X, Y = sm.batch(N_TRAIN); step = run_epoch(m, opt, X, Y, pseudo, every, step)
    return m


# %% [markdown]
#  ## Unlabelled pools, pseudo-window selection, gate

# %%
def make_pool(pool, rng=None):
    """pool is 'A', 'B' or 'AB' (whole pool, natural sampling)."""
    return [(s, n) for s in pool for n in MATS[s]]

def draw_windows(ent, n, rng):
    L = np.array([len(MATS[s][nm]) for s, nm in ent], float); cnt = rng.multinomial(n, L / L.sum()); X, S = [], []
    for (s, nm), c in zip(ent, cnt):
        if c == 0: continue
        m = MATS[s][nm]; st = rng.randint(0, len(m) - W, size=c)
        X.append(np.stack([m[t:t + W] for t in st])); S += [s] * c
    return np.concatenate(X), np.array(S)

def select_pseudo(Y, M, keep, src, seed):
    rng = np.random.RandomState(seed); idx = np.flatnonzero(keep); info = {}
    if len(idx) == 0: return idx, M[:0], info
    pres = np.stack([((Y[idx] == c) & M[idx]).any(1) for c in range(NC)], 1); sel = set()
    for c in range(NC):
        cand = idx[pres[:, c]]
        info[f"nwin_{CLASS_NAMES[c]}"] = len(cand); info[f"Bshare_before_{CLASS_NAMES[c]}"] = float((src[cand] == "B").mean()) if len(cand) else np.nan
        if len(cand) > MAX_PER_CLASS: cand = rng.choice(cand, MAX_PER_CLASS, False)
        sel.update(cand.tolist())
    sel = np.array(sorted(sel)); Ms = M[sel].copy(); Ys = Y[sel]
    cnt = [int((Ms & (Ys == c)).sum()) for c in range(NC)]; cap = max(TS_CAP_FLOOR, int(TS_CAP_RATIO * max(cnt[c] for c in TARGET_IDS)))
    for c in range(NC):
        if cnt[c] > cap:
            r, cc = np.nonzero(Ms & (Ys == c)); d = rng.choice(len(r), cnt[c] - cap, False); Ms[r[d], cc[d]] = False
    k2 = Ms.sum(1) >= MIN_CONF; sel, Ms = sel[k2], Ms[k2]; Ys = Y[sel]
    for c in range(NC):
        w = sel[((Ys == c) & Ms).any(1)]
        info[f"Bshare_after_{CLASS_NAMES[c]}"] = float((src[w] == "B").mean()) if len(w) else np.nan
        info[f"conf_ts_{CLASS_NAMES[c]}"] = int((Ms & (Ys == c)).sum())
    info.update(n_selected=len(sel), B_share_candidates=float((src == "B").mean()), B_share_selected=float((src[sel] == "B").mean()) if len(sel) else np.nan)
    return sel, Ms, info

def quality_rows(Yt, Y, M, keep, meta):
    t = Yt[keep] if keep.any() else Yt[:0]; rows = []
    n_true = np.bincount(Yt.ravel(), minlength=NC)
    for c in range(-1, NC):
        sel = (M[keep] & (Y[keep] == c)) if c >= 0 else M[keep]; a = int(sel.sum()) if keep.any() else 0
        k = int((sel & (t == c)).sum()) if (c >= 0 and keep.any()) else int((sel & (t == Y[keep])).sum()) if keep.any() else 0
        nt = int(n_true[c]) if c >= 0 else int(Yt.size)
        rows.append({**meta, "class": "ALL" if c < 0 else CLASS_NAMES[c], "n_true": nt, "n_assigned": a, "n_correct": k,
                     "precision": k / a if a else np.nan, "coverage": k / nt if nt else np.nan})
    return rows

def gate_pick(rows_r, ctrl_val):
    """Validation-only gate vs same-round control; best passing round or baseline (None)."""
    best, bg = None, -1e9
    for r in rows_r:
        c = ctrl_val[r["round"]]; sc = lambda d: d["val_target"] if np.isfinite(d["val_target"]) else d["val_macro"]
        gain = sc(r) - sc(c); drop = c["val_macro"] - r["val_macro"]
        if gain > GATE_MARGIN and drop <= GATE_MACRO_TOL and gain > bg: best, bg = r, gain
    return best

def finish_chain(rows, ctx, base_meta):
    """Append the 'gated' row (best validated round, or the baseline) to a pseudo-label chain."""
    g = gate_pick(rows, ctx["ctrl_val"]); tgt = g if g is not None else ctx["base_row"]
    rows.append({**tgt, **base_meta, "round": 0 if g is None else g["round"], "kind": "gated"}); return rows


# %% [markdown]
#  ## Resumable stage runner (used by 02_C1, 03_C2, 04_C3)

# %%
class StopRun(Exception): pass
DEADLINE = time.time() + float(os.environ.get("DEADLINE_H", 20)) * 3600
def check_stop():
    if (OUT / "STOP").exists() or (DEADLINE and time.time() > DEADLINE): raise StopRun

def atomic_dump(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def unit_path(stage, fi, seed): return OUT / f"units_{stage}" / f"f{fi+1:02d}_s{seed}.pkl"

def seeds_for(pool): return SEEDS if pool == "A" else [0]      # as in final.py: other pools use seed 0 only

def make_ctx(fi, seed, sp, c0):
    base = new_model(); base.load_state_dict(c0["base_state"]); base.eval()
    set_seed(SEED + 5000 + fi); Xv, Yv = LabelledSampler(sp["val"], SEED + 5000 + fi).batch(N_VAL)
    return dict(fi=fi, seed=seed, test_ind=sp["test_ind"], base=base, base_row=[r for r in c0["rows"] if r["kind"] == "baseline"][0],
                ctrl_val=c0["ctrl_val"], names_train=sp["train"], names_val=sp["val"], names_test=sp["test"],
                EV=cut_labelled(sp["val"]), ET=cut_labelled(sp["test"]), Xv=Xv, Yv=Yv)

def run_stage(stage, runner, pools):
    """For every (fold, seed) with a finished C0 unit, run runner(pool, ctx) -> (rows, quality, selection, calibration) per pool."""
    SP = load_splits()
    try:
        for fi in sorted(SP):
            if fi % N_WORKERS != WORKER: continue
            for seed in SEEDS:
                c0p = unit_path("C0", fi, seed)
                if not c0p.exists(): print(f"missing C0 unit {c0p.name}; run 01_C0.py first"); continue
                todo = [p for p in pools if seed in seeds_for(p)]
                path = unit_path(stage, fi, seed)
                st = pickle.load(open(path, "rb")) if path.exists() else dict(rows=[], q=[], s=[], c=[], done=[])
                todo = [p for p in todo if f"{stage}|{p}" not in st["done"]]
                if not todo: continue
                ctx = make_ctx(fi, seed, SP[fi], pickle.load(open(c0p, "rb")))
                for p in todo:
                    check_stop(); print(f"--- {stage} | fold {fi+1} seed {seed} | pool {p}")
                    r, q, s, c = runner(p, ctx)
                    st["rows"] += r; st["q"] += q; st["s"] += s; st["c"] += c; st["done"].append(f"{stage}|{p}"); atomic_dump(st, path)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
    except StopRun:
        print("stopped cleanly; rerun to resume")



