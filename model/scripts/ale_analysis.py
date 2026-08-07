"""
ALE (Accumulated Local Effects) analysis for KINN and ANN.
Replaces standard PDP to address feature correlation issue raised by reviewer.

Computes ALE for:
  - w/b ratio (correlated with water and binder contents)
  - SCM% (correlated with FA, SC, SF, total binder)
  - AGE (less correlated, included for completeness)

Compares KINN vs ANN vs Yeh's Equation.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = r"C:\Users\Tianjie Zhang\OneDrive - Rutgers University\Documents\GitHub\concrete strength\model"
os.chdir(MODEL_DIR)
sys.path.insert(0, os.path.join(MODEL_DIR, 'scripts'))
from models import RegressionModel, ANNModel, FEATURES, TARGET, DATA_FILE

NEURONS     = 52
EPOCHS      = 300
BATCH_SIZE  = 32
LR          = 0.000765
TRAIN_SIZE  = 0.85
SEED        = 42

# --- Feature indices ---
print("Features:", FEATURES)
WB_IDX  = FEATURES.index('w/b')
AGE_IDX = FEATURES.index('AGE')
# SCM% isn't a direct feature in raw; closest is FA%, SS%, SF%
# Use SCM% = FA% + SS% + SF% conceptually; but in the model it's not a single feature.
# We'll use FA% as the SCM proxy (most impactful SCM in the dataset)
FA_IDX  = FEATURES.index('FA%') if 'FA%' in FEATURES else None
print(f"WB_IDX={WB_IDX}, AGE_IDX={AGE_IDX}, FA_IDX={FA_IDX}")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
df.dropna(inplace=True)
X_raw = df[FEATURES].values
y_raw = df[TARGET].values

# ── Fit physical params on ALL data ───────────────────────────────────────
def fc_model_fit(X, a, b, e, d):
    AGE = np.maximum(X[:, AGE_IDX], 1e-6)
    wb  = X[:, WB_IDX]
    return (a * np.log(AGE) + b) * (e * np.power(AGE, d)) ** (-wb)

params, _ = curve_fit(fc_model_fit, X_raw, y_raw, p0=[1.,1.,1.,1.], maxfev=10000)
a, b, e, d = params
print(f"Physical params: a={a:.4f}, b={b:.4f}, e={e:.4f}, d={d:.4f}")

# ── Scale ──────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
y_scaler = StandardScaler()
X_all = scaler.fit_transform(X_raw)
y_all = y_scaler.fit_transform(y_raw.reshape(-1, 1))

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, train_size=TRAIN_SIZE, random_state=SEED)

Xtrain = torch.tensor(X_tr, dtype=torch.float32)
Xtest  = torch.tensor(X_te, dtype=torch.float32)
ytrain = torch.tensor(y_tr, dtype=torch.float32)

# Precompute scaler tensors for unscaling inside loss (on CPU)
_x_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
_x_std  = torch.tensor(scaler.scale_, dtype=torch.float32)
_y_std  = float(y_scaler.scale_[0])

def kinn_loss(outputs, targets, inputs, lam=0.5):
    """outputs, targets in scaled-y space; inputs in scaled-x space."""
    mse = nn.MSELoss()(outputs, targets)
    # Unscale AGE and w/b to original values for the physical equation
    AGE_raw = torch.clamp(inputs[:, AGE_IDX] * _x_std[AGE_IDX] + _x_mean[AGE_IDX], min=1.0)
    wb_raw  = inputs[:, WB_IDX] * _x_std[WB_IDX] + _x_mean[WB_IDX]
    fc_phys_raw = (a * torch.log(AGE_raw) + b) * (e * torch.pow(AGE_raw, d)) ** (-wb_raw)
    # Scale physical prediction to y-scaled space
    y_mean = float(y_scaler.mean_[0])
    fc_phys_s = (fc_phys_raw - y_mean) / _y_std
    residual = outputs.squeeze() - fc_phys_s
    residual = torch.nan_to_num(residual, nan=0.0)
    phys_loss = torch.mean(residual**2)
    return (1 - lam) * mse + lam * phys_loss

# ── Train KINN ────────────────────────────────────────────────────────────
print("Training KINN...")
torch.manual_seed(SEED)
kinn = RegressionModel(Xtrain.shape[1], NEURONS)
opt  = optim.Adam(kinn.parameters(), lr=LR)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xtrain, ytrain),
    batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED))
kinn.train()
for ep in range(EPOCHS):
    for xb, yb in loader:
        opt.zero_grad()
        kinn_loss(kinn(xb), yb, xb).backward()
        opt.step()
kinn.eval()

# ── Train ANN ─────────────────────────────────────────────────────────────
print("Training ANN...")
torch.manual_seed(SEED)
ann = ANNModel(Xtrain.shape[1], NEURONS)
opt2 = optim.Adam(ann.parameters(), lr=LR)
loader2 = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xtrain, ytrain),
    batch_size=BATCH_SIZE, shuffle=True,
    generator=torch.Generator().manual_seed(SEED))
ann.train()
crit = nn.MSELoss()
for ep in range(EPOCHS):
    for xb, yb in loader2:
        opt2.zero_grad()
        crit(ann(xb), yb).backward()
        opt2.step()
ann.eval()

# Quick R² check
with torch.no_grad():
    y_true = y_scaler.inverse_transform(
        torch.tensor(y_te, dtype=torch.float32).numpy()).flatten()
    yk = y_scaler.inverse_transform(kinn(Xtest).numpy()).flatten()
    ya = y_scaler.inverse_transform(ann(Xtest).numpy()).flatten()
print(f"KINN R²={r2_score(y_true,yk):.4f}  ANN R²={r2_score(y_true,ya):.4f}")

# ── ALE computation ────────────────────────────────────────────────────────
def ale_continuous(model_fn, X_scaled, feat_idx, n_bins=40):
    """
    Compute 1-D ALE for a continuous feature.
    model_fn: callable X_scaled (np) -> predictions (np, shape [n])
    Returns: bin_centers (original scale), ale_values (original-scale prediction units)
    """
    x = X_scaled[:, feat_idx]
    quantiles = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    n_bins = len(quantiles) - 1

    effects   = []
    centers   = []
    counts    = []

    for i in range(n_bins):
        lo, hi = quantiles[i], quantiles[i+1]
        mask = (x >= lo) & (x < hi) if i < n_bins-1 else (x >= lo) & (x <= hi)
        n_in = mask.sum()
        if n_in == 0:
            continue
        X_lo = X_scaled[mask].copy(); X_lo[:, feat_idx] = lo
        X_hi = X_scaled[mask].copy(); X_hi[:, feat_idx] = hi
        delta = model_fn(X_hi) - model_fn(X_lo)
        effects.append(delta.mean())
        centers.append((lo + hi) / 2)
        counts.append(n_in)

    effects = np.array(effects)
    centers = np.array(centers)
    counts  = np.array(counts)

    # Accumulate
    ale = np.cumsum(effects)
    # Centre by weighted mean
    ale -= np.average(ale, weights=counts)

    # Convert centers back to original scale using scaler
    dummy = np.zeros((len(centers), X_scaled.shape[1]))
    dummy[:, feat_idx] = centers
    centers_orig = scaler.inverse_transform(dummy)[:, feat_idx]

    # Convert ALE to original prediction scale
    # ALE is in scaled-y space; multiply by y_scaler.scale_ to get MPa
    ale_orig = ale * y_scaler.scale_[0]

    return centers_orig, ale_orig


def kinn_predict(X_np):
    with torch.no_grad():
        return kinn(torch.tensor(X_np, dtype=torch.float32)).numpy().flatten()

def ann_predict(X_np):
    with torch.no_grad():
        return ann(torch.tensor(X_np, dtype=torch.float32)).numpy().flatten()

# Yeh's equation (on original scale, but we need scaled inputs → unscale first)
def yeh_predict_scaled(X_np):
    X_orig = scaler.inverse_transform(X_np)
    AGE = np.maximum(X_orig[:, AGE_IDX], 1e-6)
    wb  = X_orig[:, WB_IDX]
    fc  = (a * np.log(AGE) + b) * (e * np.power(AGE, d)) ** (-wb)
    # Return in y-scaled space so ALE subtraction works
    return y_scaler.transform(fc.reshape(-1,1)).flatten()

print("Computing ALE for w/b...")
wb_centers_k, wb_ale_k = ale_continuous(kinn_predict, X_all, WB_IDX)
wb_centers_a, wb_ale_a = ale_continuous(ann_predict,  X_all, WB_IDX)
wb_centers_y, wb_ale_y = ale_continuous(yeh_predict_scaled, X_all, WB_IDX)

print("Computing ALE for AGE...")
age_centers_k, age_ale_k = ale_continuous(kinn_predict, X_all, AGE_IDX)
age_centers_a, age_ale_a = ale_continuous(ann_predict,  X_all, AGE_IDX)
age_centers_y, age_ale_y = ale_continuous(yeh_predict_scaled, X_all, AGE_IDX)

if FA_IDX is not None:
    print("Computing ALE for FA%...")
    fa_centers_k, fa_ale_k = ale_continuous(kinn_predict, X_all, FA_IDX)
    fa_centers_a, fa_ale_a = ale_continuous(ann_predict,  X_all, FA_IDX)
    fa_centers_y, fa_ale_y = ale_continuous(yeh_predict_scaled, X_all, FA_IDX)

# ── Publication-quality plot ────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'Times New Roman',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
})

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

colors = {'KINN': '#1a6faf', 'ANN': '#c0392b', "Yeh's Eq.": '#27ae60'}
styles = {"Yeh's Eq.": ('--', 1.8), 'ANN': ('-.', 1.8), 'KINN': ('-', 2.2)}
panel_labels = ['(a)', '(b)', '(c)']

panel_data = [
    (axes[0], wb_centers_y, wb_ale_y, wb_centers_a, wb_ale_a, wb_centers_k, wb_ale_k,
     'Water-to-Binder Ratio (w/b)', 'Effect of w/b on Compressive Strength'),
    (axes[1], age_centers_y, age_ale_y, age_centers_a, age_ale_a, age_centers_k, age_ale_k,
     'Curing Age (days)', 'Effect of Curing Age on Compressive Strength'),
]
if FA_IDX is not None:
    panel_data.append(
        (axes[2], fa_centers_y, fa_ale_y, fa_centers_a, fa_ale_a, fa_centers_k, fa_ale_k,
         'Fly Ash Replacement Ratio (FA%)', 'Effect of FA% on Compressive Strength'))

for pi, pdata in enumerate(panel_data):
    ax = pdata[0]
    cy, ay_y, ca, ay_a, ck, ay_k, xlabel, title = pdata[1:]

    for label, cx, ale in [("Yeh's Eq.", cy, ay_y), ('ANN', ca, ay_a), ('KINN', ck, ay_k)]:
        ls, lw = styles[label]
        ax.plot(cx, ale, color=colors[label], lw=lw, ls=ls, label=label, zorder=3)

    ax.axhline(0, color='#888', lw=0.8, ls=':', zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('ALE of $f_c$ (MPa)')
    ax.set_title(title, pad=6)
    ax.legend(loc='best', framealpha=0.85, edgecolor='#ccc')
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    # Panel label (a)(b)(c)
    ax.text(-0.12, 1.04, panel_labels[pi], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

plt.tight_layout(pad=1.5)
out_png = os.path.join(MODEL_DIR, 'Figure6_ALE.png')
out_pdf = os.path.join(MODEL_DIR, 'Figure6_ALE.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
plt.rcParams.update(plt.rcParamsDefault)
