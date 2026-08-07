"""
Generate Figure 6(f): single-panel ALE for w/b ratio.
Reuses trained models from ale_wb_scm.py logic.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

MODEL_DIR = r"C:\Users\Tianjie Zhang\OneDrive - Rutgers University\Documents\GitHub\concrete strength\model"
os.chdir(MODEL_DIR)
sys.path.insert(0, os.path.join(MODEL_DIR, 'scripts'))
from models import RegressionModel, ANNModel, FEATURES, TARGET, DATA_FILE, \
                   A_FIT as a, B_FIT as b, E_FIT as e, D_FIT as d

NEURONS = 52; EPOCHS = 300; BATCH_SIZE = 32; LR = 0.000765; SEED = 42
WB_IDX = FEATURES.index('w/b'); AGE_IDX = FEATURES.index('AGE')

df    = pd.read_excel(DATA_FILE, sheet_name='Sheet1').dropna()
X_raw = df[FEATURES].values.astype(float)
y_raw = df[TARGET].values.astype(float)

scaler   = StandardScaler(); y_scaler = StandardScaler()
X_all    = scaler.fit_transform(X_raw)
y_all    = y_scaler.fit_transform(y_raw.reshape(-1, 1))

X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, train_size=0.85, random_state=SEED)
Xtr = torch.tensor(X_tr, dtype=torch.float32)
Xte = torch.tensor(X_te, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.float32)

_xm = torch.tensor(scaler.mean_, dtype=torch.float32)
_xs = torch.tensor(scaler.scale_, dtype=torch.float32)
_ys = float(y_scaler.scale_[0]); _ym = float(y_scaler.mean_[0])

def kinn_loss(out, tgt, inp, lam=0.5):
    mse = nn.MSELoss()(out, tgt)
    AGE_r = torch.clamp(inp[:, AGE_IDX] * _xs[AGE_IDX] + _xm[AGE_IDX], min=1.0)
    wb_r  = inp[:, WB_IDX] * _xs[WB_IDX] + _xm[WB_IDX]
    fc_p  = (a * torch.log(AGE_r) + b) * (e * torch.pow(AGE_r, d)) ** (-wb_r)
    fc_ps = (fc_p - _ym) / _ys
    return (1-lam)*mse + lam*torch.mean((out.squeeze()-torch.nan_to_num(fc_ps))**2)

def train(ModelClass, loss_fn):
    torch.manual_seed(SEED)
    m = ModelClass(Xtr.shape[1], NEURONS)
    opt = optim.Adam(m.parameters(), lr=LR)
    dl  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE,
        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    m.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(m, xb, yb).backward(); opt.step()
    m.eval(); return m

print("Training KINN...")
kinn = train(RegressionModel, lambda m, xb, yb: kinn_loss(m(xb), yb, xb))
print("Training ANN...")
ann  = train(ANNModel, lambda m, xb, yb: nn.MSELoss()(m(xb), yb))

with torch.no_grad():
    yk = y_scaler.inverse_transform(kinn(Xte).numpy()).flatten()
    ya = y_scaler.inverse_transform(ann(Xte).numpy()).flatten()
    yt = y_scaler.inverse_transform(y_te).flatten()
print(f"KINN R²={r2_score(yt,yk):.4f}  ANN R²={r2_score(yt,ya):.4f}")

def pred_kinn(Xs):
    with torch.no_grad():
        return kinn(torch.tensor(Xs, dtype=torch.float32)).numpy().flatten()
def pred_ann(Xs):
    with torch.no_grad():
        return ann(torch.tensor(Xs, dtype=torch.float32)).numpy().flatten()
def pred_yeh(Xs):
    Xo = scaler.inverse_transform(Xs)
    AGE = np.maximum(Xo[:, AGE_IDX], 1e-6)
    wb  = Xo[:, WB_IDX]
    fc  = (a*np.log(AGE)+b)*(e*np.power(AGE,d))**(-wb)
    return y_scaler.transform(fc.reshape(-1,1)).flatten()

def ale_1d(pred_fn, X_s, feat_idx, n_bins=50):
    x = X_s[:, feat_idx]
    qs = np.unique(np.percentile(x, np.linspace(0, 100, n_bins+1)))
    effects, centers, counts = [], [], []
    for i in range(len(qs)-1):
        lo, hi = qs[i], qs[i+1]
        mask = (x>=lo)&(x<=hi) if i==len(qs)-2 else (x>=lo)&(x<hi)
        if mask.sum()==0: continue
        Xlo=X_s[mask].copy(); Xlo[:,feat_idx]=lo
        Xhi=X_s[mask].copy(); Xhi[:,feat_idx]=hi
        d_ = pred_fn(Xhi)-pred_fn(Xlo)
        effects.append(d_.mean()); centers.append((lo+hi)/2); counts.append(mask.sum())
    effects=np.array(effects); centers=np.array(centers); counts=np.array(counts)
    ale = np.cumsum(effects)-np.average(np.cumsum(effects), weights=counts)
    dummy=np.zeros((len(centers),X_s.shape[1])); dummy[:,feat_idx]=centers
    cx = scaler.inverse_transform(dummy)[:,feat_idx]
    return cx, ale*y_scaler.scale_[0]

# ── Train XGB and RF (tree models, scale-invariant but use scaled X for consistency) ──
print("Training XGB...")
xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=SEED, n_jobs=-1, verbosity=0)
xgb.fit(X_tr, y_tr.ravel())
print(f"  XGB test R²={r2_score(y_scaler.inverse_transform(y_te), y_scaler.inverse_transform(xgb.predict(X_te).reshape(-1,1))):.4f}")

print("Training RF...")
rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                           min_samples_split=5, min_samples_leaf=2,
                           random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr.ravel())
print(f"  RF  test R²={r2_score(y_scaler.inverse_transform(y_te), y_scaler.inverse_transform(rf.predict(X_te).reshape(-1,1))):.4f}")

def pred_xgb(Xs): return xgb.predict(Xs)
def pred_rf(Xs):  return rf.predict(Xs)

print("Computing ALE w/b ...")
wb_k_x, wb_k_y = ale_1d(pred_kinn, X_all, WB_IDX)
wb_a_x, wb_a_y = ale_1d(pred_ann,  X_all, WB_IDX)
wb_y_x, wb_y_y = ale_1d(pred_yeh,  X_all, WB_IDX)
wb_x_x, wb_x_y = ale_1d(pred_xgb,  X_all, WB_IDX)
wb_r_x, wb_r_y = ale_1d(pred_rf,   X_all, WB_IDX)

# Clip x-axis to actual ALE curve range (avoid empty right tail)
x_max = max(wb_k_x.max(), wb_a_x.max(), wb_y_x.max())
WB_LIM = (df['w/b'].min(), x_max * 1.01)

# ── Single-panel publication figure ───────────────────────────────────────
plt.rcParams.update({
    'font.family':'Times New Roman','font.size':11,
    'axes.labelsize':11,'legend.fontsize':10,
    'xtick.labelsize':10,'ytick.labelsize':10,
})

fig, ax = plt.subplots(figsize=(5.5, 4.2))

COLORS = {'KINN':'#1a6faf','ANN':'#c0392b',"Yeh's Eq.":'#27ae60',
          'XGB':'#e67e22','RF':'#8e44ad'}
STYLES = {"Yeh's Eq.":('--',1.8),'ANN':('-.',1.6),'KINN':('-',2.2),
          'XGB':(':',1.8),'RF':((0,(5,2)),1.6)}

for label, cx, cy in [("Yeh's Eq.", wb_y_x, wb_y_y),
                       ('ANN',       wb_a_x, wb_a_y),
                       ('XGB',       wb_x_x, wb_x_y),
                       ('RF',        wb_r_x, wb_r_y),
                       ('KINN',      wb_k_x, wb_k_y)]:
    ls, lw = STYLES[label]
    ax.plot(cx, cy, color=COLORS[label], lw=lw, ls=ls, label=label)

ax.axhline(0, color='#888', lw=0.8, ls=':')
ax.set_xlim(0.3, 0.5)
ax.set_ylim(-15, 15)
ax.set_xlabel('w/b')
ax.set_ylabel('ALE of $f_c$ (MPa)')
ax.set_title('')
ax.legend(framealpha=0.88, edgecolor='#ccc', loc='upper right')
ax.grid(True, alpha=0.25, lw=0.6)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
out_png = os.path.join(MODEL_DIR, 'Figure6f_ALE_wb.png')
out_pdf = os.path.join(MODEL_DIR, 'Figure6f_ALE_wb.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"Done: {out_png}")
