"""
Grouped cross-validation: each unique mix design goes entirely to
either the training or the test fold -- no mix design appears in both.

This directly addresses Reviewer #6 Comment 3.

5-fold GroupKFold across all 6 models.
Outputs: grouped_cv_results.xlsx, grouped_cv_summary.xlsx

Run from the model/ directory:
    python scripts/grouped_validation.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import joblib
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import ANNModel, RegressionModel, FEATURES, TARGET, DATA_FILE

warnings.filterwarnings('ignore')

N_FOLDS  = 5
EPOCHS   = 300
WB_INDEX = 14

MIX_COLS = ['PC', 'PC_TYPE', 'FA', 'SS', 'SF', 'FAGG', 'CAGG',
            'WATER', 'AEA', 'WR_HR', 'WR', 'ACC', 'LATEX']


# ── Physics loss (same as main KINN) ─────────────────────────────────────────
def kinn_loss(outputs, targets, inputs, a, b, e, d, lam=0.5):
    mse = nn.MSELoss()(outputs, targets)
    AGE = torch.clamp(inputs[:, 0], min=1e-6)
    wb  = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)
    r = torch.abs(outputs - fc_phys.unsqueeze(1))
    r = torch.nan_to_num(r, nan=0.0, posinf=1e10, neginf=-1e10)
    mean_sq = torch.mean(r ** 2)
    r = r * torch.sqrt(mse / (mean_sq + 1e-8))
    return (1 - lam) * mse + lam * torch.mean(r)


def metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            mean_squared_error(y_true, y_pred) ** 0.5,
            mean_absolute_error(y_true, y_pred))


def run_kinn(X_tr, y_tr, X_te, y_te, a, b, e, d):
    scaler   = StandardScaler()
    y_scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    ytr = y_scaler.fit_transform(y_tr.reshape(-1, 1))

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    Xv = torch.tensor(Xte, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)

    model = RegressionModel(Xt.shape[1], 52)
    opt   = optim.Adam(model.parameters(), lr=0.0007652946585818769)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=8, shuffle=True)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            kinn_loss(model(xb), yb, xb, a, b, e, d).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = y_scaler.inverse_transform(model(Xv).numpy()).flatten()
    return metrics(y_te, pred)


def run_ann(X_tr, y_tr, X_te, y_te):
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    y_mean, y_std = y_tr.mean(),   y_tr.std()
    Xtr = (X_tr - x_mean) / x_std
    Xte = (X_te - x_mean) / x_std
    ytr = (y_tr - y_mean) / y_std

    Xt = torch.FloatTensor(Xtr)
    Xv = torch.FloatTensor(Xte)
    yt = torch.FloatTensor(ytr).view(-1, 1)

    model = ANNModel(Xt.shape[1], 52)
    opt   = optim.Adam(model.parameters(), lr=0.0007652946585818769)
    crit  = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        crit(model(Xt), yt).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xv).numpy().flatten() * y_std + y_mean
    return metrics(y_te, pred)


def run_sklearn(ModelClass, params, X_tr, y_tr, X_te, y_te):
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    y_mean, y_std = y_tr.mean(),   y_tr.std()
    Xtr = (X_tr - x_mean) / x_std
    Xte = (X_te - x_mean) / x_std
    ytr = (y_tr - y_mean) / y_std

    m = ModelClass(**params)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte) * y_std + y_mean
    return metrics(y_te, pred)


def fit_yeh(X_tr, y_tr):
    def fc_model(X, a, b, e, d):
        AGE = np.maximum(X[:, 0], 1e-6)
        wb  = X[:, WB_INDEX]
        return (a * np.log(AGE) + b) * (e * AGE ** d) ** (-wb)
    try:
        params, _ = curve_fit(fc_model, X_tr, y_tr,
                              p0=[1., 1., 1., 1.], maxfev=10000)
    except Exception:
        params = [40.5, 15.3, 6.49, 0.36]
    return params


def main():
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)

    df['mix_id'] = df.groupby(MIX_COLS).ngroup()
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df['mix_id'].values

    print(f"Total samples: {len(df)}  |  Unique mixes: {df['mix_id'].nunique()}  |  Folds: {N_FOLDS}")

    # Fixed sklearn hyperparams (from paper)
    xgb_params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1,
                  'subsample': 0.8, 'colsample_bytree': 0.8}
    rf_params  = {'n_estimators': 300, 'max_depth': 20,
                  'min_samples_split': 2, 'min_samples_leaf': 1}
    svr_params = {'C': 100, 'epsilon': 0.1, 'kernel': 'rbf', 'gamma': 'scale'}
    knn_params = {'n_neighbors': 9, 'weights': 'distance', 'metric': 'euclidean'}

    gkf = GroupKFold(n_splits=N_FOLDS)
    records = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        n_tr_mix = len(set(groups[tr_idx]))
        n_te_mix = len(set(groups[te_idx]))
        print(f"\n=== Fold {fold}/{N_FOLDS}  (train: {len(X_tr)} samples / {n_tr_mix} mixes | "
              f"test: {len(X_te)} samples / {n_te_mix} mixes) ===")

        a, b, e, d = fit_yeh(X_tr, y_tr)

        for name, fn, kwargs in [
            ('KINN',  run_kinn,    dict(a=a, b=b, e=e, d=d)),
            ('ANN',   run_ann,     {}),
            ('XGB',   run_sklearn, dict(ModelClass=xgb.XGBRegressor, params=xgb_params)),
            ('RF',    run_sklearn, dict(ModelClass=RandomForestRegressor, params=rf_params)),
            ('SVR',   run_sklearn, dict(ModelClass=SVR, params=svr_params)),
            ('KNN',   run_sklearn, dict(ModelClass=KNeighborsRegressor, params=knn_params)),
        ]:
            if name in ('KINN',):
                r2, rmse, mae = fn(X_tr, y_tr, X_te, y_te, **kwargs)
            elif name == 'ANN':
                r2, rmse, mae = fn(X_tr, y_tr, X_te, y_te)
            else:
                r2, rmse, mae = fn(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, **kwargs)
            records.append({'Fold': fold, 'Model': name,
                            'R2': r2, 'RMSE': rmse, 'MAE': mae})
            print(f"  {name:6s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    df_res = pd.DataFrame(records)
    df_res.to_excel('grouped_cv_results.xlsx', index=False)

    print("\n\n========== Summary (mean ± std across 5 folds) ==========")
    summary = df_res.groupby('Model')[['R2', 'RMSE', 'MAE']].agg(['mean', 'std'])
    summary.columns = ['R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std']
    summary = summary.loc[['KINN', 'ANN', 'XGB', 'RF', 'SVR', 'KNN']]
    print(summary.to_string())
    summary.to_excel('grouped_cv_summary.xlsx')
    print("\nSaved to grouped_cv_results.xlsx and grouped_cv_summary.xlsx")


if __name__ == '__main__':
    main()
