"""
Sensitivity analysis of the KINN loss weighting parameter lambda (λ).

At each λ value, trains one KINN and evaluates using a *hybrid* prediction
strategy: final = (1-λ)*model_prediction + λ*physics_prediction.

Outputs:
  KINN_hybrid_results.xlsx   — Test R² for each λ
  hybrid_sensitivity_plot.png

Run from the model/ directory:
    python scripts/sensitivity_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from models import RegressionModel, FEATURES, TARGET, DATA_FILE

# ── Settings ──────────────────────────────────────────────────────────────────
LAMBDA_CANDIDATES = [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
                     0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NEURONS      = 52
EPOCHS       = 300
BATCH_SIZE   = 8
LR           = 0.000765
TRAIN_SIZE   = 0.85
RANDOM_STATE = 42
WB_INDEX     = 14   # column index of w/b in FEATURES


# ── Loss function ─────────────────────────────────────────────────────────────
def kinn_loss(outputs, targets, inputs, lambda_w, a, b, e, d):
    mse = nn.MSELoss()(outputs, targets)

    AGE     = torch.clamp(inputs[:, 0], min=1e-6)
    wb      = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)

    residual = torch.abs(outputs - fc_phys.unsqueeze(1))
    residual = torch.nan_to_num(residual, nan=0.0)

    mean_sq = torch.mean(residual ** 2)
    residual = residual * torch.sqrt(mse / (mean_sq + 1e-8))

    return (1 - lambda_w) * mse + lambda_w * torch.mean(residual)


def main():
    # 1. Load data
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X_raw = df[FEATURES].values
    y_raw = df[TARGET].values

    # 2. Define empirical equation for curve_fit (fitted on training data in step 3)
    def fc_model_fit(X, a, b, e, d):
        AGE = np.maximum(X[:, 0], 1e-6)
        wb  = X[:, WB_INDEX]
        return (a * np.log(AGE) + b) * (e * np.power(AGE, d)) ** (-wb)

    # 3. Global scalers — fit on all data, then split
    scaler   = StandardScaler()
    y_scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw)
    y_all = y_scaler.fit_transform(y_raw.reshape(-1, 1))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # Fit empirical coefficients on the training partition (raw units)
    X_tr_raw = scaler.inverse_transform(X_tr)
    y_tr_raw = y_scaler.inverse_transform(y_tr).ravel()
    params, _ = curve_fit(fc_model_fit, X_tr_raw, y_tr_raw, p0=[1.0, 1.0, 1.0, 1.0],
                          maxfev=10000)
    a, b, e, d = params
    print(f"Physical params: a={a:.4f}, b={b:.4f}, e={e:.4f}, d={d:.4f}")

    Xtrain = torch.tensor(X_tr, dtype=torch.float32)
    Xtest  = torch.tensor(X_te, dtype=torch.float32)
    ytrain = torch.tensor(y_tr, dtype=torch.float32)
    ytest  = torch.tensor(y_te, dtype=torch.float32)

    # 4. Sensitivity loop
    results = []
    for lam in LAMBDA_CANDIDATES:
        print(f"Training: lambda={lam}")
        model     = RegressionModel(Xtrain.shape[1], NEURONS)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loader    = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtrain, ytrain),
            batch_size=BATCH_SIZE, shuffle=True
        )
        model.train()
        for _ in range(EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = kinn_loss(model(xb), yb, xb, lam, a, b, e, d)
                loss.backward()
                optimizer.step()

        # Hybrid prediction: blend model output with physics equation
        model.eval()
        with torch.no_grad():
            y_pred_scaled  = model(Xtest)
            y_pred_model   = y_scaler.inverse_transform(y_pred_scaled.numpy()).flatten()

        Xtest_raw = scaler.inverse_transform(Xtest.numpy())
        AGE_raw   = np.maximum(Xtest_raw[:, 0], 1e-6)
        wb_raw    = Xtest_raw[:, WB_INDEX]
        y_phys    = (a * np.log(AGE_raw) + b) * (e * np.power(AGE_raw, d)) ** (-wb_raw)

        y_final = (1 - lam) * y_pred_model + lam * y_phys
        y_true  = y_scaler.inverse_transform(ytest.numpy()).flatten()

        r2 = r2_score(y_true, y_final)
        results.append({'Lambda': lam, 'Test_R2': r2})
        print(f"  Test R2 = {r2:.4f}")

    # 5. Save and plot
    results_df = pd.DataFrame(results)
    results_df.to_excel('KINN_hybrid_results.xlsx', index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Lambda'], results_df['Test_R2'],
             marker='o', linestyle='-', linewidth=2)
    plt.title(r'Hybrid Prediction: Impact of $\lambda$ on $R^2$')
    plt.xlabel(r'$\lambda$ (Balance factor)')
    plt.ylabel(r'Test $R^2$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('hybrid_sensitivity_plot.png', dpi=300)
    plt.close()

    print("\nDone. Results saved to KINN_hybrid_results.xlsx and hybrid_sensitivity_plot.png")


if __name__ == '__main__':
    main()
