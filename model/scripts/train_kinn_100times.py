"""
Train KINN 100 times with fixed hyperparameters and fixed train/test split.
Outputs: model_evaluation_metrics_with_scaled.xlsx
         KINN_saved_models_scaled/  (100 saved .pt files)

Run from the model/ directory:
    python scripts/train_kinn_100times.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import RegressionModel, FEATURES, TARGET, DATA_FILE

# ── Hyperparameters ───────────────────────────────────────────────────────────
LAYERS       = 2
NEURONS      = 52
LEARN_RATE   = 0.0007652946585818769
EPOCHS       = 300
BATCH_SIZE   = 8
TRAIN_SIZE   = 0.85
RANDOM_STATE = 42
N_RUNS       = 100
MODEL_DIR    = 'KINN_saved_models_scaled'
OUTPUT_FILE  = 'model_evaluation_metrics_with_scaled.xlsx'
WB_INDEX     = 14  # column index of w/b in FEATURES


# ── Physics loss ──────────────────────────────────────────────────────────────
def kinn_loss(outputs, targets, inputs, a, b, e, d):
    mse = nn.MSELoss()(outputs, targets)

    AGE = torch.clamp(inputs[:, 0], min=1e-6)
    wb  = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)

    residual = torch.abs(outputs - fc_phys.unsqueeze(1))
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e10, neginf=-1e10)

    mean_sq = torch.mean(residual ** 2)
    if mean_sq.item() > 0:
        residual = residual * torch.sqrt(mse / mean_sq)

    return 0.5 * mse + 0.5 * torch.mean(residual)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_run(model, optimizer, Xtrain, ytrain, a, b, e, d):
    dataset    = torch.utils.data.TensorDataset(Xtrain, ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = kinn_loss(model(xb), yb, xb, a, b, e, d)
            loss.backward()
            optimizer.step()


def evaluate(model, Xtrain, ytrain, Xtest, ytest, y_scaler):
    model.eval()
    with torch.no_grad():
        train_pred = y_scaler.inverse_transform(model(Xtrain).numpy())
        test_pred  = y_scaler.inverse_transform(model(Xtest).numpy())
    train_true = y_scaler.inverse_transform(ytrain.numpy())
    test_true  = y_scaler.inverse_transform(ytest.numpy())

    y_std      = y_scaler.scale_[0]
    train_r2   = r2_score(train_true, train_pred)
    test_r2    = r2_score(test_true,  test_pred)
    rmse       = mean_squared_error(test_true, test_pred) ** 0.5
    mae        = mean_absolute_error(test_true, test_pred)
    return train_r2, test_r2, rmse, mae, rmse / y_std, mae / y_std


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X_raw = df[FEATURES].values
    y_raw = df[TARGET].values

    # 2. Fit empirical equation to get physical parameters
    def fc_model(X, a, b, e, d):
        AGE = X[:, 0]
        wb  = X[:, WB_INDEX]
        return (a * np.log(AGE) + b) * (e * AGE ** d) ** (-wb)

    params, _ = curve_fit(fc_model, X_raw, y_raw, p0=[1.0, 1.0, 1.0, 1.0])
    a, b, e, d = params
    print(f"Empirical params: a={a:.4f}, b={b:.4f}, e={e:.4f}, d={d:.4f}")

    # 3. Scale features and target
    scaler   = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1))
    y_std    = y_scaler.scale_[0]

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    # 4. Fixed train/test split (variability comes only from random weight init)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_tensor, y_tensor, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # 5. Run 100 times
    os.makedirs(MODEL_DIR, exist_ok=True)
    records = []

    for i in range(N_RUNS):
        print(f"Run {i + 1}/{N_RUNS}")
        model     = RegressionModel(Xtrain.shape[1], NEURONS)
        optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
        train_one_run(model, optimizer, Xtrain, ytrain, a, b, e, d)

        train_r2, test_r2, rmse, mae, s_rmse, s_mae = evaluate(
            model, Xtrain, ytrain, Xtest, ytest, y_scaler
        )
        records.append({
            'Run': i + 1, 'Layers': LAYERS, 'Neurons': NEURONS,
            'Learning Rate': LEARN_RATE,
            'Train R2': train_r2, 'Test R2': test_r2,
            'RMSE': rmse, 'MAE': mae,
            'Scaled RMSE': s_rmse, 'Scaled MAE': s_mae,
        })

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'model_run_{i + 1}.pt'))

    # 6. Save results
    pd.DataFrame(records).to_excel(OUTPUT_FILE, index=False)
    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
