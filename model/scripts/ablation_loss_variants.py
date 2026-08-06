"""
Ablation study: three loss function variants for KINN.

Variant A (Ours): normalized L1 physics residual — Eq.(15) corrected
    L_knowledge = mean( |r^k| * sqrt(L_data / mean(r^2)) )

Variant B (Unnormalized): raw L1 physics residual without any scale factor
    L_knowledge = mean( |r^k| )

Variant C (Static MSE): statically weighted physics MSE, no normalization
    L_knowledge = mean( (r^k)^2 )

All three variants use lambda=0.5, same fixed split (random_state=42),
and 100 runs with different random weight initializations.

Outputs:
  ablation_results.xlsx   — Test R2 for each variant, all 100 runs
  ablation_summary.png    — Boxplot comparison

Run from the model/ directory:
    python scripts/ablation_loss_variants.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import RegressionModel, FEATURES, TARGET, DATA_FILE

# ── Settings ──────────────────────────────────────────────────────────────────
LAMBDA_W     = 0.5       # fixed lambda for all variants
NEURONS      = 52
EPOCHS       = 300
BATCH_SIZE   = 8
LR           = 0.0007652946585818769
TRAIN_SIZE   = 0.85
RANDOM_STATE = 42
N_RUNS       = 100
WB_INDEX     = 14


# ── Loss variants ─────────────────────────────────────────────────────────────
def loss_variant_a(outputs, targets, inputs, a, b, e, d):
    """Variant A (Ours): normalized L1 residual.
    L_knowledge = mean(|r| * sqrt(L_data / mean(r^2)))
    No stop-gradient; normalization factor is in the computational graph.
    """
    mse = nn.MSELoss()(outputs, targets)
    AGE = torch.clamp(inputs[:, 0], min=1e-6)
    wb  = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)

    r = torch.abs(outputs - fc_phys.unsqueeze(1))
    r = torch.nan_to_num(r, nan=0.0, posinf=1e10, neginf=-1e10)

    mean_sq = torch.mean(r ** 2)
    r_norm  = r * torch.sqrt(mse / (mean_sq + 1e-8))

    return (1 - LAMBDA_W) * mse + LAMBDA_W * torch.mean(r_norm)


def loss_variant_b(outputs, targets, inputs, a, b, e, d):
    """Variant B: unnormalized L1 residual.
    L_knowledge = mean(|r|), no scale factor.
    """
    mse = nn.MSELoss()(outputs, targets)
    AGE = torch.clamp(inputs[:, 0], min=1e-6)
    wb  = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)

    r = torch.abs(outputs - fc_phys.unsqueeze(1))
    r = torch.nan_to_num(r, nan=0.0, posinf=1e10, neginf=-1e10)

    return (1 - LAMBDA_W) * mse + LAMBDA_W * torch.mean(r)


def loss_variant_c(outputs, targets, inputs, a, b, e, d):
    """Variant C: statically weighted physics MSE (no normalization).
    L_knowledge = mean(r^2), conventional approach.
    """
    mse = nn.MSELoss()(outputs, targets)
    AGE = torch.clamp(inputs[:, 0], min=1e-6)
    wb  = inputs[:, WB_INDEX]
    fc_phys = (a * torch.log(AGE) + b) * (e * torch.pow(AGE, d)) ** (-wb)

    r = outputs - fc_phys.unsqueeze(1)
    r = torch.nan_to_num(r, nan=0.0, posinf=1e10, neginf=-1e10)
    mse_phys = torch.mean(r ** 2)

    return (1 - LAMBDA_W) * mse + LAMBDA_W * mse_phys


VARIANTS = {
    'A_Normalized_L1 (Ours)':  loss_variant_a,
    'B_Unnormalized_L1':       loss_variant_b,
    'C_Static_MSE':            loss_variant_c,
}


# ── Training helper ───────────────────────────────────────────────────────────
def train_one_run(model, optimizer, Xtrain, ytrain, loss_fn, a, b, e, d):
    dataset    = torch.utils.data.TensorDataset(Xtrain, ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb, xb, a, b, e, d)
            loss.backward()
            optimizer.step()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X_raw = df[FEATURES].values
    y_raw = df[TARGET].values

    # 2. Fit empirical equation parameters
    def fc_model(X, a, b, e, d):
        AGE = X[:, 0]
        wb  = X[:, WB_INDEX]
        return (a * np.log(AGE) + b) * (e * AGE ** d) ** (-wb)

    params, _ = curve_fit(fc_model, X_raw, y_raw, p0=[1.0, 1.0, 1.0, 1.0])
    a, b, e, d = params
    print(f"Empirical params: a={a:.4f}, b={b:.4f}, e={e:.4f}, d={d:.4f}")

    # 3. Scale
    scaler   = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw.reshape(-1, 1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).view(-1, 1)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_tensor, y_tensor, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )

    # 4. Run ablation
    all_records = []

    for variant_name, loss_fn in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"{'='*60}")
        r2_list = []

        for run in range(1, N_RUNS + 1):
            torch.manual_seed(run)
            model     = RegressionModel(Xtrain.shape[1], NEURONS)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            train_one_run(model, optimizer, Xtrain, ytrain, loss_fn, a, b, e, d)

            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(Xtest).numpy()
                y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
                y_true = y_scaler.inverse_transform(ytest.numpy()).flatten()

            r2   = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            mae  = mean_absolute_error(y_true, y_pred)
            r2_list.append(r2)

            all_records.append({
                'Variant': variant_name,
                'Run': run,
                'Test_R2': r2,
                'RMSE': rmse,
                'MAE': mae,
            })
            if run % 20 == 0:
                print(f"  Run {run:3d}: R2={r2:.4f}")

        print(f"  >> Mean R2 = {np.mean(r2_list):.4f}  Std = {np.std(r2_list):.4f}")

    # 5. Save results
    df_out = pd.DataFrame(all_records)
    df_out.to_excel('ablation_results.xlsx', index=False)
    print("\nResults saved to ablation_results.xlsx")

    # 6. Summary statistics
    print("\n--- Summary ---")
    summary = df_out.groupby('Variant')['Test_R2'].agg(['mean', 'std', 'min', 'max'])
    print(summary.to_string())

    # 7. Boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    variant_labels = list(VARIANTS.keys())
    data_by_variant = [
        df_out[df_out['Variant'] == v]['Test_R2'].values
        for v in variant_labels
    ]
    bp = ax.boxplot(data_by_variant, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    colors = ['#4C9BE8', '#E8844C', '#4CE884']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(variant_labels) + 1))
    ax.set_xticklabels(
        ['A: Normalized L1\n(Ours)', 'B: Unnormalized L1', 'C: Static MSE'],
        fontsize=11
    )
    ax.set_ylabel('Test R²', fontsize=13)
    ax.set_title('Ablation: Knowledge Loss Variants (100 runs each, λ=0.5)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('ablation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Boxplot saved to ablation_summary.png")


if __name__ == '__main__':
    main()
