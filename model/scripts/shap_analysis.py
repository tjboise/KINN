"""
SHAP analysis for KINN and ANN.

Trains one instance of each model, then runs SHAP DeepExplainer on the test set.

Outputs:
  KINN: KINN_summary_plot.png, KINN_bar_plot.png
  ANN:  ANN_shap_summary_plot.png, shap_mean_absolute_bar_plot_ranked.png

Run from the model/ directory:
    python scripts/shap_analysis.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from models import RegressionModel, ANNModel, FEATURES, TARGET, DATA_FILE

# ── Shared settings ───────────────────────────────────────────────────────────
NEURONS      = 52
LEARN_RATE   = 0.0007652946585818769
EPOCHS       = 300
BATCH_SIZE   = 8
TRAIN_SIZE   = 0.85
RANDOM_STATE = 42

# Yeh equation parameters (pre-fitted on full dataset)
A, B, E, D = 40.49665408379565, 15.292027565616022, 6.49049168852657, 0.36000933143908614
WB_INDEX    = 14  # index of w/b in FEATURES


# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_normalise():
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X = df[FEATURES].values
    Y = df[TARGET].values

    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(), Y.std()
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_norm, Y_norm, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )
    return Xtrain, Xtest, ytrain, ytest


# ── KINN physics loss ─────────────────────────────────────────────────────────
def kinn_loss(outputs, targets, inputs):
    outputs = outputs.squeeze()
    mse     = nn.MSELoss()(outputs, targets)

    AGE     = torch.clamp(inputs[:, 0], min=1e-6)
    wb      = inputs[:, WB_INDEX]
    fc_phys = (A * torch.log(AGE) + B) * (E * torch.pow(AGE, D)) ** (-wb)

    residual = torch.abs(outputs - fc_phys)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=1e10, neginf=-1e10)

    mean_sq = torch.mean(residual ** 2)
    if mean_sq.item() > 0:
        residual = residual * torch.sqrt(mse / mean_sq)

    return 0.5 * mse + 0.5 * torch.mean(residual)


# ── Generic training ──────────────────────────────────────────────────────────
def train_model(model, optimizer, X_tensor, y_tensor, loss_fn):
    dataset    = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb, xb)
            loss.backward()
            optimizer.step()


# ── SHAP plotting helpers ─────────────────────────────────────────────────────
def plot_shap_summary(shap_values, X_tensor, save_path):
    shap.summary_plot(shap_values, X_tensor.cpu().numpy(),
                      feature_names=FEATURES, show=False)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shap_bar_ranked(shap_values_np, save_path):
    mean_abs = np.abs(shap_values_np).mean(axis=0)
    importance = sorted(zip(FEATURES, mean_abs), key=lambda x: x[1])
    sorted_feats, sorted_vals = zip(*importance)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_feats, sorted_vals, color='skyblue')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}', va='center')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.title('Mean Absolute SHAP Values for Features')
    plt.xlim(0, max(sorted_vals) * 1.1)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


# ── KINN SHAP ─────────────────────────────────────────────────────────────────
def run_kinn_shap():
    print("\n=== KINN SHAP ===")
    Xtrain, Xtest, ytrain, _ = load_and_normalise()

    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xtrain_t     = torch.FloatTensor(Xtrain).to(device)
    ytrain_t     = torch.FloatTensor(ytrain).to(device)
    Xtest_t      = torch.FloatTensor(Xtest).to(device)

    model     = RegressionModel(Xtrain_t.shape[1], NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

    def loss_fn(out, tgt, inp):
        return kinn_loss(out, tgt, inp)

    train_model(model, optimizer, Xtrain_t, ytrain_t, loss_fn)

    model.eval()
    explainer   = shap.DeepExplainer(model, Xtrain_t)
    shap_values = explainer.shap_values(Xtest_t)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    plot_shap_summary(shap_values, Xtest_t, 'KINN_summary_plot.png')

    # Bar chart (shap built-in)
    shap.summary_plot(shap_values, Xtest_t.cpu().numpy(),
                      feature_names=FEATURES, plot_type='bar', show=False)
    plt.savefig('KINN_bar_plot.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: KINN_bar_plot.png")


# ── ANN SHAP ──────────────────────────────────────────────────────────────────
def run_ann_shap():
    print("\n=== ANN SHAP ===")
    Xtrain, Xtest, ytrain, _ = load_and_normalise()

    Xtrain_t = torch.FloatTensor(Xtrain)
    ytrain_t = torch.FloatTensor(ytrain).view(-1, 1)
    Xtest_t  = torch.FloatTensor(Xtest)
    Xtest_df = pd.DataFrame(Xtest, columns=FEATURES)

    model     = ANNModel(Xtrain_t.shape[1], NEURONS)
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = criterion(model(Xtrain_t), ytrain_t)
        loss.backward()
        optimizer.step()

    model.eval()
    explainer   = shap.DeepExplainer(model, Xtrain_t)
    shap_values = explainer.shap_values(Xtest_t)

    if isinstance(shap_values, (list, tuple)):
        shap_values_np = shap_values[0]
    else:
        shap_values_np = shap_values

    # Summary beeswarm plot
    shap.summary_plot(shap_values_np, Xtest_df, feature_names=FEATURES, show=False)
    plt.savefig('ANN_shap_summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: ANN_shap_summary_plot.png")

    # Ranked bar plot
    plot_shap_bar_ranked(shap_values_np, 'shap_mean_absolute_bar_plot_ranked.png')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_kinn_shap()
    run_ann_shap()
    print("\nAll SHAP plots saved.")
