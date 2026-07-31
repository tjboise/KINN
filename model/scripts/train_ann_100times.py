"""
Train a standard ANN 100 times (randomised train/test split each run).
Outputs: ANN_best_trial_results.xlsx
         ANN_best_trial/  (100 saved .pth files)

Run from the model/ directory:
    python scripts/train_ann_100times.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import ANNModel, FEATURES, TARGET, DATA_FILE

# ── Hyperparameters ───────────────────────────────────────────────────────────
LAYERS      = 2
NEURONS     = 52
LEARN_RATE  = 0.0007652946585818769
EPOCHS      = 300
TRAIN_SIZE  = 0.85
N_RUNS      = 100
MODEL_DIR   = 'ANN_best_trial'
OUTPUT_FILE = 'ANN_best_trial_results.xlsx'


def main():
    # 1. Load and normalise data
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X = df[FEATURES].values
    Y = df[TARGET].values

    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(), Y.std()
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    os.makedirs(MODEL_DIR, exist_ok=True)
    records = []

    for run in range(1, N_RUNS + 1):
        print(f"Run {run}/{N_RUNS}")

        # Different split each run (random_state=run)
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X_norm, Y_norm, train_size=TRAIN_SIZE, random_state=run
        )
        Xtrain = torch.FloatTensor(Xtrain)
        Xtest  = torch.FloatTensor(Xtest)
        ytrain = torch.FloatTensor(ytrain).view(-1, 1)
        ytest  = torch.FloatTensor(ytest).view(-1, 1)

        # 2. Build and train
        model     = ANNModel(Xtrain.shape[1], NEURONS)
        optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(Xtrain), ytrain)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'ann_model_run_{run}.pth'))

        # 3. Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = model(Xtrain).cpu().numpy()
            test_pred  = model(Xtest).cpu().numpy()

        train_r2 = r2_score(ytrain.numpy(), train_pred)
        test_r2  = r2_score(ytest.numpy(),  test_pred)
        rmse     = mean_squared_error(ytest.numpy(), test_pred) ** 0.5
        mae      = mean_absolute_error(ytest.numpy(), test_pred)

        records.append({
            'Run': run, 'Test R2': test_r2, 'Train R2': train_r2,
            'RMSE': rmse, 'MAE': mae,
        })
        print(f"  Test R2={test_r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    pd.DataFrame(records).to_excel(OUTPUT_FILE, index=False)
    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
