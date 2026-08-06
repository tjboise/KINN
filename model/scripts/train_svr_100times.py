"""
SVR trained 100 times with randomised train/test splits.

Step 1: RandomizedSearchCV to find best hyperparameters.
Step 2: 100 independent runs using best params.

Outputs:
  SVR_best_trial_results.xlsx
  SVR_best_trial/  (100 saved .joblib files)

Run from the model/ directory:
    python scripts/train_svr_100times.py
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import FEATURES, TARGET, DATA_FILE

TRAIN_SIZE = 0.85
N_RUNS     = 100
MODEL_DIR  = 'SVR_best_trial'
OUTPUT     = 'SVR_best_trial_results.xlsx'


def main():
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X = df[FEATURES].values
    Y = df[TARGET].values

    # Global normalisation
    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(),   Y.std()
    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    # Step 1: hyperparameter search on a single split of the normalised data
    X_tr0, _, Y_tr0, _ = train_test_split(X_norm, Y_norm, train_size=TRAIN_SIZE, random_state=0)
    param_grid = {
        'C':              [0.1, 1, 10, 100, 1000],
        'epsilon':        [0.01, 0.1, 0.2, 0.5, 1],
        'kernel':         ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree':         [2, 3, 4],
        'gamma':          ['scale', 'auto'],
    }
    print("Running RandomizedSearchCV for SVR hyperparameters...")
    search = RandomizedSearchCV(
        SVR(), param_grid,
        n_iter=50, scoring='neg_mean_squared_error',
        cv=3, random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_tr0, Y_tr0)
    best_params = search.best_params_
    print(f"Best params: {best_params}")

    # Step 2: 100 runs — different random splits of the globally normalised data
    os.makedirs(MODEL_DIR, exist_ok=True)
    records = []

    for run in range(1, N_RUNS + 1):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_norm, Y_norm, train_size=TRAIN_SIZE, random_state=run
        )

        model = SVR(**best_params)
        model.fit(X_tr, y_tr)
        joblib.dump(model, os.path.join(MODEL_DIR, f'svr_model_run_{run}.joblib'))

        ytrain_pred = model.predict(X_tr)
        ytest_pred  = model.predict(X_te)

        train_r2 = r2_score(y_tr, ytrain_pred)
        test_r2  = r2_score(y_te, ytest_pred)
        rmse     = mean_squared_error(y_te * y_std + y_mean,
                                      ytest_pred * y_std + y_mean) ** 0.5
        mae      = mean_absolute_error(y_te * y_std + y_mean,
                                       ytest_pred * y_std + y_mean)

        records.append({'Run': run, 'Test R2': test_r2, 'Train R2': train_r2,
                        'RMSE': rmse, 'MAE': mae})
        print(f"Run {run:3d}: Test R2={test_r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    pd.DataFrame(records).to_excel(OUTPUT, index=False)
    print(f"\nDone. Results saved to {OUTPUT}")


if __name__ == '__main__':
    main()
