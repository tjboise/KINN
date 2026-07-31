"""
KNN trained 100 times with randomised train/test splits.

Step 1: GridSearchCV to find best hyperparameters.
Step 2: 100 independent runs using best params.

Best params found during original paper work:
  n_neighbors=9, weights='distance', metric='euclidean'

Outputs:
  KNN_best_trial_results.xlsx
  KNN_best_trial/  (100 saved .joblib files)

Run from the model/ directory:
    python scripts/train_knn_100times.py
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import FEATURES, TARGET, DATA_FILE

TRAIN_SIZE = 0.85
N_RUNS     = 100
MODEL_DIR  = 'KNN_best_trial'
OUTPUT     = 'KNN_best_trial_results.xlsx'


def main():
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X = df[FEATURES].values
    Y = df[TARGET].values

    x_mean, x_std = X.mean(0), X.std(0)
    X_norm = (X - x_mean) / x_std
    y_mean, y_std = Y.mean(), Y.std()
    Y_norm = (Y - y_mean) / y_std

    # Step 1: hyperparameter search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights':     ['uniform', 'distance'],
        'metric':      ['euclidean', 'manhattan'],
    }
    print("Running GridSearchCV for KNN hyperparameters...")
    search = GridSearchCV(
        KNeighborsRegressor(), param_grid,
        cv=5, scoring='r2', n_jobs=-1
    )
    search.fit(X_norm, Y_norm)
    best_params = search.best_params_
    print(f"Best params: {best_params}  (CV R2={search.best_score_:.4f})")

    # Step 2: 100 runs
    os.makedirs(MODEL_DIR, exist_ok=True)
    records = []

    for run in range(1, N_RUNS + 1):
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X_norm, Y_norm, train_size=TRAIN_SIZE, random_state=run
        )
        model = KNeighborsRegressor(**best_params)
        model.fit(Xtrain, ytrain)
        joblib.dump(model, os.path.join(MODEL_DIR, f'knn_model_run_{run}.joblib'))

        ytrain_pred = model.predict(Xtrain)
        ytest_pred  = model.predict(Xtest)

        train_r2 = r2_score(ytrain, ytrain_pred)
        test_r2  = r2_score(ytest,  ytest_pred)
        rmse     = mean_squared_error(ytest, ytest_pred) ** 0.5
        mae      = mean_absolute_error(ytest, ytest_pred)

        records.append({'Run': run, 'Test R2': test_r2, 'Train R2': train_r2,
                        'RMSE': rmse, 'MAE': mae})
        print(f"Run {run:3d}: Test R2={test_r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    pd.DataFrame(records).to_excel(OUTPUT, index=False)
    print(f"\nDone. Results saved to {OUTPUT}")


if __name__ == '__main__':
    main()
