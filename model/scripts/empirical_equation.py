"""
Yeh's empirical equation fitted 100 times with different random train/test splits.

Equation: fc = (a·log(AGE) + b) · (e·AGE^d)^(−w/b)
Parameters a, b, e, d are fitted separately for each split via curve_fit.

Output: empirical_equation_results.xlsx

Run from the model/ directory:
    python scripts/empirical_equation.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from models import FEATURES, TARGET, DATA_FILE

WB_INDEX = 14   # column index of w/b in FEATURES


def fc_model(X, a, b, e, d):
    AGE = np.clip(X[:, 0], 1e-6, None)   # guard against log(0)
    wb  = X[:, WB_INDEX]
    return (a * np.log(AGE) + b) * (e * np.power(AGE, d)) ** (-wb)


def main():
    df = pd.read_excel(DATA_FILE, sheet_name='Sheet1')
    df.dropna(inplace=True)
    X = df[FEATURES].values
    y = df[TARGET].values

    records = []
    for run in range(1, 101):
        print(f"Run {run}/100")
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, y, train_size=0.85, random_state=run
        )
        try:
            popt, _ = curve_fit(fc_model, Xtrain, ytrain, p0=[1.0, 1.0, 1.0, 1.0])
            a, b, e, d = popt

            y_train_pred = fc_model(Xtrain, a, b, e, d)
            y_test_pred  = fc_model(Xtest,  a, b, e, d)

            # Skip runs where fitting diverged
            if not (np.all(np.isfinite(y_train_pred)) and np.all(np.isfinite(y_test_pred))):
                print(f"  Skipped: non-finite predictions")
                continue

            train_r2 = r2_score(ytrain, y_train_pred)
            test_r2  = r2_score(ytest,  y_test_pred)
            rmse     = np.sqrt(mean_squared_error(ytest, y_test_pred))
            mae      = mean_absolute_error(ytest, y_test_pred)

            records.append({
                'Run': run, 'a': a, 'b': b, 'e': e, 'd': d,
                'Train R2': train_r2, 'Test R2': test_r2,
                'RMSE': rmse, 'MAE': mae,
            })
            print(f"  Test R2={test_r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        except Exception as ex:
            print(f"  Skipped: {ex}")

    pd.DataFrame(records).to_excel('empirical_equation_results.xlsx', index=False)
    print(f"\nDone. {len(records)} successful runs saved to empirical_equation_results.xlsx")


if __name__ == '__main__':
    main()
