"""
Generate Partial Dependence Plots (PDP) from pre-computed CSV files.

Expected input files in the model/ directory:
  pdp_results_7.csv, pdp_results_28.csv, pdp_results_56.csv
  (columns: 'FA% (Original Scale)', 'XGBoost', 'KINN', 'ANN', 'RF', 'SVR')

Outputs: PDP_XGBoost.png, PDP_KINN.png, PDP_ANN.png, PDP_RF.png, PDP_SVR.png

Run from the model/ directory:
    python scripts/pdp_plots.py
"""
import pandas as pd
import matplotlib.pyplot as plt


AGES   = [7, 28, 56]
COLORS = ['blue', 'orange', 'green']
X_COL  = 'FA% (Original Scale)'
XLIM   = (0.15, 0.3)


def load_pdp_csvs():
    csvs = {age: pd.read_csv(f'pdp_results_{age}.csv') for age in AGES}
    return csvs


def plot_model(csvs, model_col: str, xlim=XLIM, save_path: str = None, grid: bool = True):
    plt.figure()
    for age, color in zip(AGES, COLORS):
        plt.plot(csvs[age][X_COL], csvs[age][model_col],
                 label=f'AGE={age}', color=color)
    plt.xlabel('FA%')
    plt.ylabel('Predicted fc')
    plt.legend()
    if grid:
        plt.grid()
    plt.xlim(*xlim)
    plt.tight_layout()
    path = save_path or f'PDP_{model_col}.png'
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def main():
    csvs = load_pdp_csvs()

    # XGBoost — slightly wider x range
    plot_model(csvs, 'XGBoost', xlim=(0, 0.3), save_path='PDP_XGBoost.png', grid=True)

    # KINN (previously labelled SKIRM in CSV files — uses the 'KINN' column)
    plot_model(csvs, 'KINN',    save_path='PDP_KINN.png',    grid=True)
    plot_model(csvs, 'ANN',     save_path='PDP_ANN.png',     grid=True)
    plot_model(csvs, 'RF',      save_path='PDP_RF.png',      grid=True)
    plot_model(csvs, 'SVR',     save_path='PDP_SVR.png',     grid=False)

    print("All PDP plots saved.")


if __name__ == '__main__':
    main()
