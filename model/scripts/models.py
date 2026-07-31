"""
Shared model definitions: KINN (RegressionModel) and ANN (ANNModel).
Both have identical architecture; KINN uses a physics-informed loss, ANN uses plain MSE.
"""
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """Used for KINN training (custom physics loss) and KINN SHAP analysis."""
    def __init__(self, input_dim: int, neurons: int = 52):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANNModel(nn.Module):
    """Used for standard ANN training (MSE loss) and ANN SHAP analysis."""
    def __init__(self, input_dim: int, neurons: int = 52):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, neurons),
            nn.ReLU(),
            nn.Linear(neurons, neurons),
            nn.ReLU(),
            nn.Linear(neurons, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Shared constants ──────────────────────────────────────────────────────────

FEATURES = [
    'AGE', 'PC', 'PC_TYPE', 'FA', 'SS', 'SF',
    'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC',
    'VOID', 'w/b', 'b/a', 'CAGG%', 'FAGG%', 'FA%', 'SS%', 'SF%',
]
TARGET = 'fc (MPa)'
DATA_FILE = 'updated_fc_predictions.xlsx'

# Fitted Yeh empirical equation parameters (from full-dataset curve_fit)
# fc = (a*log(AGE) + b) * (e * AGE^d)^(-w/b)
# Index of w/b in FEATURES is 14 (0-based), raw-data column index is also 14
A_FIT = 40.49665408379565
B_FIT = 15.292027565616022
E_FIT = 6.49049168852657
D_FIT = 0.36000933143908614
