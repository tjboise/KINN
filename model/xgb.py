import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

filename = 'Concrete Database.xlsx'
sheetname = 'Sheet1'
dataset = pd.read_excel(filename, sheetname, header=0)
X= dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_column_name=list(X_train.columns)

param_grid = {
    'max_depth': np.arange(3, 10, 1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'gamma': np.arange(0, 0.5, 0.1),
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'n_estimators': [100, 200, 300, 400, 500]
}

xgb = XGBRegressor(objective='reg:squarederror')

random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=50, scoring='neg_mean_squared_error', cv=3, verbose=3, random_state=42, n_jobs=24)

random_search.fit(X_train, y_train)

best_xgb = random_search.best_estimator_
predictions = best_xgb.predict(X_test)
model=best_xgb

mse = mean_squared_error(y_test, predictions)

print("Best estimator: ", best_xgb)
print("Best parameters: ", random_search.best_params_)
print("Best validation score: ", random_search.best_score_)
print("MSE on test data: ", mse)
print(best_xgb.score(X_train, y_train))
print(best_xgb.score(X_test, y_test))

