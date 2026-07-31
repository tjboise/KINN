import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt; plt.style.use('seaborn')
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

filename = 'Concrete Database.xlsx'
sheetname = 'Sheet1'
dataset = pd.read_excel(filename, sheetname, header=0)
X= dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
Xtrain_column_name=list(Xtrain.columns)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = 12)
rf_random.fit(Xtrain, ytrain)
rf_random.best_params_
rf_model = rf_random.best_estimator_


# Predict test set data
random_forest_predict=rf_model.predict(Xtest)

# Verify the accuracy
random_forest_R2=metrics.r2_score(ytest,random_forest_predict)
random_forest_RMSE=metrics.mean_squared_error(ytest,random_forest_predict)**0.5
random_forest_MAE=metrics.mean_absolute_error(ytest,random_forest_predict)
print('R-squared is {0}, RMSE is {1}, and MAE is {2}.'.format(random_forest_R2,
                                                              random_forest_RMSE,
                                                              random_forest_MAE))
