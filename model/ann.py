
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import optuna


df = pd.read_excel('Concrete Database.xlsx', sheet_name='Sheet1')
df.dropna(inplace=True)

Y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values

x_mean = X.mean(0)
x_std = X.std(0)
X_normal = (X - x_mean) / x_std

y_mean = Y.mean()
y_std = Y.std()
Y_normal = (Y - y_mean) / y_std

Xtrain, Xtest, ytrain, ytest = train_test_split(X_normal, Y_normal, train_size=0.85, random_state=42)


def create_model(trial):

    layers = trial.suggest_int('layers', 1, 3)
    neurons = trial.suggest_int('neurons', 8, 64)
    #activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    learn_rate = trial.suggest_loguniform('learn_rate', 1e-4, 1e-1)

    model = Sequential()
    model.add(Dense(neurons, input_dim=X.shape[1], activation='relu', kernel_initializer='he_uniform'))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))

    optimizer = tf.keras.optimizers.Adam(learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def objective(trial):
    model = create_model(trial)
    estimator = KerasRegressor(build_fn=lambda: model, epochs=100, batch_size=8, verbose=0)
    estimator.fit(Xtrain, ytrain, verbose=0)

    y_pred = estimator.predict(Xtest)
    return mean_squared_error(ytest, y_pred)



study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)


best_trial = study.best_trial
print('Best trial:', best_trial.params)


best_model = create_model(best_trial)

best_model.fit(Xtrain, ytrain, epochs=100, batch_size=8, verbose=0)

best_model.save('best_model.h5')

ytrain_pred = pd.DataFrame(best_model.predict(Xtrain))
ytest_pred = pd.DataFrame(best_model.predict(Xtest))

print(r2_score(ytrain, ytrain_pred))
print(r2_score(ytest, ytest_pred))

R2=metrics.r2_score(ytest,ytest_pred)
RMSE=metrics.mean_squared_error(ytest,ytest_pred)**0.5
MAE=metrics.mean_absolute_error(ytest,ytest_pred)
print('R-squared is {0}, RMSE is {1}, and MAE is {2}.'.format(R2,
                                                              RMSE,
                                                              MAE))