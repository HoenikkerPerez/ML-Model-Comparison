import numpy as np
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm._classes import SVR


def ensamble_srv_model_selection(X_train, y_train):
    gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
    C_range = 10 ** np.arange(-3, 6, step=1, dtype=float)
    bagging = BaggingRegressor(RegressorChain(SVR(kernel="rbf", C=1, gamma=0.1)))
    gs = Pipeline([('scl', StandardScaler()), ('reg', bagging)])

    gs.fit(X_train, y_train)
    return gs
