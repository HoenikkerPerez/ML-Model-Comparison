import numpy as np
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection._search import GridSearchCV
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm._classes import SVR

from src.utils.plots import plot_search_results
from src.utils.preprocessing import mean_euclidian_error_loss


def srv_model_selection(X_train, y_train):
    coarse_gs = srv_gridsearch(X_train, y_train)
    gamma, C = coarse_gs.best_params_['reg__base_estimator__C'], coarse_gs.best_params_['reg__base_estimator__gamma']
    plot_search_results(coarse_gs, "SRV parameters")
    finer_gs = srv_gridsearch(X_train, y_train, is_fine_search=True, coarse_gamma=gamma, coarse_C=C)
    return finer_gs


def srv_gridsearch(X_train, y_train, is_fine_search=False, coarse_gamma=None, coarse_C=None):
    if is_fine_search:
        gamma_range = np.linspace(coarse_gamma / 10., coarse_gamma * 10, num=21)
        C_range = np.linspace(coarse_C / 10., coarse_C * 10., num=21)
        gamma_range = np.linspace(coarse_gamma / 10., coarse_gamma * 10, num=3)
        C_range = np.linspace(coarse_C / 10., coarse_C * 10., num=3)
    else:
        gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
        C_range = 10 ** np.arange(-3, 6, step=1, dtype=float)

        gamma_range = 10 ** np.arange(-4, 1, step=1, dtype=float)
        C_range = 10 ** np.arange(-1, 1, step=1, dtype=float)

    pipe_svr = Pipeline([('scl', StandardScaler()),
                         ('reg', RegressorChain(SVR(kernel="rbf")))])
    tuned_parameters = {'reg__base_estimator__gamma': gamma_range,
                        'reg__base_estimator__C': C_range}
    # Coarse Gridsearch
    scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
    gs = GridSearchCV(estimator=pipe_svr,
                      cv=5,
                      param_grid=tuned_parameters,
                      scoring=scorer,
                      verbose=1,
                      n_jobs=-1,
                      return_train_score=True)
    gs.fit(X_train, y_train)
    for param in gs.best_params_:
        print(param + ": " + str(gs.best_params_[param]))
    print("MEE: %0.2f" % abs(gs.best_score_))
    return gs
