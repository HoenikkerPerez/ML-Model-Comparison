import numpy as np
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.ensemble._forest import RandomForestRegressor
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection._search import GridSearchCV
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm._classes import SVR

from src.utils.preprocessing import mean_euclidian_error_loss


def random_forest_model_selection(X_train, y_train):
    n_estimators = [1000, 5000]
    max_depths = np.arange(5, 10)
    max_features = np.arange(1, X_train.shape[1])
    r_forest_gs = Pipeline([('scl', StandardScaler()),
                            ('reg', RandomForestRegressor(random_state=42))])

    tuned_parameters = {'reg__n_estimators': n_estimators,
                        'reg__max_depth': max_depths,
                        'reg__max_features': max_features,
                        }

    scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
    # gs = GridSearchCV(estimator=pipe_svr, cv=5, param_grid=tuned_parameters, scoring=scorer, verbose=1, n_jobs=-1)
    gs = GridSearchCV(estimator=r_forest_gs,
                      cv=5,
                      param_grid=tuned_parameters,
                      scoring=scorer,
                      return_train_score=True)

    gs.fit(X_train, y_train)
    for param in gs.best_params_:
        print(param + ": " + str(gs.best_params_[param]))
    print("MEE: %0.2f" % abs(gs.best_score_))