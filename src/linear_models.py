import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model._coordinate_descent import Lasso
from sklearn.linear_model._ridge import Ridge, RidgeClassifier
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._split import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.preprocessing._polynomial import PolynomialFeatures, SplineTransformer

from src.utils.io import save_gridsearch_results
from src.utils.plots import plot_search_results
from src.utils.preprocessing import cup_create_df, cup_split_data_target, mean_euclidian_error_loss


def accuracy_scorer(y_true, pred_y):
    y_true = np.where(y_true > 0, 1, 0)
    pred_y = np.where(pred_y > 0, 1, 0)
    accuracy = (pred_y == y_true).sum() / pred_y.shape[0]
    return accuracy


def linear_model_selection(X_train, y_train):
    # Hyperparameters
    pipe_lm = Pipeline([('scl', StandardScaler()),
                        ('reg', MultiOutputRegressor(LinearRegression()))])

    result = pipe_lm.fit(X_train, y_train)

    return pipe_lm


def linear_lbe_regularized_model_selection(X_train, y_train, lbe="poly"):
    class PlaceholderTransformer(BaseEstimator, TransformerMixin):
        def fit(self): pass

        def transform(self): pass

    # Hyperparameters
    pipe = Pipeline([('scl', StandardScaler()),
                     # ('lbe', PlaceholderTransformer()),
                     ('lbe', PolynomialFeatures()),
                     ('reg', MultiOutputRegressor(Ridge()))])

    transforms = [PolynomialFeatures(), SplineTransformer()]
    degrees = np.arange(1, 10)
    n_knots = np.arange(2, 10)
    degrees = np.arange(1, 3)
    n_knots = np.arange(2, 4)
    alpha_range = np.linspace(.00001, .001, 10)

    # param_grid = [{
    #     'lbe': [PolynomialFeatures()],
    #     'lbe__degree': degrees,
    #     'reg__estimator__alpha': alpha_range},
    #     {
    #         'lbe': [SplineTransformer()],
    #         'lbe__degree': degrees,
    #         'lbe__n_knots': n_knots,
    #         'reg__estimator__alpha': alpha_range}]
    param_grid = [{
        # 'lbe': [PolynomialFeatures()],
        'lbe__degree': degrees,
        'reg__estimator__alpha': alpha_range}]

    gs = GridSearchCV(estimator=pipe,
                      cv=5,
                      param_grid=param_grid,
                      # , 'poly__n_knots':n_knots},
                      scoring=make_scorer(mean_euclidian_error_loss, greater_is_better=False),
                      # verbose=1,
                      n_jobs=-1,
                      return_train_score=True)
    gs.fit(X_train, y_train)
    for param in gs.best_params_:
        print(param + ": " + str(gs.best_params_[param]))
    print("MEE: %0.2f" % abs(gs.best_score_))

    return gs


def LASSO_model_selection(X_train, y_train, mode="regression"):
    # Hyperparameters
    if mode == "regression":
        print()
        print("--------- LASSO REGRESSION MODEL SELECTION ---------")
        alpha_range = np.logspace(-4, 0, 5)
        pipe_svr = Pipeline([('scl', StandardScaler()),
                             ('reg', MultiOutputRegressor(Lasso()))])

        tuned_parameters = {'reg__estimator__alpha': alpha_range}

        # Gridsearch
        scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
#         # gs = GridSearchCV(estimator=pipe_svr, cv=5, param_grid=tuned_parameters, scoring=scorer, verbose=1, n_jobs=-1)
        gs = GridSearchCV(estimator=pipe_svr,
                          cv=5,
                          param_grid=tuned_parameters,
                          scoring=scorer,
                          # verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("MEE: %0.2f" % abs(gs.best_score_))
        return gs

    elif mode == "classifier":
        print()
        print("--------- LASSO CLASSIFIER MODEL SELECTION ---------")

        # convert target to -1, +1 and return and then treats the problem as a regression task, optimizing the same
        #       # objective as above. The predicted class corresponds to the sign of the regressor’s prediction.
        alpha_range = np.logspace(-4, 4, 9)
        eta0_range = np.logspace(-4, 0, 5)
        # lr_range = np.logspace(-4, 0, 5)
        # pipe_svr = RidgeClassifier()
        model = SGDClassifier(loss='squared_error', penalty="l1", learning_rate="invscaling")
        tuned_parameters = {'alpha': alpha_range,
                            'eta0': eta0_range}

        # Gridsearch
#         # gs = GridSearchCV(estimator=pipe_svr, cv=5, param_grid=tuned_parameters, scoring=scorer, verbose=1, n_jobs=-1)
        gs = GridSearchCV(estimator=model,
                          cv=5,
                          param_grid=tuned_parameters,
                          scoring="accuracy",
                          # verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("ACCURACY: %0.2f" % abs(gs.best_score_))
        return gs


def RIDGE_model_selection(X_train, y_train, mode="regression"):
    if mode == "regression":
        print()
        print("--------- RIDGE REGRESSION MODEL SELECTION ---------")
        # Hyperparameters
        alpha_range = np.logspace(-4, 0, 5)
        pipe_svr = Pipeline([('scl', StandardScaler()),
                             ('reg', MultiOutputRegressor(Ridge()))])

        tuned_parameters = {'reg__estimator__alpha': alpha_range}
        # Gridsearch
        scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
        gs = GridSearchCV(estimator=pipe_svr,
                          cv=5,
                          param_grid=tuned_parameters,
                          scoring=scorer,
                          # verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("MEE: %0.2f" % abs(gs.best_score_))
        return gs
    elif mode == "classifier":
        print()
        print("--------- RIDGE CLASSIFIER MODEL SELECTION ---------")
        # Hyperparameters
        alpha_range = np.logspace(-4, 4, 9)
        eta0_range = np.logspace(-4, 0, 5)
        # lr_range = np.logspace(-4, 0, 5)
        # pipe_svr = RidgeClassifier()
        pipe_svr = SGDClassifier(loss='squared_error', penalty="l2", learning_rate="invscaling")
        tuned_parameters = {'alpha': alpha_range,
                            'eta0': eta0_range}
        # Gridsearch
        scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
        gs = GridSearchCV(estimator=pipe_svr,
                          cv=5,
                          param_grid=tuned_parameters,
                          scoring="accuracy",
                          # verbose=1,
                          n_jobs=1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("ACCURACY: %0.2f" % abs(gs.best_score_))
        return gs


def linear_lbe_reg_model_selection(X_train, y_train, mode="regression"):
    class PlaceholderTransformer(BaseEstimator, TransformerMixin):
        def fit(self): pass

        def transform(self): pass

    class PlaceholderEstimator(BaseEstimator, TransformerMixin):
        def fit(self): pass

        def score(self): pass

    if mode=="regression":
        print()
        print("--------- LBE+LINEAR REGRESSION MODEL SELECTION ---------")
        # Hyperparameters
        alpha_range = np.linspace(.00001, .001, 100)

        # Hyperparameters
        pipe = Pipeline([('scl', StandardScaler()),
                         ('lbe', PlaceholderTransformer()),
                         ('reg', PlaceholderEstimator())])

        regressor = [MultiOutputRegressor(Ridge()), MultiOutputRegressor(Lasso())]
        transforms = [PolynomialFeatures(), SplineTransformer()]
        degrees = np.arange(1, 5)
        n_knots = np.arange(2, 5)
        alpha_range = np.linspace(.001, .01, 10)
        param_grid = [
            {'lbe': [PolynomialFeatures()], 'lbe__degree': degrees, 'reg__estimator__alpha': alpha_range, 'reg': regressor},
            {'lbe': [SplineTransformer()], 'lbe__degree': degrees, 'lbe__n_knots': n_knots,
             'reg__estimator__alpha': alpha_range, 'reg': regressor}]

        # Gridsearch
        scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
        gs = GridSearchCV(estimator=pipe,
                          cv=5,
                          param_grid=param_grid,
                          scoring=scorer,
                          # verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("MEE: %0.2f" % abs(gs.best_score_))
        return gs
    elif mode=="classifier":
        print()
        print("--------- LBE+LINEAR CLASSIFIER MODEL SELECTION ---------")
        # Hyperparameters
        alpha_range = np.logspace(-4, 4, 9)
        eta0_range = np.logspace(-4, 0, 5)
        optimizer = SGDClassifier(loss='squared_error', penalty="l2", learning_rate="invscaling", max_iter=10000)
        pipe = Pipeline([('lbe', PlaceholderTransformer()),
                         ('reg', optimizer)])
        degrees = np.arange(1, 5)
        n_knots = np.arange(2, 5)
        alpha_range = np.linspace(.001, .01, 10)
        param_grid = [
            {'lbe': [PolynomialFeatures()], 'lbe__degree': degrees, 'reg__alpha': alpha_range, 'reg__eta0': eta0_range},
            {'lbe': [SplineTransformer()], 'lbe__degree': degrees, 'lbe__n_knots': n_knots,
             'reg__alpha': alpha_range, 'reg__eta0': eta0_range}]

        # Gridsearch
        gs = GridSearchCV(estimator=pipe,
                          cv=5,
                          param_grid=param_grid,
                          scoring="accuracy",
                          # verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("ACCURACY: %0.2f" % abs(gs.best_score_))
        return gs


def model_assessment(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mee = mean_euclidian_error_loss(y_pred, y_test)
    print("[Model Assessment] MEE: %0.2f" % abs(mee))
    print("------------------------------------")
    return mee
