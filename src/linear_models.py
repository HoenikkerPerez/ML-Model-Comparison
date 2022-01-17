import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model._base import LinearRegression
from sklearn.linear_model._coordinate_descent import Lasso
from sklearn.linear_model._ridge import Ridge
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._validation import cross_val_score, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.preprocessing._polynomial import PolynomialFeatures, SplineTransformer
from sklearn.utils._testing import ignore_warnings

from src.utils.preprocessing import mean_euclidian_error_loss

warnings.simplefilter("ignore")

def accuracy_scorer(y_true, pred_y):
    y_true = np.where(y_true > 0, 1, 0)
    pred_y = np.where(pred_y > 0, 1, 0)
    accuracy = (pred_y == y_true).sum() / pred_y.shape[0]
    return accuracy



def linear_model_selection(X_train, y_train):
    print("--------- LINEAR REGRESSION ---------")
    # Hyperparameters
    pipe_lm = Pipeline([('scl', StandardScaler()),
                        ('reg', MultiOutputRegressor(LinearRegression()))])

    pipe_lm.fit(X_train, y_train)
    scores = cross_validate(pipe_lm, X_train, y_train, cv=5, 
                            scoring=make_scorer(mean_euclidian_error_loss, greater_is_better=False),
                            return_train_score=True)
    print("TRAIN SCORE\t\t\tVALIDATION SCORE")
    best_mean_test = scores['test_score'].mean()
    best_std_test = scores['test_score'].std()
    best_mean_train = scores['train_score'].mean()
    best_std_train = scores['train_score'].std()
    print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
        abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))

    return pipe_lm



@ignore_warnings(category=ConvergenceWarning)
def linear_lbe_regularized_model_selection(X_train, y_train, lbe="poly"):
    print("--------- LBE + LINEAR REGRESSION ---------")
    # Hyperparameters
    pipe = Pipeline([('scl', StandardScaler()),
                     ('lbe', PolynomialFeatures()),
                     ('reg', MultiOutputRegressor(LinearRegression()))])

    degrees = np.arange(1, 6)
    # degrees = list(zip([1,2,3,4], [2,3,4,5]))

    param_grid = [{
        # 'lbe': [PolynomialFeatures()],
        'lbe__degree': degrees}]

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
    print("TRAIN SCORE\t\t\tVALIDATION SCORE")
    best_mean_test = gs.cv_results_["mean_test_score"][gs.best_index_]
    best_std_test = gs.cv_results_["std_test_score"][gs.best_index_]
    best_mean_train = gs.cv_results_["mean_train_score"][gs.best_index_]
    best_std_train = gs.cv_results_["std_train_score"][gs.best_index_]
    print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
    abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))


    return gs
    # class GSWrapper: pass
    # GSWrapper.best_estimator_ = pipe
    # return GSWrapper

def LASSO_plot_coefficients(X_train, y_train, mode="regression"):
    # PLOT COEFFICIENTS Note that if alpha value is too large the penalty will be too large and hence none of
    # the coefficients can be non-zero. If the penalty is too small you will overfit the model and this will not
    # be the best cross validated solution
    n_alphas = 200
    alphas = np.logspace(-5, 3, n_alphas)
    coefs = []
    for a in alphas:
        ridge = Lasso(alpha=a, fit_intercept=False)
        ridge.fit(X_train, y_train[:, 1])
        coefs.append(ridge.coef_)
    # Display results
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Lasso coefficients as a function of alpha")
    plt.axis("tight")
    plt.savefig('results/images/lasso_coeff.png', bbox_inches='tight')

    plt.show()

def LASSO_model_selection(X_train, y_train, mode="regression"):
    # Hyperparameters
    if mode == "regression":
        print("--------- LASSO REGRESSION MODEL SELECTION ---------")
        n_alphas = 200
        alpha_range = np.logspace(-6, 3, n_alphas)
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
        print("TRAIN SCORE\t\t\tVALIDATION SCORE")
        best_mean_test = gs.cv_results_["mean_test_score"][gs.best_index_]
        best_std_test = gs.cv_results_["std_test_score"][gs.best_index_]
        best_mean_train = gs.cv_results_["mean_train_score"][gs.best_index_]
        best_std_train= gs.cv_results_["std_train_score"][gs.best_index_]
        print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
        return gs

    elif mode == "classifier":
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

def RIDGE_plot_coefficients(X_train, y_train, mode="regression"):
    # PLOT COEFFICIENTS Note that if alpha value is too large the penalty will be too large and hence none of
    # the coefficients can be non-zero. If the penalty is too small you will overfit the model and this will not
    # be the best cross validated solution
    n_alphas = 200
    alphas = np.logspace(-4, 6, n_alphas)
    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X_train, y_train[:, 1])
        coefs.append(ridge.coef_)
    # Display results
    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Ridge coefficients as a function of alpha")
    plt.axis("tight")
    plt.savefig('results/images/ridge_coeff.png', bbox_inches='tight')
    plt.show()

def RIDGE_model_selection(X_train, y_train, mode="regression"):
    if mode == "regression":
        print("--------- RIDGE REGRESSION MODEL SELECTION ---------")
        # Hyperparameters
        n_alphas = 200
        alpha_range = np.logspace(-6, 6, n_alphas)
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
        best_mean_test = gs.cv_results_["mean_test_score"][gs.best_index_]
        best_std_test = gs.cv_results_["std_test_score"][gs.best_index_]
        best_mean_train = gs.cv_results_["mean_train_score"][gs.best_index_]
        best_std_train = gs.cv_results_["std_train_score"][gs.best_index_]
        print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
        abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
        return gs
    elif mode == "classifier":
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

def linear_lbe_reg_plot_coefficients(X_train, y_train, mode="regression"):
    # PLOT COEFFICIENTS Note that if alpha value is too large the penalty will be too large and hence none of
    # the coefficients can be non-zero. If the penalty is too small you will overfit the model and this will not
    # be the best cross validated solution

    # So, ridge regression shrinks the coefficients and it helps to reduce the model complexity and multi-collinearity
    # So Lasso regression not only helps in reducing over-fitting but it can help us in feature selection
    n_alphas = 200
    alphas = np.logspace(-5, 2, n_alphas)
    coefs = []
    for a in alphas:
        ridge = Pipeline([('scl', StandardScaler()),
                         ('lbe', PolynomialFeatures(degree=4)),
                         ('reg', Lasso(alpha=a))])
        ridge.fit(X_train, y_train[:, 1])
        coefs.append(ridge["reg"].coef_)
        # INPUT: 1181->1001 with all poly terms
        # INPUT: 1181->386 with only interaction terms [interaction_only=True)]
        print("#INPUT: {}->{}".format(X_train.shape[1], len(ridge["reg"].coef_)))
    # Display results
    ax = plt.gca()

    ax.plot(alphas, coefs)

    ax.set_xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("weights")
    plt.title("Linear coefficients as a function of alpha")
    plt.axis("tight")
    plt.savefig('results/images/linear_lbe_reg_coeff.png', bbox_inches='tight')
    plt.show()


@ignore_warnings(category=ConvergenceWarning)
def linear_lbe_reg_model_selection(X_train, y_train, mode="regression"):
    class PlaceholderTransformer(BaseEstimator, TransformerMixin):
        def fit(self): pass

        def transform(self): pass

    class PlaceholderEstimator(BaseEstimator, TransformerMixin):
        def fit(self): pass

        def score(self): pass

    if mode == "regression":
        print("--------- LBE+LINEAR REGRESSION MODEL SELECTION ---------")

        pipe = Pipeline([('scl', StandardScaler()),
                         ('lbe', PolynomialFeatures()),
                         ('reg', MultiOutputRegressor(Lasso()))])

        regressor = [MultiOutputRegressor(Ridge()), MultiOutputRegressor(Lasso())]
        transforms = [PolynomialFeatures(), SplineTransformer()]
        degrees = np.arange(2, 6)
        n_knots = np.arange(2, 5)
        n_alpha = 10
        alpha_range = np.logspace(-4, 0, n_alpha)

        # # TODO remove
        # n_alpha = 3
        # degrees = np.arange(2, 4)
        # alpha_range = np.logspace(-4, 0, n_alpha)
        # # TODO endremove
        # param_grid = [
        #     {'lbe': [PolynomialFeatures()], 'lbe__degree': degrees, 'reg__estimator__alpha': alpha_range,
        #      'reg': regressor},
        #     {'lbe': [SplineTransformer()], 'lbe__degree': degrees, 'lbe__n_knots': n_knots,
        #      'reg__estimator__alpha': alpha_range, 'reg': regressor}]
        param_grid = {'lbe__degree': degrees, 'reg__estimator__alpha': alpha_range}
        # Gridsearch
        scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
        gs = GridSearchCV(estimator=pipe,
                          cv=5,
                          param_grid=param_grid,
                          scoring=scorer,
                          verbose=1,
                          n_jobs=-1,
                          return_train_score=True)

        gs.fit(X_train, y_train)
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        best_mean_test = gs.cv_results_["mean_test_score"][gs.best_index_]
        best_std_test = gs.cv_results_["std_test_score"][gs.best_index_]
        best_mean_train = gs.cv_results_["mean_train_score"][gs.best_index_]
        best_std_train = gs.cv_results_["std_train_score"][gs.best_index_]
        print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
            abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
        return gs
    elif mode == "classifier":
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
    print()
    return mee
