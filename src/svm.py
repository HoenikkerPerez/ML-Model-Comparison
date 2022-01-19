import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics._scorer import make_scorer
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._split import train_test_split
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler, MinMaxScaler
from sklearn.svm._classes import SVR, SVC

from src.utils.io import save_gridsearch_results
from src.utils.plots import plot_search_results, plot_search_heatmap
from src.utils.preprocessing import mean_euclidian_error_loss

DEBUG = False

# CLASSIFICATION
def svc_model_selection(X_train, y_train):
    print("--------- SVM CLASSIFIER MODEL SELECTION ---------")
    coarse_gs = svc_gridsearch(X_train, y_train)
    C, gamma = coarse_gs.best_params_['C'], coarse_gs.best_params_['gamma']
    plot_search_results(coarse_gs, "SVC parameters", vmin=0, vmax=1)
    finer_gs = svc_gridsearch(X_train, y_train, is_fine_search=True, coarse_gamma=gamma, coarse_C=C)
    return finer_gs


def svc_gridsearch(X_train, y_train, is_fine_search=False, coarse_gamma=None, coarse_C=None):
    if DEBUG:
        if is_fine_search:
            gamma_range = np.linspace(coarse_gamma / 10., coarse_gamma * 10, num=3)
            C_range = np.linspace(coarse_C / 10., coarse_C * 10., num=3)
        else:
            gamma_range = 10 ** np.arange(-4, 1, step=1, dtype=float)
            C_range = 10 ** np.arange(-1, 1, step=1, dtype=float)
    else:
        if is_fine_search:
            gamma_range = np.linspace(coarse_gamma / 10., coarse_gamma * 10, num=21)
            C_range = np.linspace(coarse_C / 10., coarse_C * 10., num=21)
        else:
            gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
            C_range = 10 ** np.arange(-3, 6, step=1, dtype=float)

    model = SVC(kernel="rbf")
    # print(model.get_params().keys())
    tuned_parameters = {'gamma': gamma_range,
                        'C': C_range}

    # Coarse Gridsearch
    gs = GridSearchCV(estimator=model,
                      cv=5,
                      param_grid=tuned_parameters,
                      scoring="accuracy",
                      # verbose=3,
                      n_jobs=-1,
                      return_train_score=True)
    gs.fit(X_train, y_train)
    if is_fine_search:
        for param in gs.best_params_:
            print(param + ": " + str(gs.best_params_[param]))
        print("ACCURACY: %0.2f" % abs(gs.best_score_))
        print("---------------------------------------")
    return gs


# REGRESSION
class MultioutputSVR(BaseEstimator, RegressorMixin):
    def __init__(self, svr0, svr1, C0=None, C1=None, gamma0=None, gamma1=None):
        self.gamma1 = gamma1
        self.gamma0 = gamma0
        self.C1 = C1
        self.C0 = C0
        self.svr0 = svr0
        self.svr1 = svr1
        self.outdim = 2

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        # self.svr0 = SVR(kernel='rbf', C=self.C0, gamma=self.gamma0)
        self.svr1 = Pipeline([

            ('reg', SVR(kernel='rbf', C=self.C1, gamma=self.gamma1))])
        # self.svr1 = SVR(kernel='rbf', C=self.C1, gamma=self.gamma1)

    def fit(self,X_train,y_train):
        self.svr0.fit(X_train, y_train[:, 0])
        # pred_0 = self.svr0.predict(X_train)
        # X_train_plusone = np.column_stack((X_train, pred_0))
        self.svr1.fit(X_train, y_train[:, 1])
        return self

    def predict(self, X_test):
        output = np.zeros(shape=(X_test.shape[0], self.outdim))
        pred_0 = self.svr0.predict(X_test)
        output[:,0] = pred_0
        # X_train_plusone = np.column_stack((X_test, pred_0))
        output[:,1] = self.svr1.predict(X_test)
        return output


class MultioutSVR:
    def __init__(self, svrs):
        self.svrs = svrs
        self.outdim = len(svrs)

    def fit(self,X_train,y_train):
        for i in range(self.outdim):
            self.svrs[i].fit(X_train, y_train[:, i])

    def predict(self, X_test):
        output = np.zeros(shape=(X_test.shape[0], self.outdim))
        for i in range(self.outdim):
            output[:,i] = self.svrs[i].predict(X_test)
        return output

def svr_model_selection(X_train, y_train):
    print("--------- SVR MODEL SELECTION ---------")
    C0, gamma0, eps0, C1, gamma1, eps1 = svr_gridsearch(X_train, y_train)
    finer_gs = svr_gridsearch(X_train, y_train, is_fine_search=True, coarse_gamma=[gamma0, gamma1], coarse_C=[C0, C1], coarse_eps=[eps0,eps1])
    # plot_search_heatmap(finer_gs, "SVR Finer Gridsearch")
    # plot_search_results(finer_gs, "SVR Finer parameters")
    return finer_gs


def svr_gridsearch(X_train, y_train, is_fine_search=False, coarse_gamma=None, coarse_C=None, coarse_eps=.1):
    if DEBUG:
        if is_fine_search:
            gamma_range0,  gamma_range1 = np.linspace(coarse_gamma[0] / 10., coarse_gamma[0] * 10, num=3), np.linspace(coarse_gamma[1] / 10., coarse_gamma[1] * 10, num=3)
            C_range0, C_range1 = np.linspace(coarse_C[0] / 10., coarse_C[0] * 10., num=3), np.linspace(coarse_C[1] / 10., coarse_C[1] * 10., num=3)
        else:
            gamma_range = 10 ** np.arange(-4, 1, step=1, dtype=float)
            C_range = 10 ** np.arange(-1, 1, step=1, dtype=float)
            gamma_range0 = gamma_range1 = gamma_range
            C_range0 = C_range1 = C_range
            # epsilon_range = [0, 0.01, 0.1, 1, 10]

    else:
        if is_fine_search:
            gamma_range0, gamma_range1 = np.linspace(coarse_gamma[0] / 5., coarse_gamma[0] * 5, num=25), np.linspace(
                coarse_gamma[1] / 5., coarse_gamma[1] * 5, num=25)
            C_range0, C_range1 = np.linspace(coarse_C[0] / 5., coarse_C[0] * 5., num=25), np.linspace(
                coarse_C[1] / 5., coarse_C[1] * 5., num=25)
        # epsilon_range = [coarse_eps]
        else:
            gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
            C_range = 10 ** np.arange(-3, 4, step=1, dtype=float)
            gamma_range0 = gamma_range1 = gamma_range
            C_range0 = C_range1 = C_range
            # epsilon_range = [0, 0.01, 0.1, 1, 10]

    # TRAINING FIRST SVR OVER Y0
    pipe_svr = Pipeline([
                         ('reg', SVR(kernel="rbf"))])
    tuned_parameters = {'reg__gamma': gamma_range0,
                        'reg__C': C_range0,
                        }
    scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
    print("MODEL SELECTIOJN SVR0")

    gs0 = GridSearchCV(estimator=pipe_svr,
                      cv=5,
                      param_grid=tuned_parameters,
                      scoring=scorer,
                      # verbose=10,
                      n_jobs=4,
                      return_train_score=True)
    gs0.fit(X_train, y_train[:,0])
    for param in gs0.best_params_:
        print(param + ": " + str(gs0.best_params_[param]))
    # store results

    save_gridsearch_results(gs0, "results/svr/svr_gs_resuls_0.csv")
    best_mean_test = gs0.cv_results_["mean_test_score"][gs0.best_index_]
    best_std_test = gs0.cv_results_["std_test_score"][gs0.best_index_]
    best_mean_train = gs0.cv_results_["mean_train_score"][gs0.best_index_]
    best_std_train = gs0.cv_results_["std_train_score"][gs0.best_index_]
    print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
        abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
    if not is_fine_search:
        plot_search_heatmap(gs0, "SVR Coarse Gridsearch Y0")
        plot_search_results(gs0, "SVR Coarse parameters Y0")

    print("---------------------------------------")

    # TRAINING FIRST SVR OVER Y0
    pipe_svr1 = Pipeline([
                         ('reg', SVR(kernel="rbf"))])
    tuned_parameters = {'reg__gamma': gamma_range0,
                        'reg__C': C_range0,
                        }

    print("MODEL SELECTIOJN SVR1")
    gs1 = GridSearchCV(estimator=pipe_svr1,
                       cv=5,
                       param_grid=tuned_parameters,
                       scoring=scorer,
                       # verbose=10,
                       n_jobs=4,
                       return_train_score=True)
    gs1.fit(X_train, y_train[:, 1])

    for param in gs1.best_params_:
        print(param + ": " + str(gs1.best_params_[param]))
    best_mean_test = gs1.cv_results_["mean_test_score"][gs1.best_index_]
    best_std_test = gs1.cv_results_["std_test_score"][gs1.best_index_]
    best_mean_train = gs1.cv_results_["mean_train_score"][gs1.best_index_]
    best_std_train = gs1.cv_results_["std_train_score"][gs1.best_index_]
    print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
        abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
    print("---------------------------------------")
    save_gridsearch_results(gs0, "results/svr/svr_gs_resuls_1.csv")
    if not is_fine_search:
        plot_search_heatmap(gs1, "SVR Coarse Gridsearch Y1")
        plot_search_results(gs1, "SVR Coarse parameters Y1")
    if is_fine_search:
        # wrap the SVRs in the MiltioutputSVR class
        return MultioutputSVR(gs0.best_estimator_, gs1.best_estimator_)
    else:
        return gs0.best_params_['reg__C'], gs0.best_params_['reg__gamma'], .1, gs1.best_params_['reg__C'], gs1.best_params_['reg__gamma'], .1

def svr_poly_gridsearch(X_train, y_train):
    gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
    C_range = 10 ** np.arange(-3, 6, step=1, dtype=float)
    gamma_range = [1]
    C_range = [.1]
    degree_range = [3,4,5]

    pipe_svr = Pipeline([
                         ('reg', RegressorChain(SVR(kernel="poly", cache_size=6000)))])
    tuned_parameters = {'reg__base_estimator__gamma': gamma_range,
                        'reg__base_estimator__C': C_range,
                        'reg__base_estimator__degree': degree_range}
    # tuned_parameters = {'reg__estimator__gamma': [0.08],
    #                     'reg__estimator__C': [14]}
    # Coarse Gridsearch
    scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
    gs = GridSearchCV(estimator=pipe_svr,
                      cv=[(slice(None), slice(None))],
                      param_grid=tuned_parameters,
                      scoring=scorer,
                      # verbose=10,
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
    print("---------------------------------------")
    return gs



def svr_poly_time_analysis(X_train, y_train):
    # def mean_euclidian_error_loss(y_true, pred_y):
    #     l2_norms = np.linalg.norm(y_true - pred_y)
    #     return np.mean(l2_norms)

    X_train, X_inner_test, y_train, y_inner_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_set_threshold = np.arange(50, X_train.shape[0], 50)
    pipe_svr = Pipeline([
                         ('reg', RegressorChain(SVR(kernel="poly",
                                                    C=.1,
                                                    gamma=1,
                                                    degree=2,
                                                    cache_size=6000)))])
    times = []
    print(" \tN_SAMPLE, \tTIME[s], \tMEE")
    for n_samples in train_set_threshold:
        tic = time.perf_counter()
        x_t = X_train[:n_samples, :]
        y_t = y_train[:n_samples,:]
        pipe_svr.fit(x_t,y_t)
        elapsed = time.perf_counter() - tic
        times.append(elapsed)
        y_pred = pipe_svr.predict(X_inner_test)
        sup_vec = pipe_svr["reg"].estimators_[1].n_support_[0]
        sup_vec_ration = sup_vec / n_samples
        mee = mean_euclidian_error_loss(y_pred, y_inner_test)
        print(" \t{}, \t{:.2f}, \t{:.2f}, \t{:.2f}".format(n_samples, elapsed, mee,sup_vec_ration ))
        # print("[Model Assessment] MEE: %0.2f" % abs(mee))
        # print("------------------------------------")
        # print()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(train_set_threshold, times, "o-", color="b", label="Train MEE")
    plt.xlabel("Training set size")
    plt.ylabel("Time [s]")
    plt.show()

    return gs