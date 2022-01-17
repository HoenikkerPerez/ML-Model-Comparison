import time
from multiprocessing import Pool

import numpy as np
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._validation import cross_validate
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm._classes import SVR

from src.utils.preprocessing import mean_euclidian_error_loss



def my_kernel(g_p, g_r, d, rho):
    """Clousure for respecting sklearn kernel callback signature"""

    def kernel(x1, x2):
        m_kernel = (rho * polynomial_kernel(x1, x2, degree=d, gamma=g_p, coef0=0) + (
            1 - rho) * rbf_kernel(x1, x2, gamma=g_r))
        # + (1-rho) *sigmoid_kernel(x1, x2))
        return m_kernel

    return kernel


def fit_svr(X_train, y_train, g_p, g_r, d, rho, eps=.1, C=14):
    tic = time.perf_counter()
    svr = Pipeline([('scl', StandardScaler()),
                    ('reg', RegressorChain(SVR(kernel=my_kernel(g_p, g_r, d, rho), epsilon=eps, C=C, cache_size=10000)))])
    svr.fit(X_train, y_train)
    scores = cross_validate(svr,
                            X_train,
                            y_train,
                            cv=5,
                            scoring=make_scorer(mean_euclidian_error_loss, greater_is_better=False),
                            return_train_score=True)
    best_mean_test = scores['test_score'].mean()
    best_std_test = scores['test_score'].std()
    best_mean_train = scores['train_score'].mean()
    best_std_train = scores['train_score'].std()
    print(" %.3f,%.3f,%d,%.1f,%.2f,%0.3f,%0.3f,%0.3f,%0.3f,%.2f,%.1f,%.2f" % (

        g_p, g_r, d, rho,C,
        abs(best_mean_train),
        best_std_train,
        abs(best_mean_test),
        best_std_test,
        time.perf_counter() - tic,
        eps,
        svr["reg"].estimators_[1].n_support_[0]/X_train.shape[0]))
    # print("(%.1f,%.1f,%d,%.1f) %0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)      %.2f" % (
    #     g_p, g_r, d, rho, abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test, time.perf_counter()-tic))


def mixed_kernel_srv_model_selection(X_train, y_train):
    """
    References: Smits, G.F.; Jordaan, E.M. (2002). [IEEE 2002 International Joint Conference on Neural Networks (
    IJCNN) - Honolulu, HI, USA (12-17 May 2002)] Proceedings of the 2002 International Joint Conference on Neural
    Networks. IJCNN'02 (Cat. No.02CH37290) - Improved SVM regression using mixtures of kernels. , (), 2785–2790.
    doi:10.1109/IJCNN.2002.1007589
    """

    # p = Pool(4)
    print("g_p,g_r,d,rho,mean_train,std_train,mean_test,std_test,time,eps,sv_ratio")
    # for eps in [0, .1, 1]:
    for C in [.01, .1, 1, 10]:
        for g_p in [.1]:# [.001, .01, .1]:
            for g_r in [.1, .5]: #[.001, .01, .1]:
                for d in [1, 2, 3, 4]:
                    for rho in [0, .2, .4, .6, .8, 1]:
                        # p.map(fit_svr, (X_train, y_train, g_p, g_r, d, rho))
                        fit_svr(X_train, y_train, g_p, g_r, d, rho, eps=.1, C=C)
    exit()


# def mixed_kernel_srv_model_selection(X_train, y_train):
#     """
#     References: Smits, G.F.; Jordaan, E.M. (2002). [IEEE 2002 International Joint Conference on Neural Networks (
#     IJCNN) - Honolulu, HI, USA (12-17 May 2002)] Proceedings of the 2002 International Joint Conference on Neural
#     Networks. IJCNN'02 (Cat. No.02CH37290) - Improved SVM regression using mixtures of kernels. , (), 2785–2790.
#     doi:10.1109/IJCNN.2002.1007589
#     """
#
#     def mix_kernel(x1, x2):
#         global g, d, rho
#         m_kernel = (rho * polynomial_kernel(x1, x2, degree=d, gamma=g_p) + (
#             1 - rho) * rbf_kernel(x1, x2, gamma=g_r))
#         # m_kernel = (rho * rbf_kernel(x1, x2, gamma=g_p) + (
#         #              1 - rho) * rbf_kernel(x1, x2, gamma=g_r))
#         return m_kernel
#
#     svr = RegressorChain(SVR(kernel=MixedKernel, C=14, shrinking=False))
#     ens_svr = Pipeline([('scl', StandardScaler()), ('reg', svr)])
#     C_range = [14]
#     print(svr.get_params().keys())
#     tuned_parameters = {'reg__base_estimator__C': C_range}
#     scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
#     for r in np.arange(0, 1, .1):
#         global rho
#         rho = r
#         print("rho: ", rho)
#         gs = GridSearchCV(estimator=ens_svr,
#                           cv=5,
#                           param_grid=tuned_parameters,
#                           scoring=scorer,
#                           # verbose=1,
#                           n_jobs=-1,
#                           return_train_score=True)
#         gs.fit(X_train, y_train)
#         for param in gs.best_params_:
#             print(param + ": " + str(gs.best_params_[param]))
#         best_mean_test = gs.cv_results_["mean_test_score"][gs.best_index_]
#         best_std_test = gs.cv_results_["std_test_score"][gs.best_index_]
#         best_mean_train = gs.cv_results_["mean_train_score"][gs.best_index_]
#         best_std_train = gs.cv_results_["std_train_score"][gs.best_index_]
#         print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
#             abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test))
#         print("---------------------------------------")
#     return gs


# # Wrapper class for the custom kernel
# class MixedKernel(BaseEstimator, TransformerMixin):
#     def __init__(self, g_p, g_r, d, rho):
#         super(MixedKernel, self).__init__()
#         self.g_p = g_p
#         self.g_r = g_r
#         self.d = d
#         self.rho = rho
#
#     def transform(self, X):
#         return mixed_kernel(X, self.X_train_, g_p=self.g_p, g_r=self.g_r, d=self.d, rho=self.rho)
#
#     def fit(self, X, y=None, **fit_params):
#         self.X_train_ = X
#         return self

def ensamble_srv_model_selection(X_train, y_train):
    bagging = BaggingRegressor(RegressorChain(SVR(kernel="rbf")), max_samples=.4)
    ens_svr = Pipeline([('scl', StandardScaler()), ('reg', bagging)])

    gamma_range = 10 ** np.arange(-7, 3, step=1, dtype=float)
    C_range = 10 ** np.arange(-3, 6, step=1, dtype=float)
    tuned_parameters = {'reg__base_estimator__base_estimator__gamma': gamma_range,
                        'reg__base_estimator__base_estimator__C': C_range}
    print(ens_svr.get_params().keys())
    scorer = make_scorer(mean_euclidian_error_loss, greater_is_better=False)
    gs = GridSearchCV(estimator=ens_svr,
                      cv=5,
                      param_grid=tuned_parameters,
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
    print("---------------------------------------")
    return gs
