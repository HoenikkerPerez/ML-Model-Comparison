import multiprocessing
import time
from multiprocessing import Pool
import threading
from queue import Queue

import numpy as np
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.metrics._scorer import make_scorer
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._validation import cross_validate
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing._data import MinMaxScaler, StandardScaler
from sklearn.svm._classes import SVR

from src.svm import MultioutputSVR
from src.utils.preprocessing import mean_euclidian_error_loss

threadLock = threading.Lock()

queue_svr = Queue()
results0 = []
results1 = []

def worker(idx):
    while True:
        item = queue_svr.get()
        fit_svr(item["X_train"], item["y_train"], item["g_p"], item["g_r"], item["d"], item["rho"], item["eps"], item["C"], item["svr_idx"])
        queue_svr.task_done()



def my_kernel(g_p, g_r, d, rho):
    """Clousure for respecting sklearn kernel callback signature"""

    def kernel(x1, x2):
        m_kernel = (rho * polynomial_kernel(x1, x2, degree=d, gamma=g_p, coef0=0) + (
            1 - rho) * rbf_kernel(x1, x2, gamma=g_r))
        # m_kernel = (rho * polynomial_kernel(x1, x2, degree=d, gamma=g_p, coef0=0) + (
        #     1 - rho) * rbf_kernel(x1, x2, gamma=g_r))
        return m_kernel

    return kernel

def s_print(*a, **b):
    """Thread safe print function"""
    with threadLock:
        print(*a, **b)

def fit_svr(X_train, y_train, g_p, g_r, d, rho, eps=.1, C=14, svr_idx=None):
    global results0, results1
    tic = time.perf_counter()
    svr = Pipeline([
                    ('reg',
                     SVR(kernel=my_kernel(g_p, g_r, d, rho), epsilon=eps, C=C, cache_size=10000))])

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
    s_print(" %.3f,%.3f,%d,%.1f,%.1f,"
          "%.2f,%0.3f,%0.3f,%0.3f,%0.3f,%.2f,%.2f" % (
        g_p, g_r, d, rho, eps, C,
        abs(best_mean_train),
        best_std_train,
        abs(best_mean_test),
        best_std_test,
        time.perf_counter() - tic,
        svr["reg"].n_support_[0] / X_train.shape[0]))
    
    result = {"g_p":g_p, "g_r":g_r, "d": d, "rho": rho, "eps":eps, "C": C,
        "best_mean_train": abs(best_mean_train),
        "best_std_train": best_std_train,
        "best_mean_test": abs(best_mean_test),
        "best_std_test": best_std_test,
        "time": time.perf_counter() - tic,
        "sv_ratio": svr["reg"].n_support_[0] / X_train.shape[0]}
    
    if svr_idx == 0:
        results0.append(result)
    elif svr_idx== 1:
        results1.append(result)
    # print("(%.1f,%.1f,%d,%.1f) %0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)      %.2f" % (
    #     g_p, g_r, d, rho, abs(best_mean_train), best_std_train, abs(best_mean_test), best_std_test, time.perf_counter()-tic))


def mixed_kernel_srv_model_selection(X_train, y_train):
    print("MODEL MIXED KERNEL FUNCTION")

    n_threads = multiprocessing.cpu_count()
    for idx in range(n_threads):
        threading.Thread(target=worker,args=(idx,), daemon=True).start()
    global queue_svr
    """
    References: Smits, G.F.; Jordaan, E.M. (2002). [IEEE 2002 International Joint Conference on Neural Networks (
    IJCNN) - Honolulu, HI, USA (12-17 May 2002)] Proceedings of the 2002 International Joint Conference on Neural
    Networks. IJCNN'02 (Cat. No.02CH37290) - Improved SVM regression using mixtures of kernels. , (), 2785–2790.
    doi:10.1109/IJCNN.2002.1007589
    """
    with open("results_mixed_kernel0def.csv", "w") as fd:
        fd.write("g_p,g_r,d,rho,eps,C,mean_train,std_train,mean_test,std_test,time,sv_ratio\n")
    with open("results_mixed_kernel1def.csv", "w") as fd:
        fd.write("g_p,g_r,d,rho,eps,C,mean_train,std_train,mean_test,std_test,time,sv_ratio\n")

    # p = Pool(4)
    print("g_p,g_r,d,rho,eps,C,mean_train,std_train,mean_test,std_test,time,sv_ratio")
    for svr_idx in [0,1]:
        # for eps in [0, .1, 1]:
        for C in [.1, 1, 10, 100]:
            for g_p in [.01, .1]:
                for g_r in [.01, .1]:
                    for d in [1,2,3,4,5]:
                        for rho in [.1,.2,.3,.4,.5,.6,.7,.8,.9]: #[0, .1, .2, .3, .4, .5, .6, .7, .8,.9, 1]:
                            eps = .1
                            task = {"X_train": X_train,
                                    "y_train": y_train[:,svr_idx],
                                    "g_p": g_p,
                                    "g_r": g_r,
                                    "d": d,
                                    "rho": rho,
                                    "eps": eps,
                                    "C": C,
                                    "svr_idx": svr_idx}
                            queue_svr.put(task)

    queue_svr.join()
    global results0, results1
    results0 = sorted(results0, key=lambda d: d['best_mean_test'])
    results1 = sorted(results1, key=lambda d: d['best_mean_test'])

    best_model_0 = results0[0]
    best_model_1 = results1[0]

    with open("results_mixed_kernel0def.csv","a") as fd:
        for res in results0:
            fd.write("%.3f,%.3f,%d,%.1f,%.1f,"
                    "%.2f,%0.3f,%0.3f,%0.3f,%0.3f,%.2f,%.2f\n" % (res["g_p"], res["g_r"], res["d"], res["rho"], res["eps"], res["C"],
                 res["best_mean_train"],
                 res["best_std_train"],
                 res["best_mean_test"],
                 res["best_std_test"],
                 res["time"],
                 res["sv_ratio"]))

    with open("results_mixed_kernel1def.csv","a") as fd:
        for res in results1:
            fd.write("%.3f,%.3f,%d,%.1f,%.1f,"
                    "%.2f,%0.3f,%0.3f,%0.3f,%0.3f,%.2f,%.2f\n" % (res["g_p"], res["g_r"], res["d"], res["rho"], res["eps"], res["C"],
                 res["best_mean_train"],
                 res["best_std_train"],
                 res["best_mean_test"],
                 res["best_std_test"],
                 res["time"],
                 res["sv_ratio"]))

    # best_model_0 = dict()
    # best_model_0["g_p"] = 1
    # best_model_0["g_r"] = 1
    # best_model_0["d"] = 4
    # best_model_0["rho"] = .6
    # best_model_0["eps"] = .1
    # best_model_0["C"] = 10
    #
    # best_model_1 = dict()
    # best_model_1["g_p"] = 1
    # best_model_1["g_r"] = 1
    # best_model_1["d"] = 4
    # best_model_1["rho"] = .4
    # best_model_1["eps"] = .1
    # best_model_1["C"] = 10

    svr0 = Pipeline([('scl', StandardScaler()),
                    ('reg', SVR(kernel=my_kernel(best_model_0["g_p"], best_model_0["g_r"], best_model_0["d"],
                                                 best_model_0["rho"]), epsilon=best_model_0["eps"], C= best_model_0["C"], cache_size=10000))])
    svr1 = Pipeline([('scl', StandardScaler()),
                     ('reg', SVR(kernel=my_kernel(best_model_1["g_p"], best_model_1["g_r"], best_model_1["d"],
                                                  best_model_1["rho"]), epsilon=best_model_1["eps"], C=best_model_1["C"], cache_size=10000))])


    multi_model = MultioutputSVR(svr0, svr1)

    score = cross_validate(multi_model,
                            X_train,
                            y_train,
                            cv=5,
                            scoring=make_scorer(mean_euclidian_error_loss, greater_is_better=False),
                            return_train_score=True,
                           n_jobs=-1)


    test_score = score["test_score"]
    mean_train = score["train_score"]
    print("Best params SVR0:")
    print("gamma_p: {}\ngamma_r: {}\nd:{}\nrho: {}\n".format(best_model_0["g_p"], best_model_0["g_r"], best_model_0["d"],
                                                 best_model_0["rho"]))
    print("Best params SVR1:")
    print("gamma_p: {}\ngamma_r: {}\nd:{}\nrho: {}\n".format(best_model_1["g_p"], best_model_1["g_r"], best_model_1["d"],
                                                 best_model_1["rho"]))
    print("%0.2f (\u03C3=%0.2f)\t\t\t%0.2f (\u03C3=%0.2f)" % (
        abs(mean_train.mean()), mean_train.std(), abs(test_score.mean()), test_score.std()))
    print("---------------------------------------")
    return multi_model


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
