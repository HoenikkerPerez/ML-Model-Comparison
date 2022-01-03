import numpy as np
from sklearn.model_selection._split import train_test_split
import os

from sklearn.preprocessing._encoders import OneHotEncoder

from src.ensamble_svr import ensamble_srv_model_selection
from src.linear_models import linear_model_selection, model_assessment, linear_lbe_regularized_model_selection, \
    LASSO_model_selection, RIDGE_model_selection, linear_lbe_reg_model_selection
from src.random_forest import random_forest_model_selection
from src.svm import svr_model_selection, svc_model_selection
from src.utils.io import save_gridsearch_results, load_gridsearch_results
from src.utils.plots import plot_search_results, plot_search_df_results
from src.utils.preprocessing import cup_create_df, cup_split_data_target, monk_create_df, monk_split_data_target

def model_assessment(model, X_train, y_train, X_test, y_test, mode="regression"):
    if mode == "regression":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).sum() / y_pred.shape[0]
        print("[MODEL ASSESSMENT]")
        print("ACCURACY: %0.2f" % accuracy)
        print("------------------------------------")
        print()

        return accuracy
    elif mode == "classifier":
        y_train = np.where(y_train == 0, -1, y_train)
        y_test = np.where(y_test == 0, -1, y_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred > 0, 1, 0)

        accuracy = (y_pred == y_test).sum() / y_pred.shape[0]
        print("[MODEL ASSESSMENT]")
        print("ACCURACY: %0.2f" % accuracy)
        print("------------------------------------")
        print()

        return accuracy

for monk_idx in [1,2,3]:
    print("\t\tMONK{} EXPERIMENTS:".format(monk_idx))
    train_path = "data/monk/monks-{}.train".format(monk_idx)
    test_path = "data/monk/monks-{}.test".format(monk_idx)

    train_path = os.path.join(os.path.dirname(__file__), train_path)
    test_path = os.path.join(os.path.dirname(__file__), test_path)

    train_df = monk_create_df(train_path)
    test_df = monk_create_df(test_path)
    # Drop first column - remove index columns
    X_train, y_train = monk_split_data_target(train_df)
    X_test, y_test = monk_split_data_target(test_df)

    # one hot encoding
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(X_train)
    X_train = one_hot_encoder.transform(X_train)
    X_test = one_hot_encoder.transform(X_test)

    # LINEAR MODEL WITH L1 REGULARIZATION
    lasso_gs = LASSO_model_selection(X_train, y_train, mode="classifier")
    plot_search_results(lasso_gs, "LASSO parameters", vmin=0.5, vmax=1.1)
    lasso_acc = model_assessment(lasso_gs.best_estimator_, X_train, y_train, X_test, y_test)
    # save_gridsearch_results(lasso_gs, "../results/linear/lasso_gs_results.csv")

    # LINEAR MODEL WITH L2 REGULARIZATION
    ridge_gs = RIDGE_model_selection(X_train, y_train, mode="classifier")
    plot_search_results(ridge_gs, "RIDGE parameters", vmin=0.5, vmax=1.1)
    ridge_acc = model_assessment(ridge_gs.best_estimator_, X_train, y_train, X_test, y_test)

    # LINEAR MODEL WITH LBE AND REGULARIZATION
    lbe_reg_gs = linear_lbe_reg_model_selection(X_train, y_train, mode="classifier")
    plot_search_results(ridge_gs, "LBE WITH REGULARIZATION parameters", vmin=0.5, vmax=1.1)
    lbe_reg_mee = model_assessment(lbe_reg_gs.best_estimator_, X_train, y_train, X_test, y_test)
    # save_gridsearch_results(lbe_reg_gs, "../results/linear/lbe_reg_results.csv")

    # SUPPORT VECTOR CLASSIFIER
    svc_gs = svc_model_selection(X_train, y_train)
    plot_search_results(svc_gs, "SVR parameters", vmin=0.5, vmax=1.1)
    svc_acc = model_assessment(svc_gs.best_estimator_, X_train, y_train, X_test, y_test)
    # save_gridsearch_results(svc_gs, "results/svc/svc_results.csv")

    # RANDOM FOREST
    random_forest_gs = random_forest_model_selection(X_train, y_train, mode="classifier")
    plot_search_results(random_forest_gs, "RANDOM FOREST CLASS parameters", vmin=0.5, vmax=1.1)
    lbe_reg_mee = model_assessment(random_forest_gs.best_estimator_, X_train, y_train, X_test, y_test)
    # save_gridsearch_results(lbe_reg_mee, "results/ranndom_forest/random_forest_results.csv")

# PLOTS
# all


"""# LINEAR MODEL
linear_gs = linear_model_selection(X_train, y_train)
linear_mee = model_assessment(linear_gs, X_train, y_train, X_inner_test, y_inner_test)
###### save_gridsearch_results(linear_gs, "results/linear/linear_gs_results.csv")



# LINEAR MODEL WITH LBE
linear_lbe_gs = linear_lbe_regularized_model_selection(X_train, y_train)
linear_lbe_mee = model_assessment(linear_lbe_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(linear_lbe_gs, "results/linear/linear_lbe_regularized_gs_results.csv")





# LINEAR MODEL WITH LBE AND REGULARIZATION
lbe_reg_gs = linear_lbe_reg_model_selection(X_train, y_train)
plot_search_results(ridge_gs, "LBE WITH REGULARIZATION parameters")
lbe_reg_mee = model_assessment(lbe_reg_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lbe_reg_gs, "../results/linear/lbe_reg_results.csv")
"""
# SVR
# svr_gs = svc_model_selection(X_train, y_train)
#plot_search_results(svr_gs, "SVR parameters")
#lbe_reg_mee = model_assessment(svr_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
#save_gridsearch_results(svr_gs, "../results/svr/svr_results.csv")
"""
# ENSAMBLE SVR
ens_svr_gs = ensamble_srv_model_selection(X_train, y_train)
# plot_search_results(ens_svr_gs, "ENSAMBLE SVR parameters")
lbe_reg_mee = model_assessment(ens_svr_gs, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(ens_svr_gs, "../results/svr/ens_svr_results.csv")

# RANDOM FOREST
results_df = load_gridsearch_results("results/linear/linear_lbe_regularized_gs_results.csv")
plot_search_df_results(results_df, "ENSAMBLE SVR parameters")
random_forest_gs = random_forest_model_selection(X_train, y_train)
plot_search_results(random_forest_gs, "ENSAMBLE SVR parameters")
lbe_reg_mee = model_assessment(random_forest_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lbe_reg_mee, "../results/ranndom_forest/random_forest_results.csv")
"""
# PLOTS
# all