from sklearn.model_selection._split import train_test_split
import os

#from src.ensamble_svr import ensamble_srv_model_selection, mixed_kernel_srv_model_selection
"""from src.linear_models import linear_model_selection, model_assessment, linear_lbe_regularized_model_selection, \
    LASSO_model_selection, RIDGE_model_selection, linear_lbe_reg_model_selection, LASSO_plot_coefficients, \
    RIDGE_plot_coefficients, linear_lbe_reg_plot_coefficients
from src.random_forest import random_forest_model_selection
from src.svm import svr_model_selection, svr_poly_gridsearch, svr_poly_time_analysis
from src.utils.io import save_gridsearch_results, load_gridsearch_results
from src.utils.plots import plot_search_results, plot_search_df_results, plot_search_heatmap, plot_mixed_kernel_results"""
from src.mlp import mlcup_model_selection, mlcup_model_assessment, mlcup_model_testing, mlcup_model_prediction
from src.utils.preprocessing import cup_create_df, cup_split_data_target

import pandas as pd

train_path = "data/ml-cup21/ML-CUP21-TR.csv"
test_path = "data/ml-cup21/ML-CUP21-TS.csv"

train_path = os.path.join(os.path.dirname(__file__), train_path)
test_path = os.path.join(os.path.dirname(__file__), test_path)

train_df = cup_create_df(train_path, True)
test_df = cup_create_df(test_path, False)
# Drop first column - remove index columns
new_df = train_df.drop(columns="id", axis=1, inplace=False)
test_df = test_df.drop(columns="id", axis=1, inplace=False)
X_test = test_df.to_numpy()
print(X_test.shape)

data, target = cup_split_data_target(new_df)
# transform into numpy arrays
X_train = data.to_numpy()
y_train = target.to_numpy()

X_train, X_inner_test, y_train, y_inner_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# MODELS:

# LINEAR MODEL
"""
linear_gs = linear_model_selection(X_train, y_train)
linear_mee = model_assessment(linear_gs, X_train, y_train, X_inner_test, y_inner_test)
###### save_gridsearch_results(linear_gs, "results/linear/linear_gs_results.csv")


# LINEAR MODEL WITH LBE
linear_lbe_gs = linear_lbe_regularized_model_selection(X_train, y_train)
linear_lbe_mee = model_assessment(linear_lbe_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(linear_lbe_gs, "results/linear/linear_lbe_regularized_gs_results.csv")

# LINEAR MODEL WITH L1 REGULARIZATION
lasso_gs = LASSO_model_selection(X_train, y_train)

LASSO_plot_coefficients(X_train, y_train)
plot_search_results(lasso_gs, "LASSO parameters")
lasso_gs_mee = model_assessment(lasso_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lasso_gs, "results/linear/lasso_gs_results.csv")

# LINEAR MODEL WITH L2 REGULARIZATION
ridge_gs = RIDGE_model_selection(X_train, y_train)
RIDGE_plot_coefficients(X_train, y_train)
plot_search_results(ridge_gs, "RIDGE parameters")
ridge_gs_mee = model_assessment(ridge_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(ridge_gs, "results/linear/ridge_gs_results.csv")

# LINEAR MODEL WITH LBE AND REGULARIZATION
linear_lbe_reg_plot_coefficients(X_train, y_train)
lbe_reg_gs = linear_lbe_reg_model_selection(X_train, y_train)
plot_search_results(lbe_reg_gs, "LBE + Linear + Regularization parameters")
plot_search_heatmap(lbe_reg_gs, "LBE + Linear + Regularization", svm=False)
lbe_reg_mee = model_assessment(lbe_reg_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lbe_reg_gs, "results/linear/lbe_reg_results.csv")


# SVR
print("*****************   POLYNOMIAL ***********************")
# svr_gs = svr_poly_gridsearch(X_train, y_train)
svr_poly_time_analysis(X_train, y_train)
# lbe_reg_mee = model_assessment(svr_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
print("*****************  END POLYNOMIAL ***********************")

svr_gs = svr_model_selection(X_train, y_train)
plot_search_results(svr_gs, "SVR parameters")
# plot_search_heatmap(svr_gs, "SVR Gridsearch")
lbe_reg_mee = model_assessment(svr_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(svr_gs, "results/svr/svr_results.csv")


# ENSAMBLE SVR
ens_svr_gs = ensamble_srv_model_selection(X_train, y_train)
# plot_search_results(ens_svr_gs, "ENSAMBLE SVR parameters")
ens_svr_mee = model_assessment(ens_svr_gs, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(ens_svr_gs, "../results/svr/ens_svr_results.csv")


# MIXED KERNELS
mk_svr_gs = mixed_kernel_srv_model_selection(X_train, y_train)
plot_mixed_kernel_results('results/svr/mixed_kernels.csv')
mk_svr_mee = model_assessment(mk_svr_gs, X_train, y_train, X_inner_test, y_inner_test)

# RANDOM FOREST
# results_df = load_gridsearch_results("results/linear/linear_lbe_regularized_gs_results.csv")
# plot_search_df_results(results_df, "ENSAMBLE SVR parameters")
random_forest_gs = random_forest_model_selection(X_train, y_train)
plot_search_results(random_forest_gs, "ENSAMBLE SVR parameters")
lbe_reg_mee = model_assessment(random_forest_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lbe_reg_mee, "results/ranndom_forest/random_forest_results.csv")"""

# PLOTS
# all


# # MLP
# # starts model selection and returns dataframe with optimal hyperparameters
# # optimal_df = mlcup_model_selection(X_train, y_train)
# # or read already saved csv file with results of the model selection
# kfold_cv_df = pd.read_csv("./results/mlp/cup_results_GS.csv")
# get optimal hyperparameter values according to the minimum validation loss
# optimal_df = kfold_cv_df[kfold_cv_df.mean_val_loss == kfold_cv_df.mean_val_loss.min()]
# #train a new MLP model and evaluate on internal test set
# mlcup_model_assessment(optimal_df, X_train, y_train, X_inner_test, y_inner_test)

#load model and test it
path="./results/mlp/models/cup_model_1"
mlcup_model_testing(path, X_inner_test, y_inner_test)

#prediction
#mlcup_model_prediction(path=path, X_test=X_test)

# predict_btest('data//ml-cup21//ML-CUP21-TS.csv', linear_gs)