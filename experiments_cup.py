from sklearn.model_selection._split import train_test_split
import os

from src.ensamble_svr import ensamble_srv_model_selection
from src.linear_models import linear_model_selection, model_assessment, linear_lbe_regularized_model_selection, \
    LASSO_model_selection, RIDGE_model_selection, linear_lbe_reg_model_selection
from src.random_forest import random_forest_model_selection
from src.svm import svr_model_selection
from src.utils.io import save_gridsearch_results, load_gridsearch_results
from src.utils.plots import plot_search_results, plot_search_df_results
from src.utils.preprocessing import cup_create_df, cup_split_data_target

train_path = "data/ml-cup21/ML-CUP21-TR.csv"
test_path = "data/ml-cup21/ML-CUP21-TS.csv"

train_path = os.path.join(os.path.dirname(__file__), train_path)
test_path = os.path.join(os.path.dirname(__file__), test_path)

train_df = cup_create_df(train_path, True)
test_df = cup_create_df(test_path, False)
# Drop first column - remove index columns
new_df = train_df.drop(columns="id", axis=1, inplace=False)
data, target = cup_split_data_target(new_df)
# transform into numpy arrays
X_train = data.to_numpy()
y_train = target.to_numpy()
# shuffling data
X_train, X_inner_test, y_train, y_inner_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# MODELS:

"""# LINEAR MODEL
linear_gs = linear_model_selection(X_train, y_train)
linear_mee = model_assessment(linear_gs, X_train, y_train, X_inner_test, y_inner_test)
###### save_gridsearch_results(linear_gs, "results/linear/linear_gs_results.csv")



# LINEAR MODEL WITH LBE"""
linear_lbe_gs = linear_lbe_regularized_model_selection(X_train, y_train)
linear_lbe_mee = model_assessment(linear_lbe_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(linear_lbe_gs, "results/linear/linear_lbe_regularized_gs_results.csv")
"""
# LINEAR MODEL WITH L1 REGULARIZATION
lasso_gs = LASSO_model_selection(X_train, y_train)
plot_search_results(lasso_gs, "LASSO parameters")
lasso_gs_mee = model_assessment(lasso_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lasso_gs, "../results/linear/lasso_gs_results.csv")

# LINEAR MODEL WITH L2 REGULARIZATION
ridge_gs = RIDGE_model_selection(X_train, y_train)
plot_search_results(ridge_gs, "RIDGE parameters")
ridge_gs_mee = model_assessment(ridge_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(ridge_gs, "../results/linear/ridge_gs_results.csv")

# LINEAR MODEL WITH LBE AND REGULARIZATION
lbe_reg_gs = linear_lbe_reg_model_selection(X_train, y_train)
plot_search_results(ridge_gs, "LBE WITH REGULARIZATION parameters")
lbe_reg_mee = model_assessment(lbe_reg_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(lbe_reg_gs, "../results/linear/lbe_reg_results.csv")

# SVR
svr_gs = srv_model_selection(X_train, y_train)
plot_search_results(svr_gs, "SVR parameters")
lbe_reg_mee = model_assessment(svr_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(svr_gs, "../results/svr/svr_results.csv")

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