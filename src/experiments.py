from sklearn.model_selection._split import train_test_split

from src.linear_models import linear_model_selection, model_assessment, linear_lbe_regularized_model_selection, \
    LASSO_model_selection, RIDGE_model_selection, linear_lbe_reg_model_selection
from src.svr import srv_model_selection
from src.utils.io import save_gridsearch_results
from src.utils.plots import plot_search_results
from src.utils.preprocessing import create_df, split_data_target

train_path = "../data/ml-cup21/ML-CUP21-TR.csv"
test_path = "../data/ml-cup21/ML-CUP21-TS.csv"

train_df = create_df(train_path, True)
test_df = create_df(test_path, False)
# Drop first column - remove index columns
new_df = train_df.drop(columns="id", axis=1, inplace=False)
data, target = split_data_target(new_df)
# transform into numpy arrays
X_train = data.to_numpy()
y_train = target.to_numpy()
# shuffling data
X_train, X_inner_test, y_train, y_inner_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# LINEAR MODEL
# linear_gs = linear_model_selection(X_train, y_train)
# linear_mee = model_assessment(linear_gs, X_train, y_train, X_inner_test, y_inner_test)
####### save_gridsearch_results(linear_gs, "../results/linear/linear_gs_results.csv")

# MODELS:

# # LINEAR MODEL WITH LBE
# linear_lbe_gs = linear_lbe_regularized_model_selection(X_train, y_train)
# linear_lbe_mee = model_assessment(linear_lbe_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(linear_lbe_gs, "../results/linear/linear_lbe_regularized_gs_results.csv")
#
# # LINEAR MODEL WITH L1 REGULARIZATION
# lasso_gs = LASSO_model_selection(X_train, y_train)
# plot_search_results(lasso_gs, "LASSO parameters")
# lasso_gs_mee = model_assessment(lasso_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(lasso_gs, "../results/linear/lasso_gs_results.csv")
#
# # LINEAR MODEL WITH L2 REGULARIZATION
# ridge_gs = RIDGE_model_selection(X_train, y_train)
# plot_search_results(ridge_gs, "RIDGE parameters")
# ridge_gs_mee = model_assessment(ridge_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(ridge_gs, "../results/linear/ridge_gs_results.csv")
#
# # LINEAR MODEL WITH LBE AND REGULARIZATION
# lbe_reg_gs = linear_lbe_reg_model_selection(X_train, y_train)
# plot_search_results(ridge_gs, "LBE WITH REGULARIZATION parameters")
# lbe_reg_mee = model_assessment(lbe_reg_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
# save_gridsearch_results(lbe_reg_gs, "../results/linear/lbe_reg_results.csv")

# SVR
svr_gs = srv_model_selection(X_train, y_train)
plot_search_results(svr_gs, "SVR parameters")
lbe_reg_mee = model_assessment(svr_gs.best_estimator_, X_train, y_train, X_inner_test, y_inner_test)
save_gridsearch_results(svr_gs, "../results/svr/svr_results.csv")