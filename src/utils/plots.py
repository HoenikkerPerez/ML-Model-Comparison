import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_search_df_results(results, title):
    results['mean_train_score'] = results['mean_train_score'].abs()
    results['mean_test_score'] = results['mean_test_score'].abs()

    # Results from grid search
    means_test = results['mean_test_score'].to_numpy()
    stds_test = results['std_test_score'].to_numpy()
    means_train = results['mean_train_score'].to_numpy()
    stds_train = results['std_train_score'].to_numpy()

    results = results[results != "--"]
    results = results.dropna(axis=1)

    # Getting indexes of values per hyper-parameter
    masks_names = list(results.columns.values[6:])
    best_row = results[results.mean_test_score == results.mean_test_score.min()]
    best_row = best_row.iloc[:, 6:]
    # if only 2 parameters, draw complete heatmap
    if len(masks_names) > 1:
        # Ploting results
        fig, ax = plt.subplots(1, len(best_row.columns), sharex='none', sharey='all')
        fig.suptitle(title)

        for idx, p in enumerate(masks_names):

            # fix all but one param
            plotted_var_param = masks_names[idx]
            # best_plotted_param = best_row[p].values[0]
            # best idxs are the one with fixed best parameters
            fixed_params = masks_names[:idx] + masks_names[idx + 1:]
            single_results = results.copy()
            for i, fp in enumerate(fixed_params):
                single_results = single_results[single_results[fp] == best_row[fp].values[0]]
            best_idxs = list(single_results.index)
            mean_test = means_test[best_idxs]
            std_test = stds_test[best_idxs]
            mean_train = means_train[best_idxs]
            std_train = stds_train[best_idxs]
            param = single_results[plotted_var_param].to_numpy()

            ax[idx].plot(param, mean_train, "o-", color="b", label="Train MEE")
            ax[idx].fill_between(param, mean_train + std_train, mean_train - std_train, color="b", alpha=0.2)
            ax[idx].plot(param, mean_test, "o-", color="g", label="Test MEE")
            ax[idx].fill_between(param, mean_test + std_test, mean_test - std_test, color="g", alpha=0.2)

            ax[idx].set_ylim([0, 3])
            ax[idx].set_xlabel(p.upper())
    plt.legend()
    plt.show()


def plot_search_results(grid, title):
    # Results from grid search
    results = grid.cv_results_
    means_test = np.abs(results['mean_test_score'])
    stds_test = results['std_test_score']
    means_train = np.abs(results['mean_train_score'])
    stds_train = results['std_train_score']

    # Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    if len(masks_names) > 1:
        for p_k, p_v in grid.best_params_.items():
            masks.append(list(results['param_' + p_k].data == p_v))

        params = grid.param_grid

        # Ploting results
        fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all')
        fig.suptitle(title)
        # fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i + 1:])
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            param = np.array(params[p])
            mean_test = np.abs(np.array(means_test[best_index]))
            std_test = np.array(stds_test[best_index])
            mean_train = np.abs(np.array(means_train[best_index]))
            std_train = np.array(stds_train[best_index])

            ax[i].plot(param, mean_train, "o-", color="b", label="Train MEE")
            ax[i].fill_between(param, mean_train + std_train, mean_train - std_train, color="b", alpha=0.2)
            ax[i].plot(param, mean_test, "o-", color="g", label="Test MEE")
            ax[i].fill_between(param, mean_test + std_test, mean_test - std_test, color="g", alpha=0.2)
            # ax[i].errorbar(x, np.abs(y_1), e_1, linestyle='--', marker='o', label='test')
            # ax[i].errorbar(x, np.abs(y_2), e_2, linestyle='-', marker='^', label='train')
            # ax[i].set_xscale("log")
            ax[i].set_ylim([0, 3])
            ax[i].set_xlabel(p.upper())
    else:
        p = masks_names[0]
        param = grid.param_grid[p]
        fig, ax = plt.subplots(1)
        fig.suptitle(title)
        ax.set_xscale('log')
        ax.plot(param, means_train, "o-", color="b", label="Train MEE")
        ax.fill_between(param, means_train + stds_train, means_train - stds_train, color="b", alpha=0.2)
        ax.plot(param, means_test, "o-", color="g", label="Test MEE")
        ax.fill_between(param, means_test + stds_test, means_test - stds_test, color="g", alpha=0.2)
        ax.set_xlabel(p.upper())
    plt.legend()
    plt.show()
