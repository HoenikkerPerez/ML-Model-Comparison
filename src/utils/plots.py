import numbers

import matplotlib.pyplot as plt
import numpy as np
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
            ax[idx].set_xlabel(p)
    plt.legend()
    plt.show()


def plot_search_results(grid, title, vmin=0, vmax=3):
    # Results from grid search
    results = grid.cv_results_
    means_test = np.abs(results['mean_test_score'])
    stds_test = results['std_test_score']
    means_train = np.abs(results['mean_train_score'])
    stds_train = results['std_train_score']
    removables = ["reg__estimator__", "reg__base_estimator__", "lbe__", "reg__"]
    # Getting indexes of values per hyper-parameter
    masks = []
    masks_names = list(grid.best_params_.keys())
    if len(masks_names) > 1:
        for p_k, p_v in grid.best_params_.items():
            masks.append(list(results['param_' + p_k].data == p_v))

        params = grid.param_grid

        # Ploting results
        fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=(8,4))
        fig.suptitle(title)
        # fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
        for i, p in enumerate(masks_names):
            m = np.stack(masks[:i] + masks[i + 1:])
            best_parms_mask = m.all(axis=0)
            best_index = np.where(best_parms_mask)[0]
            param = np.array(params[p])
            if not isinstance(params[p][0], str) and not isinstance(params[p][0], numbers.Number):
                param = np.array([str(x) for x in params[p]])
            else:
                fmt = lambda x, pos: '{:.0f}e-3'.format((x - 1) * 1e3, pos)
                if np.max(param) / np.min(param) > 10:
                    ax[i].set_xscale('log')
                # ax[i].yaxis.set_major_formatter(FuncFormatter(fmt))
                # ax[i].ticklabel_format(axis='x', style='sci')
            mean_test = np.abs(np.array(means_test[best_index]))
            std_test = np.array(stds_test[best_index])
            mean_train = np.abs(np.array(means_train[best_index]))
            std_train = np.array(stds_train[best_index])

            ax[i].plot(param, mean_train, "--", color="b", label="Train MEE")
            ax[i].fill_between(param, mean_train + std_train, mean_train - std_train, color="b", alpha=0.2)
            ax[i].plot(param, mean_test, "-", color="g", label="Test MEE")
            ax[i].fill_between(param, mean_test + std_test, mean_test - std_test, color="g", alpha=0.2)
            # ax[i].errorbar(x, np.abs(y_1), e_1, linestyle='--', marker='o', label='test')
            # ax[i].errorbar(x, np.abs(y_2), e_2, linestyle='-', marker='^', label='train')
            # ax[i].set_xscale("log")
            ax[i].set_ylim([vmin, vmax])

            xlabel=p
            for removable in removables:
                if p.startswith(removable):
                    xlabel=p.replace(removable,"")
                    break
            ax[i].set_xlabel(xlabel)
    else:
        p = masks_names[0]
        param = grid.param_grid[p]
        fig, ax = plt.subplots(1)
        fig.suptitle(title)
        ax.set_xscale('log')
        ax.plot(param, means_train, "--", color="b", label="Train MEE")
        ax.fill_between(param, means_train + stds_train, means_train - stds_train, color="b", alpha=0.2)
        ax.plot(param, means_test, "-", color="g", label="Test MEE")
        ax.fill_between(param, means_test + stds_test, means_test - stds_test, color="g", alpha=0.2)
        xlabel = p
        for removable in removables:
            if p.startswith(removable):
                xlabel = p.replace(removable, "")
                break
        ax.set_xlabel(xlabel)
    plt.legend()
    plt.savefig('results/images/'+ title.replace(" ", "_").replace("+", "_"), bbox_inches='tight')
    plt.show()


def plot_search_heatmap(grid, title, vmin=None, vmax=None, svm=True):
    if svm:
        C_range = grid.param_grid['reg__C']
        gamma_range = grid.param_grid['reg__gamma']
        xlabel="gamma"
        ylabel = "C"
    else:
        C_range = grid.param_grid['reg__base_estimator__alpha']
        gamma_range = grid.param_grid['lbe__base_degree']
        xlabel = "degree"
        ylabel="alpha"

    scores = np.abs(grid.cv_results_["mean_test_score"].reshape(len(C_range), len(gamma_range)))
    if vmin is None: vmin = np.min(scores)
    if vmax is None: vmax = np.max(scores)

    plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.binary,
        vmin=vmin,
        vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(title)
    plt.savefig('results/images/' + title.replace(" ", "_").replace("+", "_"), bbox_inches='tight')
    plt.show()

    # TRAIN SCORE GRIDSEARCH
    scores = np.abs(grid.cv_results_["mean_train_score"].reshape(len(C_range), len(gamma_range)))
    vmin = np.min(scores)
    vmax = np.max(scores)

    plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.binary,
        vmin=vmin,
        vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(title+ "[TRAIN MEE]")
    plt.savefig('results/images/' + title.replace(" ", "_").replace("+", "_") + "_train", bbox_inches='tight')
    plt.show()

    # TRAIN TIME GRIDSEARCH
    scores = np.abs(grid.cv_results_['mean_fit_time'].reshape(len(C_range), len(gamma_range)))
    vmin = np.min(scores)
    vmax = np.max(scores)

    plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(
        scores,
        interpolation="nearest",
        cmap=plt.cm.binary,
        vmin=vmin,
        vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(title + "[TIME]")
    plt.savefig('results/images/' + title.replace(" ", "_").replace("+", "_") + "_time", bbox_inches='tight')
    plt.show()

def plot_learning_curves_mlp(history=None, path='', name='', loss='MSE'):
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss", linestyle='dashed')
    plt.title(f'Training and validation loss {name}')
    plt.ylabel(loss)
    plt.xlabel("EPOCHS")
    plt.legend()
    new_path=path+"{}_loss_reg.png".format(name)
    plt.savefig(new_path)

    if "accuracy" in history.history.keys():
        plt.figure()
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", linestyle='dashed')
        plt.title(f'Training and validation accuracy {name}')
        plt.ylabel("ACCURACY")
        plt.xlabel("EPOCHS")
        plt.legend()
        new_path = path + "{}_REG_accuracy.png".format(name)
        plt.savefig(new_path)

    plt.show()

def plot_mixed_kernel_results(filename0, filename1, title="Mixed Kernel dependence of \u03C1", vmin=0, vmax=1):
    df0 = pd.read_csv(filename0,sep=",")
    df0 = df0.sort_values(by=['mean_test'])
    df1 = pd.read_csv(filename1, sep=",")
    df1 = df0.sort_values(by=['mean_test'])

    fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # fig, ax = plt.subplots(1)
    fig.suptitle(title)

    df0 = df0.sort_values(by=['rho'])
    df0 = df0.loc[df0.groupby('rho')['mean_test'].idxmin()]
    # ax.set_xscale('log')
    ax[0].plot(df0["rho"], df0["mean_train"], "--bo", label="TR-MEE ")
    ax[0].fill_between(df0["rho"], df0["mean_train"] + df0["std_train"], df0["mean_train"] - df0["std_train"], color="b", alpha=0.2)
    ax[0].plot(df0["rho"], df0["mean_test"], "-go", label="VS-MEE")
    ax[0].fill_between(df0["rho"], df0["mean_test"] + df0["std_test"], df0["mean_test"] - df0["std_test"], color="g", alpha=0.2)
    ax[0].set_xlabel("\u03C1")
    # ax[1].set_ylim([vmin,vmax])
    ax[0].set_title("Y0")
    ax[0].set_xlabel("\u03C1")
    ax[0].set_ylabel("MEE")
    ax[0].legend()

    df1 = df1.sort_values(by=['rho'])
    df1 = df1.loc[df1.groupby('rho')['mean_test'].idxmin()]
    # ax.set_xscale('log')
    ax[1].plot(df1["rho"], df1["mean_train"], "--bo", label="TR-MEE ")
    ax[1].fill_between(df1["rho"], df1["mean_train"] + df1["std_train"], df1["mean_train"] - df1["std_train"], color="b", alpha=0.2)
    ax[1].plot(df1["rho"], df1["mean_test"], "-go", label="VS-MEE")
    ax[1].fill_between(df1["rho"], df1["mean_test"] + df1["std_test"], df1["mean_test"] - df1["std_test"], color="g", alpha=0.2)
    ax[1].set_xlabel("\u03C1")
    # ax[1].set_ylim([vmin,vmax])
    ax[1].set_title("Y1")
    ax[1].set_xlabel("\u03C1")
    ax[1].legend()

    ax[1].legend()

    plt.savefig('results/images/Mixed_Kernel_dependence_of_rho', bbox_inches='tight')
    plt.show()
    # exit()


def plot_mixed_kernel_results_multidegree(filename, title="Mixed Kernel dependence of \u03C1", vmin=0, vmax=1):
    df = pd.read_csv(filename,sep=",")
    df = df.sort_values(by=['mean_test'])
    print(df.head())

    fig, ax = plt.subplots(1,2, sharex='none',sharey=True ,figsize=(8, 4))
    # fig, ax = plt.subplots(1)
    fig.suptitle(title)
    ms = ["s","^","P","*","+", "8"]
    cs = ["b","g","r","y", "m", "c"]
    for i in range(5):
        # degree=3
        df3 = df[df["d"]==i+1]
        # df3 = df3[df3["g_p"]==.1]
        # df3 = df3[df3["g_r"]==.1]
        df3 = df3.sort_values(by=['rho'])
        df3 = df3.loc[df3.groupby('rho')['mean_test'].idxmin()]
        # ax.set_xscale('log')
        ax[0].plot(df3["rho"], df3["mean_train"], "--"+cs[i]+ms[i], label="TR-MEE (d={})".format(i+1))
        ax[0].fill_between(df3["rho"], df3["mean_train"] + df3["std_train"], df3["mean_train"] - df3["std_train"], color=cs[i], alpha=0.2)
        ax[1].plot(df3["rho"], df3["mean_test"], "-"+cs[i]+ms[i], label="VS-MEE (d={})".format(i+1))
        ax[1].fill_between(df3["rho"], df3["mean_test"] + df3["std_test"], df3["mean_test"] - df3["std_test"], color=cs[i], alpha=0.2)
    ax[0].set_xlabel("\u03C1")
    # ax[1].set_ylim([vmin,vmax])
    ax[0].set_title("Training Set")
    ax[1].set_xlabel("\u03C1")
    ax[1].set_title("Validation Set")
    ax[0].set_ylabel("MEE")
        # ax.set_title("Poly degree = 3")

    # # degree=4
    # df4 = df[df["d"]==4]
    # df4 = df4[df4["g_p"]==.1]
    # df4 = df4[df4["g_r"]==.1]
    # ax.plot(df4["rho"], df4["mean_train"], "--^", label="TR-MEE (d=4)")
    # ax.fill_between(df4["rho"], df4["mean_train"] + df4["std_train"], df4["mean_train"] - df4["std_train"],
    #                    color="b", alpha=0.2)
    # ax.plot(df4["rho"], df4["mean_test"], "-^", label="VS-MEE (d=4)")
    # ax.fill_between(df4["rho"], df4["mean_test"] + df4["std_test"], df4["mean_test"] - df4["std_test"], color="g", alpha=0.2)

    ax[0].legend()
    ax[1].legend()
    # # ax.set_xscale('log')
    # ax[1].plot(df4["rho"], df4["mean_train"], "--", color="b", label="Train MEE")
    # ax[1].fill_between(df4["rho"], df4["mean_train"] + df4["std_train"], df4["mean_train"] - df4["std_train"], color="b", alpha=0.2)
    # ax[1].plot(df4["rho"], df4["mean_test"], "-", color="g", label="Test MEE")
    # ax[1].fill_between(df4["rho"], df4["mean_test"] + df4["std_test"], df4["mean_test"] - df4["std_test"], color="g", alpha=0.2)
    # ax[1].set_xlabel("rho")
    # ax[1].set_ylabel("MEE")
    # ax[1].set_title("Poly degree = 4")
    plt.savefig('results/images/'+ title.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", ""), bbox_inches='tight')
    plt.show()
    # exit()
