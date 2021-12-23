import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


def create_df(path, training):
    if training:
        columns = ["id", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "x", "y"]
        # skipping first 7 rows as they are comments and not actual data
        df = pd.read_csv(path, skiprows=7, usecols=range(0, 13), names=columns)
    else:
        columns = ["id", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
        # skipping first 7 rows as they are comments and not actual data
        df = pd.read_csv(path, skiprows=7, usecols=range(0, 11), names=columns)

    return df


def split_data_target(df):
    # Select last 2 columns of dataframe
    # and first 10 columns
    N = 2
    M = 10

    data = df.iloc[:, : M]
    target_df = df.iloc[:, -N:]
    # data.plot()
    # target_df.plot()

    return data, target_df


def mean_euclidian_error_loss(y_true, pred_y):
    l2_norms = np.linalg.norm(y_true - pred_y, axis=1)
    return np.mean(l2_norms, axis=0)
