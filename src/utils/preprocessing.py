import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


def cup_create_df(path, training):
    if training:
        columns = ["id", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "x", "y"]
        # skipping first 7 rows as they are comments and not actual data
        df = pd.read_csv(path, skiprows=7, usecols=range(0, 13), names=columns)
    else:
        columns = ["id", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
        # skipping first 7 rows as they are comments and not actual data
        df = pd.read_csv(path, skiprows=7, usecols=range(0, 11), names=columns)

    return df


def cup_split_data_target(df):
    # Select last 2 columns of dataframe
    # and first 10 columns
    N = 2
    M = 10

    data = df.iloc[:, : M]
    target_df = df.iloc[:, -N:]
    # data.plot()
    # target_df.plot()

    return data, target_df

def monk_create_df(path):
    columns = ["id", "Class", "a1", "a2", "a3", "a4", "a5", "a6", "monk_id"]
    # skipping first 7 rows as they are comments and not actual data
    df = pd.read_csv(path, names=columns, delimiter=" ")
    df = df.drop('id', axis='columns')
    df = df.drop("monk_id", axis='columns')
    return df


def monk_split_data_target(df):
    y = df["Class"]
    x = df.drop("Class", axis=1)
    return x.to_numpy(), y.to_numpy()

def mean_euclidian_error_loss(y_true, pred_y):
    l2_norms = np.linalg.norm(y_true - pred_y, axis=1)
    return np.mean(l2_norms, axis=0)
