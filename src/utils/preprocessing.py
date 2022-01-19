import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

    #shuffle dei dati
    df = df.sample(frac=1).reset_index(drop=True)

    df = df.drop('id', axis='columns')
    df = df.drop("monk_id", axis='columns')
    return df


def monk_split_data_target(df):
    y = df["Class"]
    x = df.drop("Class", axis=1)
    return x.to_numpy(), y.to_numpy()


def mean_euclidian_error_loss(y_true, pred_y):
    if y_true.ndim > 1:
        l2_norms = np.linalg.norm(y_true - pred_y, axis=1)
        return np.mean(l2_norms, axis=0)
    else:
        return np.mean(np.abs(y_true - pred_y))

def create_monks_df(path):
    columns = ["class", "a1", "a2", "a3", "a4", "a5", "a6", "id"]
    df = pd.DataFrame(columns=columns)

    # Using readline()
    file1 = open(path, 'r')

    i = 0
    while True:
        # Get next line from file
        line = file1.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        elems = line.strip().split(" ")
        df.loc[i] = elems
        i += 1

    file1.close()

    return df


def create_monks_dataset(df):
    for elem in df.columns:
        if elem == "id":
            pass
        else:
            df[elem] = df[elem].astype(int)

    target = df["class"]
    tmp_df = df.drop('class', inplace=False, axis=1)
    tmp_df = tmp_df.drop('id', inplace=False, axis=1)

    # one hot encoding
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(tmp_df)

    data = one_hot_encoder.transform(tmp_df)

    list_of_cat = one_hot_encoder.categories_

    j = 1
    columns = []
    for lst in list_of_cat:
        for elem in lst:
            columns.append("column" + str(j))
            j += 1

    df_encoded = pd.DataFrame(data=data, columns=columns)

    return df_encoded, target