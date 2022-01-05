from sklearn.model_selection import KFold
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
import itertools as it
import pandas as pd
from src.utils.plots import plot_learning_curves_mlp

from multiprocessing import Pool
import multiprocessing as mp
#GLOBAL VARIABLES

EPOCHS = 5000
VALIDATION_SPLIT_MONKS = 0.15
VALIDATION_SPLIT_CUP = 0.2
TEST_SIZE_MLCUP = 0.2
EARLY_STOPPING_PATIENCE_CUP = 40
EARLY_STOPPING_PATIENCE_MONKS = 500
KFOLD_SPLITS=5
WORKERS_POOL=100

stop_early_monks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE_MONKS)
monks_callbacks = [stop_early_monks]

stop_early_cup = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE_CUP)
cup_callbacks = [stop_early_cup]

logging.basicConfig(filename='./results/mlp/results.log', filemode='a+', level=logging.INFO)

def create_monks_nn(units=1, weights=0.5, lambda_reg=0.0, learning_rate=0.01):
    model = Sequential()
    negative = -1 * weights
    custom_weight_matrix = tf.keras.initializers.RandomUniform(
        minval=-negative, maxval=weights, seed=None
    )

    model.add(Dense(units, activation='tanh',
                    kernel_initializer=custom_weight_matrix,
                    kernel_regularizer=l2(lambda_reg)))
    # output
    model.add(Dense(1, activation='tanh', kernel_initializer=custom_weight_matrix,
                    kernel_regularizer=l2(lambda_reg)))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])

    return model

def mean_euclidian_error_loss(y_true, y_pred):
    error = y_true - y_pred
    l2_norm = tf.norm(error, ord=2, axis=1)
    return tf.keras.backend.mean(l2_norm, axis=0)


def create_cup_nn(units=1, layers=1, lambda_reg=0.0, activation_function='relu', learning_rate=0.01):
    model = Sequential()

    if activation_function == 'relu':
        act = tf.keras.layers.ReLU()
    elif activation_function == 'elu':
        act = tf.keras.layers.ELU()
    elif activation_function == 'leakyrelu':
        act = tf.keras.layers.LeakyReLU()

    for i in range(0, layers):
        model.add(tf.keras.layers.Dense(units=units, activation=act,
                    kernel_regularizer=l2(lambda_reg)))

    model.add(tf.keras.layers.Dense(2, activation='linear', kernel_regularizer=l2(lambda_reg)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=mean_euclidian_error_loss)

    return model

def mlcup_worker(parameters):
    elem = parameters[0]
    input_train_df = parameters[1]
    target_train_df = parameters[2]
    kf = parameters[3]
    n_combination = parameters[4]
    return_list = parameters[5]
    j = parameters[6]

    units = elem[0]
    layers = elem[1]
    learning_rate = elem[2]
    activation_function = elem[3]
    lambda_reg = elem[4]

    val_loss_list = []
    k = 1

    for train_index, test_index in kf.split(input_train_df):
        print(f"combination {j}/{n_combination} fold {k}/{kf.n_splits}")
        k += 1
        X_train_split, X_test_split = input_train_df.iloc[train_index, :], input_train_df.iloc[test_index, :]
        y_train_split, y_test_split = target_train_df.iloc[train_index, :], target_train_df.iloc[test_index, :]

        model = create_cup_nn(units=units, learning_rate=learning_rate,
                                 activation_function=activation_function, layers=layers,
                                 lambda_reg=lambda_reg)

        history = model.fit(X_train_split, y_train_split,
                            validation_data=(X_test_split, y_test_split),
                            epochs=EPOCHS,
                            callbacks=cup_callbacks,
                            verbose=0)

        val_loss_list.append(history.history["val_loss"][-1])

    np_array = np.array(val_loss_list)

    mean = np.mean(np_array)
    std = np.std(np_array)

    return_list.append([units, layers, learning_rate,activation_function, lambda_reg, mean, std])



def monks_worker(parameters):
    elem = parameters[0]
    X_train = parameters[1]
    y_train = parameters[2]
    kf = parameters[3]
    n_combination = parameters[4]
    return_list = parameters[5]
    j = parameters[6]
    dataset = parameters[7]

    units = elem[0]
    weights = elem[1]
    learning_rate = elem[2]
    lambda_reg = elem[3]

    val_loss_list = []
    val_accuracy_list = []
    k = 1
    for train_index, test_index in kf.split(X_train):
        print(f"combination {j}/{n_combination} fold {k}/{kf.n_splits}")

        k += 1
        X_train_split = X_train.iloc[train_index, :]
        X_test_split = X_train.iloc[test_index, :]

        y_train_split, y_test_split = y_train[train_index], y_train[test_index]

        model = create_monks_nn(units=units, learning_rate=learning_rate,
                                weights=weights, lambda_reg=lambda_reg)

        history = model.fit(X_train_split, y_train_split,
                            validation_data=(X_test_split, y_test_split),
                            epochs=EPOCHS,
                            callbacks=monks_callbacks,
                            verbose=0)

        val_loss_list.append(history.history["val_loss"][-1])
        val_accuracy_list.append(history.history["val_accuracy"][-1])

    val_loss_np = np.array(val_loss_list)
    mean_val_loss = np.mean(val_loss_np)
    std_val_loss = np.std(val_loss_np)

    val_accuracy_np = np.array(val_accuracy_list)
    mean_val_accuracy = np.mean(val_accuracy_np)
    std_val_accuracy = np.std(val_accuracy_np)

    return_list.append(
        [units, weights, learning_rate, lambda_reg, mean_val_loss, std_val_loss, mean_val_accuracy, std_val_accuracy,
         dataset])


def mlcup_model_selection(X_train, y_train):

    param_grid = {
        "units": [1, 4, 16, 64, 128, 256, 512],
        'layers': [1, 2, 3, 4, 5],
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "activation_function": ['relu', 'elu', 'leakyrelu'],
        "lambda_reg": [0.0, 0.00001, 0.0001, 0.001, 0.005, 0.01]
    }

    keys = param_grid.keys()
    combinations = it.product(*(param_grid[key] for key in keys))

    param_list = list(combinations)
    df_output_columns = list(param_grid.keys())
    dataframe_output_columns = df_output_columns
    dataframe_output_columns.append("mean_val_loss")
    dataframe_output_columns.append("std_val_loss")

    kfold_cv_df = pd.DataFrame(columns=dataframe_output_columns)

    kf = KFold(n_splits=KFOLD_SPLITS, random_state=None)

    n_combination = len(param_list)

    # transform back x_train, y_train to dataframe
    columns_input = ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]
    columns_target = ["x", "y"]

    input_train_df = pd.DataFrame(X_train, columns=columns_input)
    target_train_df = pd.DataFrame(y_train, columns=columns_target)

    manager = mp.Manager()
    return_list = manager.list()

    args = []
    j = 1

    pool = Pool(WORKERS_POOL)

    for elem in param_list:
        tmp = [elem, input_train_df, target_train_df, kf, n_combination, return_list, j]
        j += 1
        args.append(tmp)

    pool.map(mlcup_worker, args)

    for elem in return_list:
        df_new_line = pd.DataFrame([elem],
                                   columns=dataframe_output_columns)

        kfold_cv_df = pd.concat([kfold_cv_df, df_new_line], ignore_index=True)

    kfold_cv_df.to_csv("./results/mlp/cup_results_GS.csv")

    # get optimal hyperparameter values according to the minimum validation loss
    optimal_df = kfold_cv_df[kfold_cv_df.mean_val_loss == kfold_cv_df.mean_val_loss.min()]

    units = optimal_df["units"].values[0]
    layers = optimal_df["layers"].values[0]
    learning_rate = optimal_df["learning_rate"].values[0]
    activation_function = optimal_df["activation_function"].values[0]
    lambda_reg = optimal_df["lambda_reg"].values[0]

    logging.info(
        f"MLCUP - lowest validation loss: {optimal_df['mean_val_loss'].values[0]} std {optimal_df['mean_val_loss'].values[0]} with units:{units} layers:{layers} learning_rate:{learning_rate} activation_function:{activation_function} lambda:{lambda_reg}")

    #return optimal hyperpatameters

    return optimal_df


def mlcup_model_assessment(optimal_df, X_train, y_train, X_test, y_test):
    print(optimal_df)
    units = optimal_df["units"].values[0]
    layers = optimal_df["layers"].values[0]
    learning_rate = optimal_df["learning_rate"].values[0]
    activation_function = optimal_df["activation_function"].values[0]
    lambda_reg = optimal_df["lambda_reg"].values[0]

    # retrain on all training set and use validation set for early stopping
    model = create_cup_nn(units=units, layers=layers, lambda_reg=lambda_reg, learning_rate=learning_rate,
                             activation_function=activation_function)
    hisotry = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=VALIDATION_SPLIT_CUP,
                        callbacks=cup_callbacks)
    path="./results/mlp/images/"
    plot_learning_curves_mlp(hisotry, path, "cup")

    model.save("./results/mlp/models/cup_model")

    # evaluate on the test set
    score = model.evaluate(X_test, y_test)
    print("Internal test set result for MLcup parallelized: {}\n".format(score))


def monks_model_selection(X_train, y_train, monks_counter):
    param_grid = {
        "units": [1, 2, 3, 4, 5],
        "weights": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "lambda_reg": [0.0, 0.00001, 0.0001, 0.001, 0.01]
    }

    keys = param_grid.keys()
    combinations = it.product(*(param_grid[key] for key in keys))

    param_list = list(combinations)
    df_output_columns = list(param_grid.keys())
    dataframe_output_columns = df_output_columns
    dataframe_output_columns.append("mean_val_loss")
    dataframe_output_columns.append("std_val_loss")
    dataframe_output_columns.append("mean_val_accuracy")
    dataframe_output_columns.append("std_val_accuracy")
    dataframe_output_columns.append("dataset")

    kfold_cv_df = pd.DataFrame(columns=dataframe_output_columns)

    pool = Pool(WORKERS_POOL)

    kf = KFold(n_splits=KFOLD_SPLITS, random_state=None)
    n_combination = len(param_list)

    manager = mp.Manager()
    return_list = manager.list()

    print("Starting KFold CV for Monk {}".format(monks_counter))

    args = []
    j = 1

    if monks_counter == 1:
        dataset = 'monks1'

    if monks_counter == 2:
        dataset = 'monks2'

    if monks_counter == 3:
        dataset = 'monks3'

    # create a list of list containing
    for elem in param_list:
        tmp = [elem, X_train, y_train, kf, n_combination, return_list, j, dataset]
        j += 1
        args.append(tmp)

    pool.map(monks_worker(), args)

    for elem in return_list:
        df_new_line = pd.DataFrame([elem],
                                   columns=["units", "weights", "learning_rate", "lambda_reg", "mean_val_loss",
                                            "std_val_loss",
                                            "mean_val_accuracy", "std_val_accuracy", "dataset"])
        kfold_cv_df = pd.concat([kfold_cv_df, df_new_line], ignore_index=True)

    print("KFold CV for Monk {} end".format(monks_counter))

    restricted_df = kfold_cv_df[kfold_cv_df.dataset == dataset]

    # saving the results into a csv file
    restricted_df.to_csv("./results/mlp/Monk_{}_results_GS.csv".format(monks_counter))

    # get optimal hyperparameter values according to the max validation accuracy
    optimal_df = restricted_df[restricted_df.mean_val_accuracy == restricted_df.mean_val_accuracy.max()]

    return optimal_df

def monks_model_assessment(optimal_df, X_train, y_train, X_test, y_test, monks_counter):
    units = optimal_df["units"].values[0]
    weights = optimal_df["weights"].values[0]
    learning_rate = optimal_df["learning_rate"].values[0]
    lambda_reg = optimal_df["lambda_reg"].values[0]

    logging.info(
        f"Highest validation accuracy: {optimal_df['mean_val_accuracy'].values[0]} std {optimal_df['std_val_accuracy'].values[0]} {units} weights {weights} learning_rate {learning_rate} lambda {lambda_reg}")

    model = create_monks_nn(units=units, weights=weights, lambda_reg=lambda_reg, learning_rate=learning_rate)
    hisotry = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT_MONKS,
                        callbacks=monks_callbacks,
                        verbose=0)

    path = "./results/mlp/images/"
    plot_learning_curves_mlp(hisotry, path, f"Monks_{monks_counter}")

    model.save("./results/mlp/models/Monks_{}_model".format(monks_counter))

    score = model.evaluate(X_test, y_test)

    print(f"Monks_{monks_counter} evaluate result accuracy: {score[1]} loss: {score[0]} \n")