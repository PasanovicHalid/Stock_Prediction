import os
from re import X
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
from keras.layers import Dropout
import keras;
import math
from typing import Tuple, Dict


def import_data_from_csv(path: str):
    """
    Imports data from a csv file and returns a pandas dataframe.
    """
    return pd.read_csv(path)


def extract_features_required_for_training(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Extracts features required for training.
    """
    return df[features]


def split_data_into_training_and_test_sets(df: pd.DataFrame, window_start: int, window_end: int, test_size: float, columns: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Normalizes the entire data set, then splits it into training and test sets based on the given ratio.
    """
    data_count = window_end - window_start
    test_data_count = int(data_count * test_size)

    end_of_training_data = window_start + test_data_count

    df, max_values = normalize_data(df, columns, window_start, window_end)

    return df.iloc[window_start : end_of_training_data], df.iloc[end_of_training_data : window_end], max_values


def normalize_data(df: pd.DataFrame, columns: list[str], start: int, end: int):
    """
    Normalizes the data in the given segment.
    """
    max_values = {}
    for column in columns:
        max_value = df.iloc[start:end][column].max()
        df.loc[start:end, column] = df.loc[start:end, column] / max_value
        max_values[column] = max_value

    return df, max_values


def denormalize_data(data_array: np.ndarray, columns: list[str], max_values: dict):
    """
    Denormalizes the data in the given segment.
    """
    for column in columns:
        data_array[:] = data_array[:] * max_values[column]

    return data_array


def create_dataset(dataset: pd.DataFrame, time_step = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset.
    """
    dataX,dataY = [],[]

    for i in range(len(dataset)-time_step-1):
        dataX.append(dataset.iloc[i:(i+time_step), 0])
        dataY.append(dataset.iloc[i + time_step, 0])

    return np.array(dataX),np.array(dataY)


def build_model(input_shape):
    """
    Builds the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(75, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))

    model.add(LSTM(75, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(75, return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(75))
    model.add(Dropout(0.1))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(512, activation="relu")) 
    model.add(Dropout(0.1))

    model.add(Dense(1, activation="relu"))
    
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["mae", 'mape'])
    model.summary()

    return model


def plot_data(loaded_data, title: str, xlabel: str, ylabel: str, datacolumn: str):
    """
    Plots the data from the dataframe.
    """
    plt.figure(figsize=(16,6))
    plt.title(title)
    plt.plot(loaded_data[datacolumn])
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.show()


def plot_data_list(loaded_data_list, title: str, xlabel: str, ylabel: str, datacolumns: list):
    """
    Plots the data from the dataframe.
    """
    plt.figure(figsize=(16,6))
    plt.title(title)
    for loaded_data, datacolumn in zip(loaded_data_list, datacolumns):
        plt.plot(loaded_data, label=datacolumn)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend()
    plt.show()


def lstm_prediction(file: str):
    file_name = file.split("/")[-1]
    file_name_without_extension = file_name.split(".")[0]

    loaded_data = import_data_from_csv(file)
    extracted_data = extract_features_required_for_training(loaded_data, ["Adj Close"])

    time_step = 5
    number_of_intervals = 5
    percentage_of_test_interval_data = 0.8

    interval_size = int(extracted_data.shape[0] / number_of_intervals)

    X_training_data, Y_training_data = [], []
    X_test_data, Y_test_data = [], []
    max_values_list = []

    for i in range(number_of_intervals):
        start = i * interval_size
        end = start + interval_size

        if(i == number_of_intervals - 1):
            end = extracted_data.shape[0]

        training_data_interval, test_data_interval, max_values = split_data_into_training_and_test_sets(extracted_data, start, end, percentage_of_test_interval_data, ["Adj Close"])
        
        X_train, Y_train =  create_dataset(training_data_interval, time_step)
        X_test, Y_test =  create_dataset(test_data_interval, time_step)

        if(i != 0):
            X_train = np.concatenate((X_test_data[i - 1], X_train))
            Y_train = np.concatenate((Y_test_data[i - 1], Y_train))

        X_training_data.append(X_train)
        Y_training_data.append(Y_train)

        X_test_data.append(X_test)
        Y_test_data.append(Y_test)

        max_values_list.append(max_values)

    X_train_shape = (X_training_data[0].shape[1], 1)
    lstm_model = build_model(X_train_shape)

    predictions_list = []
    actuals_list = []

    for i in range(number_of_intervals):
        X_training_data_interval = X_training_data[i]
        Y_training_data_interval = Y_training_data[i]
        X_test_data_interval = X_test_data[i]
        Y_test_data_interval = Y_test_data[i]
        
        file_path = "algorithms/lstm/checkpoints/" + file_name_without_extension + "/lstm-" + str(i + 1) + ".keras"

        if(os.path.exists(file_path)):
            lstm_model = load_model(file_path)
        else:
            file_path_previos = "algorithms/lstm/checkpoints/" + file_name_without_extension + "/lstm-" + str(i) + ".keras"

            if(os.path.exists(file_path_previos)):
                lstm_model = load_model(file_path_previos)

            X_training_data_interval = np.reshape(X_training_data_interval, (X_training_data_interval.shape[0], X_training_data_interval.shape[1], 1))

            t_hist = lstm_model.fit(X_training_data_interval, Y_training_data_interval, batch_size = 15, epochs = 50)

            lstm_model.save(file_path)
        
        X_test_data_interval = np.reshape(X_test_data_interval, (X_test_data_interval.shape[0], X_test_data_interval.shape[1] ,1))
        predictions = lstm_model.predict(X_test_data_interval)
        rmse = np.sqrt(np.mean(((predictions - Y_test_data_interval) ** 2)))
        print("RSME: " + str(rmse))

        denormalized_predictions = denormalize_data(predictions, ["Adj Close"], max_values_list[i])
        denormalized_actuals = denormalize_data(Y_test_data_interval, ["Adj Close"], max_values_list[i])
        
        predictions_list = np.concatenate((predictions_list, denormalized_predictions.flatten()))
        actuals_list = np.concatenate((actuals_list, denormalized_actuals.flatten()))


    plot_data_list([predictions_list, actuals_list], 'Predictions vs Actual', 'Date', 'Adj Close Price USD ($)', ['Predictions', 'Actual'])


def main():
    """
    Main function.
    files = os.listdir("data/processed")

    for file in files:
        lstm_prediction("data/processed/" + file)
    """
    lstm_prediction("data/processed/AAPL.csv")
    


if __name__ == "__main__":
    main()