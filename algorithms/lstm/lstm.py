import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model

import math
from sklearn.preprocessing import MinMaxScaler

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

def split_data_into_training_and_test_sets(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and test sets based on the given ratio.
    """
    train_size = int(len(df) * train_ratio)
    train_data = df.iloc[0:train_size]
    test_data = df.iloc[train_size:len(df)]
    return train_data, test_data

def create_dataset(dataset: pd.DataFrame, time_step = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a dataset.
    """
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset.iloc[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset.iloc[i + time_step, 0])
    return np.array(dataX),np.array(dataY)

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

def main():
    scaler = MinMaxScaler(feature_range=(0,1))
     
    loaded_data = import_data_from_csv("data\\unprocessed\\AAPL.csv")
    extracted_data = extract_features_required_for_training(loaded_data, ["Adj Close"])
    # Plot the data
    plot_data(extracted_data, 'Adj Close Price History', 'Date', 'Adj Close Price USD ($)', 'Adj Close')

    scaled_data = scaler.fit_transform(extracted_data)
    scaled_data_df = pd.DataFrame(scaled_data, columns=extracted_data.columns)

    training_data, test_data = split_data_into_training_and_test_sets(scaled_data_df, 0.75)

    plot_data_list([training_data, test_data], 'Training and Test Data', 'Date', 'Adj Close Price USD ($)', ['Adj Close', 'Adj Close'])

    time_step = 60
    X_train,Y_train =  create_dataset(training_data,time_step)
    X_test,Y_test =  create_dataset(test_data,time_step)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences = True))
    model.add(LSTM(50, return_sequences = True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    model.fit(X_train, Y_train, batch_size=1, epochs=10)
    model.save('lstm_stock_predict_v1.keras')
    """

    try:
        # Load the model
        model = load_model('lstm_stock_predict_v1.keras')
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

    if model is not None:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] ,1))
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
        print(rmse)
        plot_data_list([predictions, Y_test], 'Predictions vs Actual', 'Date', 'Adj Close Price USD ($)', ['Predictions', 'Actual'])
    else:
        print("Model is None, cannot make predictions")


if __name__ == "__main__":
    main()