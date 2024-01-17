import pandas as pd;
import os;

rolling_mean_window : int = 9
path_to_data : str = "data/"

def get_files_in_directory(path: str):
    """
    Returns a list of files in a directory.
    """
    return os.listdir(path)

def import_data_from_csv(path: str):
    """
    Imports data from a csv file and returns a pandas dataframe.
    """
    return pd.read_csv(path)

def export_data_to_csv(df: pd.DataFrame, path: str):
    """
    Exports data to a csv file.
    """
    df.to_csv(path, index=False)
    

def convert_date_column_to_datetime(df: pd.DataFrame):
    """
    Converts the date column to datetime format.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def handle_null_values_in_date_column(df: pd.DataFrame):
    """
    Handles null values in date column by putting the date of the previous row and adding 1 day.
    """

    for i in range(len(df)):
        if pd.isnull(df.loc[i, "Date"]):
            if i == 0:
                df.loc[i, "Date"] = df.loc[i+1, "Date"] - pd.DateOffset(1) # type: ignore
            else:
                df.loc[i, "Date"] = df.loc[i-1, "Date"] + pd.DateOffset(1) # type: ignore

    return df

def handle_missing_days_in_date_column(df: pd.DataFrame):
    """
    Handles missing days in date column by adding the missing days 
    """
    new_rows = []

    for i in range(len(df) - 1):
        if i == 0:
            continue
        else:      
            time_delta : pd.Timedelta = df.loc[i, "Date"] - df.loc[i-1, "Date"]       # type: ignore
            if time_delta > pd.Timedelta(days=1):
                for j in range(1, time_delta.days):
                    new_row = df.loc[i - 1].copy()
                    new_row["Date"] = df.loc[i - 1, "Date"] + pd.DateOffset(j) # type: ignore
                    new_row["Volume"] = 0
                    new_rows.append(new_row)
                
    
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df.sort_values(by="Date", inplace=True, ignore_index=True)

    return df

def rolling_mean_over_columns(df: pd.DataFrame, columns: list[str], window: int):
    """
    Calculates the rolling mean over a list of columns.
    """
    column: str
    for column in columns:
        df[column] = df[column].rolling(window=window, center=True,).mean()
    
    return df

def normalize_data(df: pd.DataFrame, columns: list[str]):
    """
    Normalizes the data.
    """
    column: str
    for column in columns:
        df[column] = df[column] / df[column].max()
    
    return df

def drop_nan_rows(df: pd.DataFrame):
    """
    Drops rows with NaN values.
    """
    return df.dropna(inplace=False)

def validate_if_date_column_has_increasing_values(df: pd.DataFrame):
    """
    Validates if the date column has increasing values.
    """
    for i in range(len(df) - 1):
        time_delta : pd.Timedelta = df.loc[i, "Date"] - df.loc[i+1, "Date"]       # type: ignore
        if time_delta != pd.Timedelta(days=-1):
            return False
    return True
    

def main():
    files = get_files_in_directory("{}unprocessed".format(path_to_data))

    for file in files:
        df = import_data_from_csv("{}unprocessed/".format(path_to_data) + file)
        df = convert_date_column_to_datetime(df)
        df = handle_null_values_in_date_column(df)
        df = handle_missing_days_in_date_column(df)
        df = rolling_mean_over_columns(df, ["Open", "High", "Low", "Close", "Adj Close"], rolling_mean_window)
        df = normalize_data(df, ["Open", "High", "Low", "Close", "Adj Close", "Volume"])
        df = drop_nan_rows(df)
        df = df.reset_index(drop=True)
        
        if not validate_if_date_column_has_increasing_values(df):
            print("Date column in " + file + " does not have valid date values.")
            continue

        export_data_to_csv(df, "{}processed/".format(path_to_data) + file)

if __name__ == "__main__":
    main()








