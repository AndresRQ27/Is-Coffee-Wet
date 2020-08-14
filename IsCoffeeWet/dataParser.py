import pandas as pd
import numpy as np


def cleanDataset(dataset, removeRows = [], 
                 nullValue = "", removeColumns = []):
    """Removes missing or empty values that can 
    generate error in the parse of the dataset

    Args:
        dataset (pd.DataFrame): Dataset to clean
        removeRows (string): Columns where to search the null values
        nullValue (string or number): Value to search and remove
        removeColumns (string): Group of columns to remove

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Removes the columns with almost all missing values
    displayNumber = 0 #Iterator to print
    if removeColumns: #Checks if empty
        for columnName in removeColumns: #Remove 
            try:
                dataset = dataset.drop(columnName, axis=1)
                displayNumber += 1
            except (KeyError):
                print("Column doesn't exist:", columnName)

    print("\nColumns removed:", displayNumber)

    # Sets al the missing values (represented as nullValue) to NaN in the
    # selected columns
    displayNumber = dataset.size #Iterator to print
    if removeRows:
        for rowName in removeRows:
            try:
                dataset = dataset.replace(
                                {rowName: nullValue}, # Columns where missing data could be
                                np.NaN)  # New value
            except (KeyError):
                print("Column doesn't exist:", columnName)
    
    dataset = dataset.dropna(axis=0,        # Drop only the rows
                             how="any")     # Removes the NaN values in the rows

    print("Rows removed:", displayNumber-dataset.size)
                             
    return dataset


def setDataTypes(dataset):
    """Reorders the dataset by merging Date & Time into a pd.datetime64[ns]
    Sets the types of the rest of the columns

    Args:
        dataset (pd.DataFrame): Dataset with the columns to change the type

    Returns:
        pd.DataFrame: dataset with all the columns set to a known type 
                      (not just "object")
    """
    # Unifies "Date" and "Time" into a "Date" as pandas uses "datetime64[ns]"
    dataset['Date'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'])
    # Changes the data to types that use less memory
    dataset = dataset.astype({"Temp Out": "float32",
                              # ?"columnName"": "---",
                              "Leaf Wet 1": "int32"})
    # Drops the "Time" columns as it has been combine into "Date"
    dataset = dataset.drop("Time", axis=1)

    return dataset


def unifyTime(dataset):
    # ***Group the rows in sequences of every 15 minutes
    dataset.resample('15min', origin = "start").agg(
                    {'Temp Out': np.mean})
    return dataset

#TEST: prints
# print(dataset.info(verbose=True))
