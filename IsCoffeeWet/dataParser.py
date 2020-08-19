import pandas as pd
import numpy as np
import warnings

# IMPORTANT: The dataset used mustn't contain trailing whitespaces


def mergeDateTime(dataset, dateName, timeName):
    """
    Merges the Date column and the Time column into a
    single one, used for indexing the dataframe with
    pandas.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to merge the date and time.
    - dateName: string.
        Name of the column with the dates. Example:
        >>> "Date"
    - timeName: string.
        Name of the column with the times. Example:
        >>> "Time"

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with a datetime column as its main
        index. Previous columns with date and time
        removed.
    """

    # Unifies "Date" and "Time" in a "datetime64[ns]"
    dataset["Datetime"] = pd.to_datetime(dataset[dateName] +
                                         ' ' + dataset[timeName])
    # Sets the datetime as the index of the DataFrame
    dataset = dataset.set_index('Datetime')
    # Drops the "Time" columns as it has been combine into "Date"
    dataset = dataset.drop([dateName, timeName], axis=1)

    return dataset


def cleanDataset(dataset, completeRows, nullValue):
    """
    Clears a dataset of all the damage values

    Returns a dataset without "damaged" values that could hurt 
    a convertion of a `dtype` in a column of the dataset

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to clean.
    - completeRows: list of strings.
        Columns where to search the null values. Example:
        >>> ["Temp Out", "Leaf Wet 1"]
    - nullValue:  object, generally string or int.
        Value to search and remove. Example:
        >>> "---"

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with entire dataset "repaired"
        (not null values other than NaNs)
    """
    for rowName in completeRows:
        # Replaces nullValue to NaN
        dataset[rowName] = dataset[rowName].replace(nullValue, np.NaN)

    return dataset


def convertNumeric(dataset, columnAndType):
    """
    Sets the type of a column according to the given pair.
    Use numeric values only (int, float, etc.) or it will fail.
    If the dataset is not clean (other values that are not NaN),
    the conversion will fail.

    Returns a dataset with the column casted to the desired
    values for easier manipulation.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to change the `dtype`.
    - columnAndFormat: list of tuples.
        Each tuple is made of 2 parameters: 
        The name (string) of the column and the `dtype` as a string 
        to be converted. Example: 
        >>> [("Temp Out", "float32"), ("Leaf Wet 1", "int32")]

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with columns type changed and missing
        values interpolated.

    Notes
    -----
    All the columns that contain NaN values must be converted to
    float (32 or 64) as this value doesn't exist for the interger.
    
    After the interpolation, an attemp to downcast the column type
    is made to save memory. This can be a problem if all the numbers
    in a column are >1000, as it will be downcasted to int; this can
    cause a lost of the decimals for that column, if it had.
    """
    # Changes the data to types to use less memory
    for nameAndType in columnAndType:
        # Cast the column to the wanted type
        dataset = dataset.astype({nameAndType[0]: nameAndType[1]})
        #TODO: round int values
        # Interpolates the NaN values
        dataset[nameAndType[0]] = dataset[nameAndType[0]].interpolate(
            method="time", limit_direction="forward", downcast="infer")

    return dataset


def sampleDataset(dataset, columnAndFunction, frequency):
    """
    Sample dataset according to a time frequency given. The dataset 
    will be group in interval of `frequency` and a numeric function will
    be applied to the rows between these timeframes to get one single value
    for each new timestamp.

    Different numeric functions can be applied to different column by 
    pairing the function with the column in a tuple in `columnAndFunction`.
    If a column is in the dataset but not in `columnAndFunction` or doesn't
    have a function assigned for the sampling, it will be ignore (and 
    removed) in the returned dataset.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to sample each column.
    - columnAndFunction: list of tuples.
        Each tuple is made of 2 parameters: 
        the name (string) of the column and the function to apply in
        the sample. Function can also be a string Example: 
        >>> [("Temp Out", np.mean)]
        >>> [("Leaf Wet 1", "last")]
    - frequency: string. Value expressed in string of the frequency to
        sample the data. Examples:
        >>> "1H" --An hour
        >>> "1D" --A day
        >>> "1M" --A month
        >>> "1h30min" --Combination of hours and minutes

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with the rows sample in the desired frequency.
    """
    # Creates the new dataset to return
    newDataset = pd.DataFrame()

    for filterFunction in columnAndFunction:
        # Generates a new DataFrame
        # By concatenating the column to the new DataFrame
        # As it gets sampled by the frequency
        newDataset = pd.concat([newDataset,
                                dataset.resample(
                                    frequency, label="right", closed="right").agg(
                                        {filterFunction[0]:filterFunction[1]})],
                               axis=1)

    #TODO: round int values
    # Clears the dataset from extra values generated while sampling
    newDataset = newDataset.interpolate(
        method="time", limit_direction="forward", downcast="infer")

    return newDataset
