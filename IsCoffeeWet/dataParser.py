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


def convertNumeric(dataset, columnAndType, nullValue):
    """
    Sets the type of a column according to the given pair.
    Supported types for conversion are `float`, `signed`
    and `unsigned`the moment. It's important that the index
    is time-base as the interpolation uses time.

    Returns a dataset with the column casted to the desired
    values for easier manipulation.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to change the `dtype`.
    - columnAndType: list of tuples.
        Each tuples consists of 2 values: name of the column
        and name type of data for the column (signed, unsigned 
        or float).
        Example:
        >>> [("Temp Out", "float"), ("Leaf Wet 1", "signed")]
    - nullValue:  object, generally string or int.
        Value to search and remove. Example:
        >>> "---"

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with columns type changed and missing
        values interpolated.

    Notes
    -----
    The precision of the converted dataset is low (32bits or lower),
    but higher precision isn't needed with the values that the sensors
    capture. Downcast is used to restrict the data type (float use
    as int when possible) and the memory footprint (float64 to float32)

    This also helps identify problems when data consistency is getting 
    lost. An example is parameters that are purely integers, if an 
    operation generates decimals in those numbers, it can be corrected.
    """
    # Sets all "nullValues" to NaN
    dataset = dataset.replace(nullValue, np.NaN)

    # Changes the data to types to use less memory
    for nameAndType in columnAndType:
        # Casting of the column type
        dataset = dataset.astype({nameAndType[0]: "float32"})

        # Interpolation of NaNs
        dataset[nameAndType[0]] = dataset[nameAndType[0]].interpolate(
            method="time", limit_direction="forward")

        # Downcast of the types
        if nameAndType[1] == "signed":

            # Round the number to remain integer after interpolation
            dataset[nameAndType[0]] = dataset[nameAndType[0]].round()
            # Downcast it to integer properly
            dataset[nameAndType[0]] = pd.to_numeric(
                dataset[nameAndType[0]], downcast="signed")

        elif nameAndType[1] == "unsigned":

            # Round the number to remain integer after interpolation
            dataset[nameAndType[0]] = dataset[nameAndType[0]].round()
            # Downcast it to integer properly
            dataset[nameAndType[0]] = pd.to_numeric(
                dataset[nameAndType[0]], downcast="unsigned")
        else:

            # Downcast it to integer properly
            dataset[nameAndType[0]] = pd.to_numeric(
                dataset[nameAndType[0]], downcast="float")

    return dataset


def sampleDataset(dataset, columnAndFunction, frequency):
    """
    Sample dataset according to a time frequency given. The dataset 
    will be group in interval of `frequency` and a numeric function will
    be applied to the rows between these timeframes to get one single value
    for each new timestamp.

    Different numeric functions can be applied to different column by 
    pairing the function with the column in a tuple in `columnAndFunction`.

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

    Notes
    -----
    The returned database may have empty rows if the dataset isn't 
    complete. It's recommended to use again `convertNumeric` function
    to interpolate the empty values and maintain the data type consistency.

    """
    # Creates the new dataset to return
    newDataset = pd.DataFrame()

    for filterFunction in columnAndFunction:
        # Generates a new DataFrame
        # Each resampling creates a column, so it is appended to get a
        # final result. It must be done this way to use a different
        # function in each column
        newDataset = pd.concat([newDataset,
                                dataset.resample(
                                    frequency, label="right", closed="right").agg(
                                        {filterFunction[0]:filterFunction[1]})],
                               axis=1)

    return newDataset


def cyclicalEncoder(dataset, encodeDays, encodeHours):

    datetime = dataset.index

    if encodeDays:
        dataset["days_sin"] = np.sin(2 * np.pi * datetime.dayofyear / 365)
        dataset["days_cos"] = np.cos(2 * np.pi * datetime.dayofyear / 365)

    if encodeHours:
        dataset["hours_sin"] = np.sin(2 * np.pi * datetime.hour / 23)
        dataset["hours_cos"] = np.cos(2 * np.pi * datetime.hour / 23)

    return dataset
