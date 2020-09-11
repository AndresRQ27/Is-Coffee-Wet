import pandas as pd
import numpy as np
import datetime

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
                                         ' ' + dataset[timeName],
                                         dayfirst=True)
    # Sets the datetime as the index of the DataFrame
    dataset = dataset.set_index('Datetime')
    # Drops the "Time" columns as it has been combine into "Date"
    dataset = dataset.drop([dateName, timeName], axis=1)

    return dataset


def convertNumeric(dataset, columnAndType, nullList):
    """
    Sets the type of a column according to the given pair.
    Supported types for conversion are `float` and `int`
    the moment. It's important that the index is time-base
    as the interpolation uses time.

    Returns a dataset with the column casted to the desired
    values for easier manipulation.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to change the `dtype`.
    - columnAndType: list of tuples.
        Each tuples consists of 2 values: name of the column
        and name type of data for the column (int or float).
        Example:
        >>> [("Temp Out", "float"), ("Leaf Wet 1", "signed")]
    - nullList:  list, generally string or int.
        List of values to search and remove. Example:
        >>> ["---", "------"]

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with columns type changed and missing
        values interpolated.
    """
    # Sets all "nullValues" to NaN
    for nullValue in nullList:
        dataset = dataset.replace(nullValue, np.NaN)

    # Changes the data to types to use less memory
    for nameAndType in columnAndType:
        try:
            # Casting of the column type
            dataset = dataset.astype({nameAndType[0]: "float64"})

            # Interpolation of NaNs
            dataset[nameAndType[0]] = dataset[nameAndType[0]].interpolate(
                method="time", limit_direction="forward")

            # Round the integers of the types
            if nameAndType[1] == "int":
                dataset[nameAndType[0]] = dataset[nameAndType[0]].round()

        except KeyError:
            print("Column \"{}\" not found in dataset".format(nameAndType[0]))

    # Applying infer_objects() function.
    dataset = dataset.infer_objects()

    return dataset


def sampleDataset(dataset, columnAndFunction, frequency):
    """
    Sample dataset according to a time frequency given. The dataset
    will be group in interval of `frequency` and a numeric function will
    be applied to the rows between these timeframes to get one single value
    for each new timestamp.

    Different numeric functions can be applied to different column by
    pairing the function with the column in a tuple in `columnAndFunction`.
    The return value is a sampled dataset into the given frequency.

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

    for nameAndFunction in columnAndFunction:
        # Generates a column of accumulated time
        # the leaf has been wet.
        if nameAndFunction[0] == "Leaf Wet Accum":
            # Difference between last and current datetime
            timeDiff = dataset.index[1:] - dataset.index[:-1]

            # Creates a new column with 0s
            dataset["Leaf Wet Accum"] = 0

            # Sets to 1 the columns that has wetness
            dataset.loc[dataset["Leaf Wet 1"]
                        > 0, "Leaf Wet Accum"] = 1

            # There is no data about the delta time of the first
            # measurement, so use the difference of the second one
            # as a hard guess. We divide by 60 to use minutes
            dataset.loc[dataset.index[0],
                        "Leaf Wet Accum"] *= timeDiff[0].seconds/60
            dataset.loc[dataset.index[1:],
                        "Leaf Wet Accum"] *= timeDiff.seconds/60

        # Generates a new DataFrame (DF)
        # Each resampling creates a column, so it is appended to get a
        # final result. It must be done this way to use a different
        # function in each column
        try:
            auxiliarDF = dataset.resample(frequency, label="right",
                                          closed="right", origin="start"
                                          ).agg({nameAndFunction[0]: nameAndFunction[1]})

            newDataset = pd.concat([newDataset, auxiliarDF], axis=1)

        except KeyError:
            print("Column \"{}\" not found in dataset".format(
                nameAndFunction[0]))

    # If the value is NaN in "Leaf Wet 1"
    # set "Leaf Wet Accum" to NaN as well
    # This is to use interpolation later in these values
    newDataset.loc[newDataset["Leaf Wet 1"].isna(),
                   "Leaf Wet Accum"] = np.NaN

    return newDataset


def cyclicalEncoder(dataset, encodeList):
    """
    Converts values that have cyclical behavior into pair of sin(x)
    and cos(x) functions. Day and time is extracted from a datetime
    index, so be sure to have it. To understand the purpose of this
    function, see `Notes`.

    The return dataset has a decomposition into sin(x) and cos(x) of
    the desired frequencies.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to extract the time from it's datetime index.
    - encodeList: list of tuples.
        Each tuple consists of two values: name of the frequency to
        name the column and the frequency value. The timestamp starts
        from seconds, so start the conversion from that point. Example:
        >>> [("Day", 60*60*24)]

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with the decomposition of sin(x) and cos(x) of
        the day and/or hour.

    Notes
    -----
    The purpose is that the NN see all days equally separated from
    themselfs, as the last day of the year is one day away of the first
    day; same happens with the hours and the minutes of a day. By using
    integers, this isn't obvious to the NN, so we help him by parsing the
    time in a more easy-to-see way.

    Moreover, compared to the previous version that covered values between
    0 and 2*pi, the actual function represents the full date in a 
    continuos wave. This helps to see de date as cyclical, while having
    a difference between (e.g.) 04-May-2014 and 04-May-2020, without taking
    into account the year directly.

    A frequency analysis of the data using the function "freqDomain" is
    strongly recommended.
    """
    timestamp_s = dataset.index.map(datetime.datetime.timestamp)

    for encoder in encodeList:
        dataset[encoder[0] +
                ' sin'] = np.sin(timestamp_s * (2 * np.pi / encoder[1]))
        dataset[encoder[0] +
                ' cos'] = np.cos(timestamp_s * (2 * np.pi / encoder[1]))

    return dataset


def dayEncoder(dataset, datetime):
    # TODO: documentation
    # TODO: update function to use the new method
    for date in datetime:
        dataset.loc[datetime, "days_sin"] = np.sin(
            2 * np.pi * datetime.dayofyear / 365)
        dataset.loc[datetime, "days_cos"] = np.cos(
            2 * np.pi * datetime.dayofyear / 365)


def hourEncoder(dataset, datetime):
    # TODO: documentation
    # TODO: update function to use the new method
    for date in datetime:
        dataset.loc[datetime, "hours_sin"] = np.sin(
            2 * np.pi * datetime.hour / 23)
        dataset.loc[datetime, "hours_cos"] = np.cos(
            2 * np.pi * datetime.hour / 23)
