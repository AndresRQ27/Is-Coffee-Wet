import numpy as np
import pandas as pd


def merge_datetime(dataset, config_file):
    """
    Merges the Date column and the Time column into a single one, used for
    indexing the dataframe with pandas.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset to merge the date and time.
    config_file: config_file.ConfigFile
        Object with the needed information to merge the dataset

    Returns
    -------
    pandas.DataFrame
        Dataset with a datetime column as its main
        index. Previous columns with date and time
        removed.
    """

    # Initialize the Series with the first column in the list
    name = config_file.datetime[0]
    datetime_column = pd.Series(dataset[name])

    # Drop the column
    dataset = dataset.drop(name, axis=1)
    config_file.columns.remove(name)

    # Check if the labels has the column in case it was
    # copied from the columns
    if name in config_file.labels:
        config_file.labels.remove(name)

    # Unifies the other columns
    for name in config_file.datetime[1:]:
        # This operates on all rows
        datetime_column += " "
        datetime_column += pd.Series(dataset[name])
        # ! Add more columns if the format ever needs it

        # Drops the added column
        dataset = dataset.drop(name, axis=1)
        config_file.columns.remove(name)

        # Check if the labels has the column in case it was
        # copied from the columns
        if name in config_file.labels:
            config_file.labels.remove(name)

    # Converts the Series into a "datetime64[ns]" column
    dataset["Datetime"] = pd.to_datetime(datetime_column,
                                         format=config_file.datetime_format)

    # Sets the datetime as the index of the DataFrame
    dataset = dataset.set_index("Datetime")

    return dataset


def convert_numeric(dataset, config_file):
    """
    Sets the type of a column according to the given pair. Supported types for
    conversion are `float` and `int` the moment. It's important that the index
    is time-base as the interpolation uses time.

    Returns a dataset with the column casted to the desired values for easier
    manipulation.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset to change the `dtype`.
    config_file: config_file.ConfigFile
        Object with the needed information to give format

    Returns
    -------
    pandas.DataFrame
        Dataset with columns type changed and missing
        values interpolated.
    """

    # Sets all "nullValues" to NaN
    for null in config_file.null_list:
        dataset = dataset.replace(null, np.NaN)

    # Changes the data to float64
    for name in config_file.columns:
        # Check if it's an special format (int, bool, others)
        if name in config_file.formats:
            # Ignores interpolation of bool types
            if config_file.formats[name] == "bool":
                continue
            # ! With elif, you can add other types

        # Casting of the column type
        dataset = dataset.astype({name: "float64"})

        # Interpolation of NaNs
        dataset[name] = dataset[name].interpolate(method="time",
                                                  limit_direction="forward")

    # Apply format to the special cases
    for name, value in config_file.formats.items():
        if value == "int":
            dataset[name] = dataset[name].round()
        # ! Add elif to evaluate other conditions
        else:
            continue

    return dataset


def sample_series(series, config_file):
    """
    Sample a series according to a time frequency given. The series will be
    group in interval of `frequency` and a numeric function will be applied
    to the rows between these timeframes to get one single value for each
    new timestamp.

    Different numeric functions can be applied to different series by
    pairing the function with the name of the series in a tuple in
    `column_function`. The return value is a sampled dataset into the given
    frequency.

    Parameters
    ----------
    series: pandas.Series 
        Series (column of the dataset) to sample.
    config_file: config_file.ConfigFile 
        Object with the needed information to resample the series

    Returns
    -------
    pandas.Series
        Series with the rows sample in the desired frequency.

    Notes
    -----
    The returned series may have empty rows if the dataset isn't complete.
    It's recommended to use again `convert_numeric` function to interpolate
    the empty values and maintain the data type consistency.

    A column of a DataFrame is consider a series by the pandas library.
    """
    # Choose between a special or mean (average) function
    if series.name in config_file.functions:
        functions = config_file.functions[series.name]
    else:
        functions = "mean"

    series = series.resample(config_file.freq, label="right",
                             closed="right", origin="start"
                             ).agg(functions)

    return series
