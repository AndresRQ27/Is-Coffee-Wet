import pandas as pd
import numpy as np
import datetime


def merge_datetime(dataset, config_file):
    """
    Merges the Date column and the Time column into a single one, used for
    indexing the dataframe with pandas.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to merge the date and time.
    - config_file: ConfigFile object.
        Object with the needed information to merge the dataset

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with a datetime column as its main
        index. Previous columns with date and time
        removed.
    """

    # Initialize the Serie with the first column in the list
    name = config_file.datetime[0]
    datetime_column = pd.Series(dataset[name])

    # Drop the column
    dataset = dataset.drop(name, axis=1)
    config_file.columns.remove(name)

    # Unifies the other columns
    for name in config_file.datetime[1:]:
        datetime_column += " "
        datetime_column += pd.Series(dataset[name])
        # ! Add more columns if the format ever needs it

        # Drops the added column
        dataset = dataset.drop(name, axis=1)
        config_file.columns.remove(name)

    # Converts the Serie into a "datetime64[ns]" column
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
    - dataset: pd.DataFrame.
        Dataset to change the `dtype`.
    - config_file: ConfigFile object.
        Object with the needed information to give format

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with columns type changed and missing
        values interpolated.
    """

    # Sets all "nullValues" to NaN
    for null in config_file.null:
        dataset = dataset.replace(null, np.NaN)

    # Changes the data to float64
    for name in config_file.columns:

        # Check if it's an special format
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


def sample_dataset(dataset, config_file):
    """
    Sample dataset according to a time frequency given. The dataset will be
    group in interval of `frequency` and a numeric function will be applied to
    the rows between these timeframes to get one single value for each new
    timestamp.

    Different numeric functions can be applied to different column by pairing
    the function with the column in a tuple in `column_function`. The return
    value is a sampled dataset into the given frequency.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to sample each column.
    - config_file: ConfigFile object.
        Object with the needed information to resample the dataset

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with the rows sample in the desired frequency.

    Notes
    -----
    The returned database may have empty rows if the dataset isn't complete.
    It's recommended to use again `convert_numeric` function to interpolate the
    empty values and maintain the data type consistency.
    """
    # Creates the new dataset to return
    new_dataset = pd.DataFrame()

    # Generates a column of accumulated time the leaf has been wet, using
    # the Leaf Wet 1 column.
    if "Leaf Wet 1" in config_file.columns:
        # Difference between last and current datetime
        # This is because the frequency isn't all the same
        time_diff = dataset.index[1:] - dataset.index[:-1]

        # Creates a new column with 0s
        dataset["Leaf Wet Accum"] = 0

        # Sets to 1 the columns that has wetness
        dataset.loc[dataset["Leaf Wet 1"]
                    > 0, "Leaf Wet Accum"] = 1

        # There is no data about the delta time of the first
        # measurement, so use the difference of the second data as a
        # hard guess. We divide by 60 to use minutes
        dataset.loc[dataset.index[0],
                    "Leaf Wet Accum"] *= time_diff[0].days * 24 * 60 + time_diff[0].seconds / 60
        dataset.loc[dataset.index[1:],
                    "Leaf Wet Accum"] *= time_diff.days * 24 * 60 + time_diff.seconds / 60

        # Add the new column to the config_file file
        config_file.columns.append("Leaf Wet Accum")
        config_file.functions["Leaf Wet Accum"] = "sum"
        config_file.formats["Leaf Wet Accum"] = "int"

    for name in config_file.columns:

        # Choose between a special or mean function
        if name in config_file.functions:
            functions = config_file.functions[name]
        else:
            functions = "mean"

        # Generates a new DataFrame (DF)
        # Each resampling creates a column, so it is appended to get a
        # final result. It must be done this way to use a different
        # function in each column
        # noinspection SpellCheckingInspection
        auxiliar_df = dataset.resample(config_file.freq, label="right",
                                       closed="right", origin="start"
                                       ).agg({name: functions})

        new_dataset = pd.concat([new_dataset, auxiliar_df], axis=1)

    # If the value is NaN in "Leaf Wet 1"
    # set "Leaf Wet Accum" to NaN as well
    # This is to use interpolation later in these values
    new_dataset.loc[new_dataset["Leaf Wet 1"].isna(),
                    "Leaf Wet Accum"] = np.NaN

    # *** No value can be over the maximum amount of minute in a given
    # *** timestamp according to its frequency, so set a cap
    # *** Change the values to the limit

    # Calculates the maximum limit for a delta timestamp
    limit = pd.Timedelta(config_file.freq)
    limit = limit.days * 24 * 60 + limit.seconds / 60

    # Sets a hard limit only to the Leaf Wet Accum column
    new_dataset["Leaf Wet Accum"].loc[new_dataset["Leaf Wet Accum"] > limit] = limit

    return new_dataset


def cyclical_encoder(dataset, config_file):
    """
    Converts values that have cyclical behavior into pair of sin(x) and
    cos(x) functions. Day and time is extracted from a datetime index, so
    be sure to have it. To understand the purpose of this function, see
    `Notes`.

    The return dataset has a decomposition into sin(x) and cos(x) of the
    desired frequencies.

    Parameters
    ----------
    - dataset: pd.DataFrame. Dataset to extract the time from it's datetime
        index.
    - config_file: ConfigFile object. Object with the needed information to
        create columns that represent cyclical trends

    Returns
    -------
    - dataset : pd.DataFrame. Dataset with the decomposition of sin(x) and
        cos(x) of the day and/or hour.

    Notes
    -----
    The purpose is that the NN see all days equally separated from
    themselves, as the last day of the year is one day away of the first
    day; same happens with the hours and the minutes of a day. By using
    integers, this isn't obvious to the NN, so we help him by parsing the
    time in a more easy-to-see way.

    Moreover, compared to the previous version that covered values between
    0 and 2*pi, the actual function represents the full date in a continuous
    wave. This helps to see de date as cyclical, while having a difference
    between (e.g.) 04-May-2014 and 04-May-2020, without taking into account
    the year directly.

    A frequency analysis of the data using the function "freq_domain" is
    strongly recommended.
    """
    # Generates a unique timestamp in second of each datetime
    timestamp_s = dataset.index.map(datetime.datetime.timestamp)

    # Divide into sin/cos columns
    for name, time in config_file.encode.items():
        dataset[name + " sin"] = np.sin(timestamp_s * (2 * np.pi / time))
        dataset[name + " cos"] = np.cos(timestamp_s * (2 * np.pi / time))

    return dataset
