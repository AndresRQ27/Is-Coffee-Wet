import datetime

import numpy as np
import pandas as pd

from IsCoffeeWet.preprocess.data_parser import sample_series


def create_leaf_wet_accum(dataset, config_file):
    """
    Creates the leaf wetness accumulated metric, based on the reports of
    the leaf wetness sensor. Measured in minutes.

    Parameters
    ----------
    dataset : ´pandas.DataFrame´
        Dataset with the information of leaf wetness.
    config_file: config_file.ConfigFile
        Object with the needed information to create the new column.

    Returns
    -------
    dataset : ´pandas.DataFrame´
        Original dataset with leaf wetness accumulated appended
    """
    print(">>> Generating leaf wetness accumulated...")

    # Difference between last and current datetime
    # This is because the frequency isn't uniform across the dataset
    time_diff = dataset.index[1:] - dataset.index[:-1]

    # Creates a new series with 0s, the same size as the dataset
    series = pd.Series(np.zeros(len(dataset)),
                       index=dataset.index,
                       dtype=np.int64,
                       name='Leaf Wet Accum')

    # Create a mask to find the rows to change the series values
    mask = dataset["Leaf Wet 1"] > 0

    # Sets 1 in the rows where wetness is read by the sensor
    series[mask] = 1

    # There is no data about the delta time of the first
    # measurement, so use the difference of the second data as a
    # hard guess. We divide by 60 to use minutes
    series.iloc[0] *= time_diff[0].days * \
        24 * 60 + time_diff[0].seconds / 60
        
    series.iloc[1:] *= time_diff.days * \
        24 * 60 + time_diff.seconds / 60

    # Add the new column to the config_file file
    config_file.columns.append("Leaf Wet Accum")
    config_file.functions["Leaf Wet Accum"] = "sum"
    config_file.formats["Leaf Wet Accum"] = "int"
    config_file.labels.append("Leaf Wet Accum")

    # Sample the Leaf Wet Accum
    series = sample_series(series=series, config_file=config_file)

    # *** No value can be over the maximum amount of minute in a given
    # *** timestamp according to its frequency, so set a cap
    # *** Change the values to the limit

    # Calculates the maximum limit for a delta timestamp
    limit = pd.Timedelta(config_file.freq)
    limit = limit.days * 24 * 60 + limit.seconds / 60

    # Sets a hard limit only to the Leaf Wet Accum column
    series.loc[series > limit] = limit

    return series


def create_cyclical_encoder(dataset_index, config_file):
    """
    Converts values that have cyclical behavior into pair of sin(x) and
    cos(x) functions. Day and time is extracted from a datetime index, so
    be sure to have it. To understand the purpose of this function, see
    `Notes`.

    The return dataset has a decomposition into sin(x) and cos(x) of the
    desired frequencies.

    Parameters
    ----------
    dataset_index: pandas.DataFrame
        Index of the main dataset. It's expected to be a datetime index.
    config_file: config_file.ConfigFile
        Object with the needed information to create columns that represent
        cyclical trends.

    Returns
    -------
    pandas.DataFrame
        Dataset with the decomposition of sin(x) and cos(x) of the day
        and/or hour.

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
    print(">>> Generating cyclical encoder...")
    
    # Generates a unique timestamp in second of each datetime
    timestamp_s = dataset_index.map(datetime.datetime.timestamp)

    new_dataset = pd.DataFrame(index=dataset_index)

    # Divide into sin/cos columns
    for name, time in config_file.encode.items():
        new_dataset[name +
                    " sin"] = np.sin(timestamp_s * (2 * np.pi / time))
        new_dataset[name +
                    " cos"] = np.cos(timestamp_s * (2 * np.pi / time))

        # Add columns to the config file
        config_file.columns.append(name + " sin")
        config_file.columns.append(name + " cos")

    return new_dataset
