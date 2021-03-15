def normalize(dataset):
    """
    Function that normalizes a dataset using the max_values absolute value for
    each column. A 10% size increase of the max_values value is added to prevent
    for unknown bigger values.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to normalize.

    Returns
    -------
    tuple [pandas.DataFrame, pandas.Series]
        Dataset with the normalized data and the max_values absolute value
        for each column
    """
    print(">>> Normalizing dataset...")
    # Calculates the absolute max_values value and increase it by 10%
    # This give a room for new unknown values so the max_values value won't be 1
    max_values = dataset.abs().max() * 1.10

    # Reduce values to [0, 1] is all positives
    # And [-1, 1] if there are negative values
    dataset = dataset / max_values

    return dataset, max_values


def de_normalize(dataset, max_values):
    """

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to de-normalize.
    max_values: pandas.Series or float
        Value to upscale the dataset. If the dataset are multiple columns,
        the max must be a Series with the same name for those columns to
        apply the operation

    Returns
    -------
    pandas.DataFrame
        Dataset with the data de-normalize. Can no longer be used to feed
        into the neural network.
    """
    print(">>> De-normalizing dataset...")

    return dataset * max_values


def standardize(dataset):
    """
    Function that standardizes the values in all the columns present in the
    dataset. Use the z-score standardization.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to standardize.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, pandas.Series]
        Dataset with the standardize data, plus the mean and standard
        deviation of each column to de-normalize later.

    Notes
    -----
    The formula used in the standardization (z-score) is:

    .. math:: z_{i} = x_{i} - \overline{x}_{s}

    Where:

    - :math:`x_{i}` is a data point :math:`(x_{1}, x_{2}, ..., x_{n})`
    - :math:`\overline{x}` is the sample mean.
    - :math:`s` is the sample standard deviation.
    """
    print(">>> Standardizing dataset...")

    # Computes the mean and standard deviation
    ds_mean = dataset.mean()
    ds_std = dataset.std()

    # Apply the method to the dataset
    dataset = (dataset - ds_mean) / ds_std

    return dataset, ds_mean, ds_std


def de_standardize(dataset, mean, std):
    """
    Inverse function of the standarization. Restores the dataset to the
    normal values to obtain the real prediction used in the final output.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to de-standardize.
    mean: pandas.Series or float
        DataFrame with the mean of each column, obtained from the standardize
        function.
    std: pandas.Series or float
        DataFrame with the std of each column, obtained from the standardize
        function.

    Returns
    -------
    pandas.DataFrame
        Dataset with the data de-standardized. Can no longer be used to feed
        into the neural network.
    """
    print(">>> De-standardizing dataset...")

    return dataset * std + mean