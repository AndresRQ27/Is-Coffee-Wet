import pandas as pd


def normalize(dataset):
    """
    Function that normalizes a dataset using the max absolute value for
    each column. A 10% size increase of the max value is added to prevent
    for unknown bigger values.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to normalize.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Dataset with the normalized data and the max absolute value
        for each column
    """
    # Calculates the absolute max value and increase it by 10%
    # This give a room for new unknown values so the max value won't be 1
    max_values = dataset.abs().max() * 1.10

    # Reduce values to [0, 1] is all positives
    # And [-1, 1] if there are negative values
    dataset = dataset / max_values

    return dataset, max_values


def de_normalize(dataset, max):
    """

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the data to de-normalize.
    max: pandas.Series or float
        Value to upscale the dataset. If the dataset are multiple columns,
        the max must be a Series with the same name for those columns to
        apply the operation

    Returns
    -------
    pandas.DataFrame
        Dataset with the data de-normalize. Can no longer be used to feed
        into the neural network.
    """

    return dataset * max


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
    return dataset * std + mean


def mape(dataset, config, model):
    """
    Calculates the Mean-Absolute-Percentage-Error (MAPE) for an un-standardize
    dataset. It is done only for the last window of the dataset, with the
    window size given by `config.forecast`.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Weather dataset without standardization/normalization
    config: config_file.ConfigFile
        Config file of the program to read the forecast
    model: tensorflow.keras.Model
        Trained NN to generate the predictions

    Returns
    -------
    pandas.DataFrame
        MAPE for each value in the last window of the dataset. Can contain
        NaNs
    """
    last_data = dataset.iloc[-config.forecast:]
    last_label = dataset[config.labels].iloc[-config.forecast:]
    prediction = model(last_data.to_numpy(), training=False)

    # Transform predictions to a DataFrame
    prediction = pd.DataFrame(prediction.numpy(),
                              columns=config.labels,
                              index=last_label.index)

    # TODO: fix infinites
    # Calculates the MAPE.
    prediction = 100 * (last_label - prediction).abs() / last_label

    return prediction


def split_dataset(dataset, config_file):
    """
    Function that split the dataset into groups in a proportion determined
    previously in the configuration file.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset to divide into training, validation and test set.
    config_file: config_file.ConfigFile
        Configuration file with the total number of data available and the
        distribution to split it.

    Returns
    -------
    tuple[pandas.Series, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Returns the datetime index in a `pandas.Series`; the training set,
        validation set and test set in a `pandas.DataFrame`.
    """

    # Resets index to add datetime as a normal column
    dataset = dataset.reset_index()

    # Pops the datetime from the dataset. Not use in the NN explicitly
    datetime_index = dataset.pop("Datetime")

    # Accumulates the ratio to use in slices.
    # Validation set is taken from the training set.
    train_ratio = config_file.training - config_file.validation  # e.g. from 0 to 0.5
    validation_ratio = config_file.training  # e.g. from 0.5 to 0.7

    # Divides the dataset
    train_ds = dataset[0:int(config_file.num_data * train_ratio)]
    val_ds = dataset[int(config_file.num_data * train_ratio):
                     int(config_file.num_data * validation_ratio)]
    test_ds = dataset[int(config_file.num_data * validation_ratio):]

    return datetime_index, train_ds, val_ds, test_ds


def load_model(path):
    """
    Function that loads a neural network model from a file, given a path to
    it.

    Parameters
    ----------
    path: string
        Path where look for the the neural network model in the filesystem.

    Returns
    -------
    tensorflow.keras.Model
        Model of the neural network.
    """
    # TODO: implement function
    # TODO: test
    return


def save_model(model, path):
    """
    Function that saves a keras model to a given path in the filesystem.
    The extension of the file is a `.tf` that keras uses to save all the
    needed information. By default, it overwrites any existing file at the
    target location.

    Parameters
    ----------
    model: tensorflow.keras.Model
        Model to save. It saves the I/O shape, the architecture of the
        network, weights and state of the optimizer
    path: string
        Path in the filesystem to save the model.

    """
    # TODO: tests

    model.save(path)
    print("Your model has been save to '{}'".format(path))


"""
# Use tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=log_dir,
                            histogram_freq=1,
                            embeddings_freq=0,
                            update_freq="epoch"
                        )
"""
