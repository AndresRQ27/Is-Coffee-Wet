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
    mean: pandas.DataFrame
        DataFrame with the mean of each column, obtained from the standardize
        function.
    std: pandas.DataFrame
        DataFrame with the std of each column, obtained from the standardize
        function.

    Returns
    -------
    pandas.DataFrame
        Dataset with the data de-standardized. Can no longer be used to feed
        into the neural network.
    """
    dataset = dataset * std + mean
    return dataset


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
