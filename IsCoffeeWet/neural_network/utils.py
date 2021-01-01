import glob
import os

import numpy as np
import tensorflow as tf

from IsCoffeeWet.neural_network import filternet_module as flm
from IsCoffeeWet.neural_network import temporal_convolutional as tcn


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
    tuple[pandas.DataFrame, pandas.Series]
        Dataset with the normalized data and the max_values absolute value
        for each column
    """
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


def mae(y_true, y_pred):
    """
    Calculates the Mean-Absolute-Error (MAE) for an un-standardize
    dataset.

    Parameters
    ----------
    y_true: pandas.DataFrame
        Ground truth values
    y_pred: pandas.DataFrame
        Predicted values by the model

    Returns
    -------
    pandas.DataFrame
        MAE for each value in the last window of the dataset. Can contain
        NaNs
    """
    return np.abs(y_true - y_pred)


def analyze_loss(y_true, y_pred, index, grouping_func, frequency="1D"):
    # Calculates the MAE of the values
    loss = mae(y_true, y_pred)

    # Sets the datetime index. It assumes these are the last values
    loss.set_index(index[-len(y_true):], inplace=True)

    # Resamples the loss in the desired frequency
    result = loss.resample(frequency, label="right",
                           closed="right", origin="start"
                           ).aggregate(func=grouping_func)
    return result


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


def load_model(path, submodel=None):
    """
    Function that loads a neural network model from a file, given a path to
    it.

    Parameters
    ----------
    path: string
        Path where look for the the neural network model in the filesystem.
    submodel: string, optional
        Acronym of the sub-model used in the architecture of the saved
        model. Valid options are `'tcn'` and `'conv_lstm'`

    Returns
    -------
    tensorflow.keras.Model
        Model of the neural network.

    Notes
    -----

    To save the weight normalization layer used in the TCN, from tensorflow
    addons, `h5` is needed, as Protobuffer can't handle it yet [1]. This
    means, features like portability between different platforms (ie.
    Tensorflow Lite, Tensorflow Serving, etc.) cannot be used.

    References
    ----------
    [1] https://github.com/tensorflow/addons/issues/1788
    """

    # If path is a directory, look for the most recent file h5
    if os.path.isdir(path):
        # Look for the most recent file in the directory
        list_of_files = glob.glob(path + "/*.h5")
        # Assign the path to the most recent h5 if there is one
        # If there isn't one, either the directory is empty or model is a ".pb"
        path = max(list_of_files, key=os.path.getctime) \
            if len(list_of_files) != 0 else path

    if submodel == "tcn":
        model = tf.keras.models.load_model(filepath=path,
                                           custom_objects={
                                               "ResidualBlock": tcn.ResidualBlock})
    elif submodel == "conv_lstm":
        model = tf.keras.models.load_model(filepath=path,
                                           custom_objects={
                                               "FilternetModule": flm.FilternetModule})
    else:
        model = tf.keras.models.load_model(filepath=path)

    return model


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
    # Creates path to save the neural network for the tests
    try:
        os.makedirs(os.path.dirname(path))
        print("Path to save the neural networks was created")
    except FileExistsError:
        print("Path to save the neural networks was found")

    tf.keras.models.save_model(model=model, filepath=path, save_format="h5")
    print("Your model has been save to '{}'".format(path))
