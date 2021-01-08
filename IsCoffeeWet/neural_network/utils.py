import glob
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet.neural_network.models import filternet_module as flm
from IsCoffeeWet.neural_network.models import temporal_convolutional as tcn


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
    print(">>> Spliting dataset into train, val and test...")

    # Resets index to add datetime as a normal column
    dataset = dataset.reset_index()

    # Pops the datetime from the dataset. Not use in the NN explicitly
    datetime_index = dataset.pop("index")

    # Gets the last year of data according to the frequency of the dataset
    # Transforms the frequency from string to Timedelta
    delta_freq = pd.Timedelta(config_file.freq)

    # Amount of steps from the dataset in a day
    delta_freq = delta_freq.days + delta_freq.seconds / 86400

    # Number of data that will be in the training set
    train_ratio = len(dataset) - int(np.floor(364 / delta_freq))

    # Removes the last 365 days of data from the training set
    train_ds = dataset[:train_ratio]
    # Leave 14 days of validation set from the val_test group
    # First 7 are the predictions, second 7 are the labels
    val_ds = dataset[train_ratio:train_ratio+config_file.forecast*2]
    # Saves the remaining data as the test set
    test_ds = dataset[train_ratio+config_file.forecast*2:]

    return datetime_index, train_ds, val_ds, test_ds


def load_model(path, name="", submodel=None):
    """
    Function that loads a neural network model from a file, given a path to
    it.

    Parameters
    ----------
    path: string
        Path where look for the the neural network model in the filesystem.
        If a path with a filename is given, then the name is ignored
    name: string, optional
        Name of the neural network. It can be given directly in the path
        for loading a specific model. Otherwise, it will look for the most
        recent model with the name given.
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
    print(">>> Loading model...")

    # If path is a directory, look for the most recent file h5
    if os.path.isdir(path):
        # Look for the most recent file in the directory
        list_of_files = glob.glob(path + "/{}*.h5".format(name))
        # Assign the path to the most recent h5 if there is one
        # If there isn't one, either the directory is empty or model is a ".pb"
        path = max(list_of_files, key=os.path.getctime) \
            if len(list_of_files) != 0 else None

    if path is not None:
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
    else:
        model = None

    return model


def save_model(model, path, name="saved_model"):
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
    print(">>> Saving model...")

    path = os.path.join(path, name+".h5")
    tf.keras.models.save_model(
        model=model, filepath=path, save_format="h5")
    print("Your model has been save to '{}'".format(path))


def compile_and_fit(model, window, nn_path, model_name, patience=4,
                    learning_rate=0.0001, max_epochs=100):
    """
    Function that compiles and train the model. It's a generic function as
    multiple modules are compiled and trained.

    Parameters
    ----------
    model: tensorflow.keras.Model
        Neural network model that will be compiled and trained
    window: window_generator.WindowGenerator
        Window that contains the train set and validation set used in
        the fitting.
    patience: int, optional
        Minimum number of epochs that must pass without significant change
        before it stops early.
    learning_rate: float, optional
        Number passed to the optimizer. Used when updating the weights
        of the network.
    max_epochs: int, optional
        Max number of epochs to train the neural network

    Returns
    -------

    tf.keras.callbacks.History
        Objects that contains the history of the model training.
    """
    print(">>> Compiling and training model...")

    # Sets an early stopping callback to prevent over-fitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=0,
                                                      patience=patience,
                                                      mode="auto",
                                                      restore_best_weights=True)

    checkpoint_path = os.path.join(nn_path, "best_{}.h5".format(model_name))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='auto',
                                                    period=1)

    # Compiles the model with the loss function, optimizer to use and metric to watch
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError(),
                           tf.metrics.MeanAbsolutePercentageError()])

    # Trains the model
    history = model.fit(window.train,
                        validation_data=window.val,
                        epochs=max_epochs,
                        callbacks=[early_stopping,
                                   checkpoint])

    # Returns a history of the metrics
    return history


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
