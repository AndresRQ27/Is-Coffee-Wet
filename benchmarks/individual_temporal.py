import copy
import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet import activation
from IsCoffeeWet import config_file as cf
from IsCoffeeWet import model_generator as mg
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import window_generator as wg

PATH = os.getcwd() + "/resources/benchmark"

# ************************************
# ********* Global Variables *********
# ************************************

g_window: wg.WindowGenerator
g_filter_size: int
g_kernel_size: int
g_dilations: int
g_input_size: tuple
prediction_data: np.ndarray
prediction_label: np.ndarray
g_mean: pd.Series
g_std: pd.Series

all_history: pd.DataFrame
prediction_result: pd.DataFrame


# ************************************

def setUpModule():
    """
    Set up module that executes before any test classes.

    Initialize all the global values used by the latter tests.
    """

    global g_window, all_history, g_input_size, prediction_data, g_mean, g_std
    global g_filter_size, g_kernel_size, g_dilations, prediction_result, prediction_label

    # *** Dataset
    # Loads the dataset
    dataset = pd.read_csv(PATH + "/database/dataset_hour.csv",
                          engine="c", index_col="Datetime", parse_dates=True)

    # Information of the dataset
    print(dataset.info(verbose=True))
    print(dataset.describe().transpose())

    # *** Config File
    # Use in the dataset partition
    config = cf.ConfigFile()

    # Use the entire data with the benchmarks, as the models won't be saved
    config.training = 1
    config.validation = 0.1
    config.num_data = dataset.shape[0]

    # *** Dataset preparation
    # standardize the dataset
    dataset, g_mean, g_std = nn.standardize(dataset)

    # Partition the dataset
    _, train_ds, val_ds, _ = nn.split_dataset(dataset, config)

    # *** Window
    # A week in hours
    input_width = 7 * 24
    label_columns = dataset.columns.tolist()

    # Removes th sin/cos columns from the labels
    label_columns = label_columns[:-4]

    # Drop the day and year columns as they aren't needed when de-standardize
    g_mean = g_mean[:-4]
    g_std = g_std[:-4]

    # Window of 7 days for testing the NN. Tested with batch size of 512
    g_window = wg.WindowGenerator(input_width=input_width,
                                  label_width=input_width,
                                  shift=input_width,
                                  train_ds=train_ds,
                                  val_ds=val_ds,
                                  test_ds=_,
                                  label_columns=label_columns,
                                  batch_size=512)

    # Saves the last window from the standardize dataset
    # Used to evaluate results from group vs individual
    prediction_data = dataset[-336:-168].to_numpy().reshape((1, 168, 19))
    prediction_label = dataset[-168:].to_numpy().reshape((1, 168, 19))

    # Arguments of the default NN. Tested with the model 2
    g_filter_size = [160, 160, 96]
    g_kernel_size = [10, 2, 2]
    g_dilations = 3
    # Input size of the model
    g_input_size = (input_width, dataset.shape[1])

    # *** Dataframe
    prediction_result = pd.DataFrame()

    # Dataframe use to store the history of each training, then save it
    try:
        # Overwrites past results.
        all_history = pd.read_csv(PATH + "/results/benchmark_prediction_temporal.csv",
                                  engine="c", index_col=0)
    except FileNotFoundError:
        # Creates new results
        all_history = pd.DataFrame()


def tearDownModule():
    """
    Tear down module that executes after all the test classes are done executing.

    Saves the pandas DataFrame that contains the history of all the test into
    a csv for later evaluation.
    """
    global all_history, prediction_result

    # Save to csv:
    history_csv = PATH + "/results/benchmark_prediction_temporal.csv"
    predict_csv = PATH + "/results/prediction/prediction_temporal.csv"

    with open(history_csv, mode='w') as file:
        all_history.to_csv(file)

    with open(predict_csv, mode='w') as file:
        prediction_result.to_csv(file)


def compile_and_fit(model, window, patience=4, learning_rate=0.0001,
                    max_epochs=100):
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
    # Sets an early stopping callback to prevent over-fitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      min_delta=0,
                                                      patience=patience,
                                                      mode="min")

    # Compiles the model with the loss function, optimizer to use and metric to watch
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError(),
                           tf.metrics.MeanAbsolutePercentageError()])

    # Trains the model
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    # Returns a history of the metrics
    return history


class Test_TestBase(unittest.TestCase):
    """
    Test class with the base scenario of the convolutional neural network.
    All other networks will be compared against it.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.predict: pd.DataFrame
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history, prediction_result

        # Dataframe is not empty
        if all_history.shape != (0, 0):
            # Resets all index so it doesn't eliminate wrong data
            all_history.reset_index(inplace=True, drop=True)
            # Locates the rows with the same name
            # Gets the index of the rows
            # Drop those rows
            all_history = all_history.drop(
                (all_history.loc[all_history["name"] == self.name]).index)

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

        # Saves the test prediction into a global prediction table
        prediction_result = prediction_result.append(self.predict)

    def test_generic_network(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations, g_mean, g_std
        global g_window, g_input_size, prediction_data, prediction_label

        # Name used to identify its data in the history
        self.name = "generic"

        output_size = (g_input_size[0], len(g_window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 15)),
                               columns=g_window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict, g_mean, g_std)

        # Graphs all the labels in the model
        for label in g_window.label_columns:
            g_window.plot(label,
                          PATH + "/results/prediction/group_temporal/",
                          (prediction_data,
                           prediction_label),
                          model)


class Test_TestLabels(unittest.TestCase):
    """
    Test class with a test per label. Each model predicts only one label at
    a time
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.predict: pd.DataFrame
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history, prediction_result

        # Dataframe is not empty
        if all_history.shape != (0, 0):
            # Resets all index so it doesn't eliminate wrong data
            all_history.reset_index(inplace=True, drop=True)
            # Locates the rows with the same name
            # Gets the index of the rows
            # Drop those rows
            all_history = all_history.drop(
                (all_history.loc[all_history["name"] == self.name]).index)

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

        # Saves the test prediction into a global prediction table
        prediction_result = prediction_result.append(self.predict)

    def test_temp_out(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Temp Out"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_high_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Hi Temp"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_low_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Low Temp"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_out_hum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Out Hum"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_wind_speed(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Wind Speed"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_hi_speed(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Hi Speed"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_bar(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Bar  "

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_rain(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Rain"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_solar_rad(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Solar Rad."

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_hi_solar_rad(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Hi Solar Rad. "

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_in_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "In Temp"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_in_hum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "In Hum"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_soil_moist(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Soils 1 Moist."

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_leaf_wet(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Leaf Wet 1"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)

    def test_leaf_wet_accum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_dilations
        global g_window, g_input_size, g_mean, g_std

        # Name used to identify its data in the history
        self.name = "Leaf Wet Accum"

        # Deep copies the entire window since most info is kept
        window = copy.deepcopy(g_window)

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        window.label_columns = [self.name]
        window.label_columns_indices = {name: i for i, name in
                                        enumerate(window.label_columns)}

        output_size = (g_input_size[0], len(window.label_columns))

        # Compiles the model with the default values. Uses gated activation
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   output_size,
                                   activation.gated_activation)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

        # Gets the column number from the label dictionary
        label_col_index = g_window.label_columns_indices[self.name]

        # Gets the prediction and saves it into a DataFrame
        predict = pd.DataFrame(model(prediction_data).numpy().reshape((168, 1)),
                               columns=window.label_columns)

        # De-standardize the prediction
        self.predict = nn.de_standardize(predict,
                                         g_mean[self.name],
                                         g_std[self.name])

        # Graphs all the labels in the model
        window.plot(self.name,
                    PATH + "/results/prediction/individual_temporal/",
                    (prediction_data,
                     prediction_label[:, :, label_col_index].reshape(1, 168, 1)),
                    model)


if __name__ == '__main__':
    unittest.main()
