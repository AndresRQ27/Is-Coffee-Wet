import unittest

import copy
import os
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

g_dataset: pd.DataFrame
g_config: cf.ConfigFile
g_train: pd.DataFrame
g_val: pd.DataFrame
g_test: pd.DataFrame
g_window: wg.WindowGenerator
g_filter_size: int
g_kernel_size: int
g_dilations: int
g_input_size: tuple
g_output_size: tuple

all_history: pd.DataFrame

# TODO: use a seed for weight initialization
# ************************************


def setUpModule():
    """
    Set up module that executes before any test classes.

    Initialize all the global values used by the latter tests.
    """

    global g_dataset, g_config, g_train, g_val, g_test, all_history, g_output_size
    global g_filter_size, g_kernel_size, g_dilations, g_window, g_input_size

    # *** Dataset
    # Loads the dataset
    g_dataset = pd.read_csv(PATH + "/database/dataset_hour.csv",
                            engine="c", index_col="Datetime", parse_dates=True)

    # Information of the dataset
    print(g_dataset.info(verbose=True))
    print(g_dataset.describe().transpose())

    # *** Config File
    # Use in the dataset partition
    g_config = cf.ConfigFile()

    # Use the entire data with the benchmarks, as the models won't be saved
    g_config.training = 1
    g_config.validation = 0.1
    g_config.num_data, g_config.num_features = g_dataset.shape

    # *** Dataset preparation
    # Normalize the dataset
    g_dataset, _, _ = nn.standardize(g_dataset)

    # Partition the dataset
    _, g_train, g_val, g_test = nn.split_dataset(g_dataset, g_config)

    # *** Window
    # A week in hours
    input_width = 7 * 24
    label_width = input_width
    label_columns = g_dataset.columns.tolist()

    # Removes th sin/cos columns from the labels
    label_columns = label_columns[:-4]

    # Window of 7 days for testing the NN
    g_window = wg.WindowGenerator(input_width=input_width,
                                  label_width=label_width,
                                  shift=label_width,
                                  train_ds=g_train,
                                  val_ds=g_val,
                                  test_ds=g_test,
                                  label_columns=label_columns)

    # Arguments of the default NN
    g_dilations = 5
    g_filter_size = [192, 64, 128, 96, 128]
    g_kernel_size = [6, 12, 10, 6, 10]

    g_input_size = (168, 19)
    g_output_size = (168, 15)

    # *** Dataframe
    # Dataframe use to store the history of each training, then save it
    all_history = pd.DataFrame()


def tearDownModule():
    """
    Tear down module that executes after all the test classes are done executing.

    Saves the pandas DataFrame that contains the history of all the test into
    a csv for later evaluation.
    """
    global all_history

    # Save to csv:
    csv_file = PATH + "/results/benchmark_temporal.csv"
    with open(csv_file, mode='w') as file:
        all_history.to_csv(file)


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
    # TODO: add tensorboard to the callback
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
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_generic_network(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_window, g_dilations
        global g_output_size, g_input_size

        # Name used to identify its data in the history
        self.name = "generic"

        # Compiles the model with the default values
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


class Test_TestNoEncoding(unittest.TestCase):
    """
    Test class with variations of the dataset. Removes encoding columns
    to evaluate the convergence of the model with/without it.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_no_encoding(self):
        """
        Function that removes the encoded columns from the partitioned
        database stored in the window
        """
        global g_filter_size, g_kernel_size, g_window, g_dilations
        global g_output_size, g_test, g_val, g_train

        # Name used to identify its data in the history
        self.name = "no_encoding"

        # Copy the original window to modify only the sets
        window_ne = copy.deepcopy(g_window)

        # Drops the encoding for each set present in the window
        window_ne.test_ds = g_test.drop(["day sin",
                                         "day cos",
                                         "year sin",
                                         "year cos"], axis=1)

        window_ne.train_ds = g_train.drop(["day sin",
                                           "day cos",
                                           "year sin",
                                           "year cos"], axis=1)

        window_ne.val_ds = g_val.drop(["day sin",
                                       "day cos",
                                       "year sin",
                                       "year cos"], axis=1)

        # Re-define the input/output size for the model
        input_width = 24 * 7  # A week in hours
        label_columns = window_ne.train_ds.shape[1]  # Number of features as an input

        # Re-defines only the input size as it's the one that changed
        input_size = (input_width, label_columns)

        # Generates a model with a custom input size as the encoding columns are not present
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   input_size,
                                   g_output_size)

        # Compiles and train the model using a window with custom training and validation sets
        self.history = compile_and_fit(model, window_ne)


class Test_TestDay(unittest.TestCase):
    """
    Test class with variations of the frequency used in the dataset. Measures
    if it's better to use hours or days as the main frequency.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_day(self):
        """
        Function that trains the model with a custom dataset resampled by days.
        """
        global g_config

        # Name used to identify its data in the history
        self.name = "day"

        # *** Dataset. All parsing process for new dataset
        # Loads the dataset
        dataset_day = pd.read_csv(PATH + "/database/dataset_day.csv",
                                  engine="c", index_col="Datetime", parse_dates=True)

        # *** Dataset preparation
        # Normalize the dataset
        dataset_day, _, _ = nn.standardize(dataset_day)

        # Copy config file
        config_day = copy.deepcopy(g_config)
        config_day.num_data = dataset_day.shape[0]  # Number of data available

        # Partition the dataset
        _, train_day, val_day, test_day = nn.split_dataset(dataset_day, config_day)

        # *** Window
        input_width = 7  # No longer hours, so only 7 days
        label_width = input_width  # Label same size as the input
        label_columns = dataset_day.columns.tolist()  # Number of features

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 7 days using a dataset resampled by days
        window_day = wg.WindowGenerator(input_width=input_width,
                                        label_width=label_width,
                                        shift=label_width,
                                        train_ds=train_day,
                                        val_ds=val_day,
                                        test_ds=test_day,
                                        label_columns=label_columns)

        input_size = (input_width, g_dataset.shape[1])  # Model's input shape
        output_size = (label_width, len(label_columns))  # Model's output shape

        # Generates a model with custom kernel size as I/O are different
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   input_size,
                                   output_size)

        # Compiles and fit the model using a window that sees days
        self.history = compile_and_fit(model, window_day)


class Test_TestBatchSize(unittest.TestCase):
    """
    Test class with variations to the batch size of the data during training.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_batch_128(self):
        global g_input_size, g_output_size, g_window
        global g_filter_size, g_kernel_size, g_dilations

        # Name used to identify its data in the history
        self.name = "batch_128"

        # *** Window
        window_128 = copy.deepcopy(g_window)
        window_128.batch_size = 128

        # Generates a generic model
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   g_output_size)

        # Compiles and fits using a generic window with batch size of 128
        self.history = compile_and_fit(model, window_128)

    def test_batch_512(self):
        global g_window, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_dilations

        # Name used to identify its data in the history
        self.name = "batch_512"

        # *** Window
        window_512 = copy.deepcopy(g_window)
        window_512.batch_size = 512

        # Generates a generic model
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   g_output_size)

        # Compiles and fits using a generic window with batch size of 512
        self.history = compile_and_fit(model, window_512)

    def test_batch_256(self):
        global g_window, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_dilations

        # Name used to identify its data in the history
        self.name = "batch_256"

        # *** Window
        window_256 = copy.deepcopy(g_window)
        window_256.batch_size = 256

        # Generates a generic model
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   g_output_size)

        # Compiles and fits using a generic window with batch size of 256
        self.history = compile_and_fit(model, window_256)


class Test_TestActivation(unittest.TestCase):
    """
    Test class with variations to the batch size of the data during training.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_gated_activation(self):
        global g_input_size, g_output_size, g_window
        global g_filter_size, g_kernel_size, g_dilations

        # Name used to identify its data in the history
        self.name = "gated_activation"

        # Generates a generic model
        model = mg.temp_conv_model(g_filter_size,
                                   g_kernel_size,
                                   g_dilations,
                                   g_input_size,
                                   g_output_size,
                                   activation=activation.gated_activation)

        # Compiles and fits using a generic window with batch size of 128
        self.history = compile_and_fit(model, g_window)


class Test_TestWindow(unittest.TestCase):
    """
    Test class with variations of the time window that the NN sees to do a
    prediction. This changes the window and the I/O shapes of the NN.
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_window_14(self):
        """
        Function that uses a window size of 14 days in the past to predict
        the next 14 days.
        """
        global g_window, g_filter_size, g_kernel_size, g_dilations

        # Name used to identify its data in the history
        self.name = "window_14"

        # *** Window
        window_14 = copy.deepcopy(g_window)
        input_width = 14 * 24
        label_width = input_width

        # Change the window values
        window_14.input_width = input_width
        window_14.label_width = label_width
        window_14.shift = label_width

        # Arguments for the model. Different input shape requires different NN configuration
        input_size = (input_width, window_14.train_ds.shape[1])  # New input shape
        output_size = (label_width, len(window_14.label_columns))  # New output shape

        # Generates a new model with custom I/O, kernel and pool size
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_dilations,
                                       input_size,
                                       output_size)

        # Compiles and fits using a window of 14 days to predict 14 days
        self.history = compile_and_fit(model, window_14)

    def test_window_21(self):
        """
        Function that uses a window size of 14 days in the past to predict
        the next 14 days.
        """
        global g_window, g_filter_size, g_kernel_size, g_dilations

        # Name used to identify its data in the history
        self.name = "window_21"

        # *** Window
        window_21 = copy.deepcopy(g_window)
        input_width = 21 * 24
        label_width = input_width

        # Change the window values
        window_21.input_width = input_width
        window_21.label_width = label_width
        window_21.shift = label_width

        # Arguments for the model. Different input shape requires different NN configuration
        input_size = (input_width, window_21.train_ds.shape[1])  # New input shape
        output_size = (label_width, len(window_21.label_columns))  # New output shape

        # Generates a new model with custom I/O, kernel and pool size
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_dilations,
                                       input_size,
                                       output_size)

        # Compiles and fits using a window of 21 days to predict 21 days
        self.history = compile_and_fit(model, window_21)


class Test_TestModels(unittest.TestCase):
    """
    Test class with variations of the architectural configurations of the model.
    This includes changes in the kernel size, filters and dilation,
    """

    def setUp(self):
        """
        Set up function executed before each individual test
        """
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        """
        Tear down function executed after each individual test
        """
        global all_history

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_model_2(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_output_size, g_input_size, g_window

        # Name used to identify its data in the history
        self.name = "model_2"

        filter_size = [160, 160, 96]
        kernel_size = [10, 2, 2]
        dilations = 3

        # Compiles the model with the default values
        model = mg.temp_conv_model(filter_size,
                                   kernel_size,
                                   dilations,
                                   g_input_size,
                                   g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_model_3(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_output_size, g_input_size, g_window

        # Name used to identify its data in the history
        self.name = "model_3"

        filter_size = [256, 96, 96, 192, 128]
        kernel_size = [6, 10, 2, 6, 12]
        dilations = 5

        # Compiles the model with the default values
        model = mg.temp_conv_model(filter_size,
                                   kernel_size,
                                   dilations,
                                   g_input_size,
                                   g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_model_4(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_output_size, g_input_size, g_window

        # Name used to identify its data in the history
        self.name = "model_4"

        filter_size = [192, 64, 128, 96, 128, 64]
        kernel_size = [6, 12, 10, 6, 10, 6]
        dilations = 6

        # Compiles the model with the default values
        model = mg.temp_conv_model(filter_size,
                                   kernel_size,
                                   dilations,
                                   g_input_size,
                                   g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


if __name__ == '__main__':
    unittest.main()
