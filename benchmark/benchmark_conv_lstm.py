import copy
import os
import unittest

import pandas as pd
import tensorflow as tf

from IsCoffeeWet.preprocess import config_file as cf
from IsCoffeeWet.neural_network import model_generator as mg
from IsCoffeeWet.neural_network import utils
from IsCoffeeWet.neural_network import window_generator as wg

PATH = os.getcwd() + "/resources/benchmark"

# ************************************
# ********* Global Variables *********
# ************************************

g_dataset: pd.DataFrame
g_config: cf.ConfigFile
g_train: pd.DataFrame
g_val: pd.DataFrame
g_window: wg.WindowGenerator
g_filter_size: int
g_kernel_size: int
g_pool_size: int
g_input_size: tuple
g_output_size: tuple
g_label_columns: list

all_history: pd.DataFrame


# ************************************

def setUpModule():
    """
    Set up module that executes before any test classes.

    Initialize all the global values used by the latter tests.
    """

    global g_dataset, g_config, g_train, g_val, g_window, all_history, g_label_columns
    global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size

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
    g_config.num_data = g_dataset.shape[0]

    # *** Dataset preparation
    # Normalize the dataset
    g_dataset, _, _ = utils.standardize(g_dataset)

    # Partition the dataset
    _, g_train, g_val, _ = utils.split_dataset(g_dataset, g_config)

    # *** Window
    # A week in hours
    input_width = 7 * 24
    label_width = input_width
    g_label_columns = g_dataset.columns.tolist()

    # Removes th sin/cos columns from the labels
    g_label_columns = g_label_columns[:-4]

    # Window of 7 days for testing the NN
    g_window = wg.WindowGenerator(input_width=input_width,
                                  label_width=label_width,
                                  shift=label_width,
                                  train_ds=g_train,
                                  val_ds=g_val,
                                  test_ds=_,
                                  label_columns=g_label_columns,
                                  batch_size=64)

    # Arguments of the default NN
    g_filter_size = [64, 64, 64, 64, 224]  # Neurons in a conv layer
    g_kernel_size = [8, 4, 12, 6]  # The kernel will see a day of data
    g_pool_size = 2  # Pooling of the data to reduce the dimensions
    # Input size of the model
    g_input_size = (input_width, g_dataset.shape[1])
    g_output_size = (label_width, len(g_label_columns)
                     )  # Output size of the model

    # *** Dataframe
    # Dataframe use to store the history of each training, then save it
    try:
        # Overwrites past results. Resets the index for no deletion of incorrect data
        all_history = pd.read_csv(PATH + "/performance/benchmark_conv_lstm.csv",
                                  engine="c", index_col=0)
        all_history = all_history.reset_index()
        all_history.pop("index")
    except FileNotFoundError:
        # Creates new results
        all_history = pd.DataFrame()


def tearDownModule():
    """
    Tear down module that executes after all the test classes are done executing.

    Saves the pandas DataFrame that contains the history of all the test into
    a csv for later evaluation.
    """
    global all_history

    # Save to csv:
    history_csv = PATH + "/performance/benchmark_conv_lstm.csv"
    with open(history_csv, mode='w') as file:
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

    def test_generic_network(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "generic"

        # Compiles the model with the default values
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  g_pool_size,
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

    def test_no_encoding(self):
        """
        Function that removes the encoded columns from the partitioned
        database stored in the window
        """
        global g_filter_size, g_kernel_size, g_pool_size, g_val, g_train
        global g_output_size, g_label_columns, g_val, g_train

        # Name used to identify its data in the history
        self.name = "no_encoding"

        # Drops the encoding for each set present in the window
        train_ds = g_train.drop(["day sin",
                                 "day cos",
                                 "year sin",
                                 "year cos"], axis=1)

        val_ds = g_val.drop(["day sin",
                             "day cos",
                             "year sin",
                             "year cos"], axis=1)

        # *** Window
        input_width = 7 * 24

        # Change the window values
        window_ne = wg.WindowGenerator(input_width=input_width,
                                       label_width=input_width,
                                       shift=input_width,
                                       train_ds=train_ds,
                                       val_ds=val_ds,
                                       label_columns=g_label_columns)

        # Re-defines only the input size as it's the one that changed
        input_size = (input_width, len(g_label_columns))

        # Generates a model with a custom input size as the encoding columns are not present
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  g_pool_size,
                                  input_size,
                                  g_output_size)

        # Compiles and train the model using a window with custom training and validation sets
        self.history = compile_and_fit(model, window_ne)


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

    def test_window_14x7(self):
        """
        Function that uses a window size of 14 days in the past to predict
        the next 7 days.
        """
        global g_window, g_filter_size, g_kernel_size

        # Name used to identify its data in the history
        self.name = "window_14x7"

        # *** Window
        input_width = 14 * 24  # Reads 14 days in the past in hours
        label_width = 7 * 24  # Predicts 7 days in the future in hours

        # Change the window values
        window_14x7 = wg.WindowGenerator(input_width=input_width,
                                         label_width=label_width,
                                         shift=label_width,
                                         train_ds=g_train,
                                         val_ds=g_val,
                                         label_columns=g_label_columns)

        # Arguments for the model. Different input shape requires different NN configuration
        # Pool size divides by 4 to reduce dimensionality
        pool_size = [4, 2]
        input_size = (input_width, len(
            window_14x7.column_indices))  # New input shape
        output_size = (label_width, len(
            g_label_columns))  # New output shape

        # Generates a new model with custom I/O, kernel and pool size
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  pool_size,
                                  input_size,
                                  output_size)

        # Compiles and fits using a window that sees 14 days to predict 7 days
        self.history = compile_and_fit(model, window_14x7)


class Test_TestModels(unittest.TestCase):
    """
    Test class with variations of the architectural configurations of the model.
    This includes changes in the kernel size (without changing the I/O shape),
    And connections inside the model that requires a similar but different model.
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

    def test_model_1(self):
        """
        Function that variates the kernel size between convolutional layers
        but using the same I/O shape.
        """
        global g_input_size, g_output_size, g_window, g_filter_size

        # Name used to identify its data in the history
        self.name = "model_1"

        # Arguments for the model.
        filter_size = [32, 32, 32, 32, 224]
        kernel_size = [12, 10, 4, 12]
        pool_size = 2

        # Generates a model with custom kernel size and pooling layer
        model = mg.deep_conv_lstm(filter_size,
                                  kernel_size,
                                  pool_size,
                                  g_input_size,
                                  g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_model_2(self):
        """
        Function that variates the kernel size between convolutional layers
        but using the same I/O shape.
        """
        global g_input_size, g_output_size, g_window, g_filter_size

        # Name used to identify its data in the history
        self.name = "model_2"

        # Arguments for the model.
        filter_size = [192, 192, 192, 192, 64]
        kernel_size = [6, 8, 8, 2]
        pool_size = 2

        # Generates a model with custom kernel size and pooling layer
        model = mg.deep_conv_lstm(filter_size,
                                  kernel_size,
                                  pool_size,
                                  g_input_size,
                                  g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


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

    def test_batch_128(self):
        global g_window, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_pool_size

        # Name used to identify its data in the history
        self.name = "batch_128"

        # *** Window
        window_128 = copy.deepcopy(g_window)
        window_128.batch_size = 128

        # Generates a generic model
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  g_pool_size,
                                  g_input_size,
                                  g_output_size)

        # Compiles and fits using a generic window with batch size of 128
        self.history = compile_and_fit(model, window_128)

    def test_batch_512(self):
        global g_window, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_pool_size

        # Name used to identify its data in the history
        self.name = "batch_512"

        # *** Window
        window_512 = copy.deepcopy(g_window)
        window_512.batch_size = 512

        # Generates a generic model
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  g_pool_size,
                                  g_input_size,
                                  g_output_size)

        # Compiles and fits using a generic window with batch size of 512
        self.history = compile_and_fit(model, window_512)

    def test_batch_256(self):
        global g_window, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_pool_size

        # Name used to identify its data in the history
        self.name = "batch_256"

        # *** Window
        window_256 = copy.deepcopy(g_window)
        window_256.batch_size = 256

        # Generates a generic model
        model = mg.deep_conv_lstm(g_filter_size,
                                  g_kernel_size,
                                  g_pool_size,
                                  g_input_size,
                                  g_output_size)

        # Compiles and fits using a generic window with batch size of 256
        self.history = compile_and_fit(model, window_256)


if __name__ == '__main__':
    unittest.main()
