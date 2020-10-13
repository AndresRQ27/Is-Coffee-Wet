import copy
import unittest

import pandas as pd
import tensorflow as tf

from IsCoffeeWet import config_file as cf
from IsCoffeeWet import model_generator
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import window_generator as wg

# Path for Linux
# PATH_BENCHMARK = "/media/andres/DATA/Code-Projects/Is-Coffee-Wet/resources/benchmark/"
# Path for Windows
# PATH_BENCHMARK = "D:/VMWare/Shared/Is-Coffee-Wet/resources/benchmark/"
# Path for Docker
PATH_BENCHMARK = "/opt/project/resources/benchmark/"

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
g_pool_size: int
g_input_size: tuple
g_output_size: tuple
max_epochs: int

all_history: pd.DataFrame


# ************************************

def setUpModule():
    """
    Set up module that executes before any test classes.

    Initialize all the global values used by the latter tests.
    """

    global g_dataset, g_config, g_train, g_val, g_test, g_window, all_history
    global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size, max_epochs

    # *** Dataset
    # Loads the dataset
    g_dataset = pd.read_csv(PATH_BENCHMARK + "benchmark_hour.csv",
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
    g_dataset = nn.standardize(g_dataset)

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
    g_filter_size = 32  # Neurons in a conv layer
    g_kernel_size = 24  # The kernel will see a day of data
    g_pool_size = 2  # Pooling of the data to reduce the dimensions
    g_input_size = (input_width, g_dataset.shape[1])  # Input size of the model
    g_output_size = (label_width, len(label_columns))  # Output size of the model
    max_epochs = 100  # Max training epochs

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
    csv_file = PATH_BENCHMARK + "benchmark_convolutional.csv"
    with open(csv_file, mode='w') as file:
        all_history.to_csv(file)


def compile_and_fit(model, window, patience=5, learning_rate=0.001):
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

    Returns
    -------

    tf.keras.callbacks.History
        Objects that contains the history of the model training.
    """
    global max_epochs

    # TODO: add tensorboard to the callback
    # Sets an early stopping callback to prevent over-fitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

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
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "generic"

        # Compiles the model with the default values
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


class Test_TestFilters(unittest.TestCase):
    """
    Test class with variations of the filter size (amount of neurons in a
    convolutional layer).
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

    def test_filter_16(self):
        """
        Test case with a filter size of 16
        """
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "filter_16"

        # Filter size to test
        filter_size = 16

        # Compiles the model with the custom filter size
        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_filter_64(self):
        """
        Test case with a filter size of 64
        """
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "filter_64"

        # Filter size to test
        filter_size = 64

        # Compiles the model with the custom filter size
        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_filter_128(self):
        """
        Test case with a filter size of 128
        """
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "filter_128"

        # Filter size to test
        filter_size = 128

        # Compiles the model with the custom filter size
        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


class Test_TestDropout(unittest.TestCase):
    """
    Test class with variations of the architectural layer. Removes the dropout
    layer in the model and test the performance with the validation set.
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

    def test_no_dropout(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "dropout"

        # Generates a model without dropout layers
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size,
                                                    None)

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
        datasets stored in the window
        """
        global g_filter_size, g_kernel_size, g_pool_size, g_test, g_val, g_train
        global g_output_size, g_window

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
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
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
        dataset_day = pd.read_csv(PATH_BENCHMARK + "benchmark_day.csv",
                                  engine="c", index_col="Datetime", parse_dates=True)

        # *** Dataset preparation
        # Normalize the dataset
        dataset_day = nn.standardize(dataset_day)

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

        kernel_size = 2  # Days the filter will see
        pool_size = [2, 1]  # Pool size to reduce dimensions
        input_size = (input_width, g_dataset.shape[1])  # Model's input shape
        output_size = (label_width, len(label_columns))  # Model's output shape

        # Generates a model with custom kernel size as I/O are different
        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        # Compiles and fit the model using a window that sees days
        self.history = compile_and_fit(model, window_day)


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
        global g_train, g_val, g_test, g_dataset, g_filter_size

        # Name used to identify its data in the history
        self.name = "window_14"

        # *** Window
        input_width = 14 * 24  # Window of 14 days in hours
        label_width = input_width  # Label same width as the input
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 14 days with the same training/validation set
        window_14 = wg.WindowGenerator(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_ds=g_train,
                                       val_ds=g_val,
                                       test_ds=g_test,
                                       label_columns=label_columns)

        # Arguments for the model. Different input shape requires different NN configuration
        kernel_size = [48, 24, 12]  # Reduce the kernel size in each layer
        pool_size = 4  # Pool size divides by 4 to reduce dimensionality
        input_size = (input_width, g_dataset.shape[1])  # New input shape
        output_size = (label_width, len(label_columns))  # New output shape

        # Generates a new model with custom I/O, kernel and pool size
        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        # Compiles and fits using a window of 14 days to predict 14 days
        self.history = compile_and_fit(model, window_14)

    def test_window_14x7(self):
        """
        Function that uses a window size of 14 days in the past to predict
        the next 7 days.
        """
        global g_train, g_val, g_test, g_dataset, g_filter_size

        # Name used to identify its data in the history
        self.name = "window_14x7"

        # *** Window
        input_width = 14 * 24  # Reads 14 days in the past in hours
        label_width = 7 * 24  # Predicts 7 days in the future in hours
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 14 days that predicts the next 7 days
        window_14x7 = wg.WindowGenerator(input_width=input_width,
                                         label_width=label_width,
                                         shift=label_width,
                                         train_ds=g_train,
                                         val_ds=g_val,
                                         test_ds=g_test,
                                         label_columns=label_columns)

        # Arguments for the model. Different input shape requires different NN configuration
        kernel_size = [48, 24, 12]  # Reduce the kernel size in each layer
        pool_size = 4  # Pool size divides by 4 to reduce dimensionality
        input_size = (input_width, g_dataset.shape[1])  # New input shape
        output_size = (label_width, len(label_columns))  # New output shape

        # Generates a new model with custom I/O, kernel and pool size
        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
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

        # Creates a DataFrame with the history
        history_df = pd.DataFrame(self.history.history)

        # Creates a new column with the name of the test to identify its data
        history_df["name"] = self.name

        # Saves each model history into the global DataFrame
        all_history = all_history.append(history_df)

    def test_kernel_dynamic(self):
        """
        Function that variates the kernel size between convolutional layers
        but using the same I/O shape.
        """
        global g_input_size, g_output_size, g_window, g_filter_size

        # Name used to identify its data in the history
        self.name = "kernel_reduction"

        # Arguments for the model. Custom kernel size and pooling layer by extend
        kernel_size = [48, 24, 12]
        pool_size = [2, 3]

        # Generates a model with custom kernel size and pooling layer
        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)

    def test_dense_15(self):
        """
        Function that uses a model in which the last convolutional layer
        has an output of 15 (in contrast to 1 in all the other networks).

        Tests if more information before expanding the dimensionality with
        the dense layer can lead to better outputs.
        """
        global g_input_size, g_output_size, g_window

        # Name used to identify its data in the history
        self.name = "dense_15"

        # Arguments for the model. Custom kernel sizes as we need 15 outputs
        # in the last conv layer. Custom filter sizes are just flavor (no need to change)
        kernel_size = [12, 48, 1]
        filter_size = [48, 24, 12]
        pool_size = [2, 2]

        # Input shape is the same
        inputs = tf.keras.layers.Input(shape=g_input_size)
        x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                                   kernel_size=kernel_size.pop(0),
                                   activation="relu")(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.MaxPool1D(pool_size=pool_size.pop(0))(x)
        x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                                   kernel_size=kernel_size.pop(0),
                                   activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.MaxPool1D(pool_size=pool_size.pop(0))(x)
        # Layer must have an output of 15
        x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                                   kernel_size=kernel_size.pop(0),
                                   activation="relu")(x)
        # Dense units will only be the number of predictions, instead of (no. predictions) * (features)
        dense = tf.keras.layers.Dense(units=g_output_size[0],
                                      activation="linear")(x)
        outputs = tf.keras.layers.Reshape([g_output_size[0], g_output_size[1]])(dense)

        # Generates a model by using a functional method
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv_model")

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


class Test_TestLearning(unittest.TestCase):
    """
    Test class with variations to the learning rate used in the optimizer.
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

    def test_learning_small(self):
        """
        Function with a small learning rate than the generic.
        """
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "learning_small"

        # Generates a generic model
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Compiles using a small learning rate and fits the model
        self.history = compile_and_fit(model, g_window, 5, 0.0001)

    def test_learning_big(self):
        """
        Function with a big learning rate than the generic
        """
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        # Name used to identify its data in the history
        self.name = "learning_big"

        # Generates a generic model
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Compiles using a big learning rate and fits the model
        self.history = compile_and_fit(model, g_window, 5, 0.01)


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

    def test_batch_64(self):
        global g_train, g_val, g_test, g_dataset, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_pool_size

        # Name used to identify its data in the history
        self.name = "batch_64"

        # *** Window
        input_width = 7 * 24
        label_width = input_width  # Label same width as the input
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Generic window with a batch size of 64
        window_64 = wg.WindowGenerator(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_ds=g_train,
                                       val_ds=g_val,
                                       test_ds=g_test,
                                       label_columns=label_columns,
                                       batch_size=64)

        # Generates a generic model
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Compiles and fits using a generic window with batch size of 64
        self.history = compile_and_fit(model, window_64)

    def test_batch_128(self):
        global g_train, g_val, g_test, g_dataset, g_input_size, g_output_size
        global g_kernel_size, g_filter_size, g_pool_size

        # Name used to identify its data in the history
        self.name = "batch_128"

        # *** Window
        input_width = 7 * 24
        label_width = input_width  # Label same width as the input
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Generic window with a batch size of 128
        window_128 = wg.WindowGenerator(input_width=input_width,
                                        label_width=label_width,
                                        shift=label_width,
                                        train_ds=g_train,
                                        val_ds=g_val,
                                        test_ds=g_test,
                                        label_columns=label_columns,
                                        batch_size=128)

        # Generates a generic model
        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        # Compiles and fits using a generic window with batch size of 128
        self.history = compile_and_fit(model, window_128)


if __name__ == '__main__':
    unittest.main()
