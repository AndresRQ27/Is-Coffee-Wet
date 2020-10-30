import copy
import os
import unittest

import pandas as pd
import tensorflow as tf

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
g_pool_size: int
g_input_size: tuple

all_history: pd.DataFrame


# ************************************

def setUpModule():
    """
    Set up module that executes before any test classes.

    Initialize all the global values used by the latter tests.
    """

    global g_window, all_history, g_input_size
    global g_filter_size, g_kernel_size, g_pool_size

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

    # Drops the encoding for each set present in the window
    dataset = dataset.drop(["day sin",
                            "day cos",
                            "year sin",
                            "year cos"], axis=1)

    # Normalize the dataset
    dataset, _, _ = nn.standardize(dataset)

    # Partition the dataset
    _, train_ds, val_ds, _ = nn.split_dataset(dataset, config)

    # *** Window
    # A week in hours
    input_width = 7 * 24
    label_columns = dataset.columns.tolist()

    # Removes th sin/cos columns from the labels
    label_columns = label_columns[:-4]

    # Window of 7 days for testing the NN. Batch size of 512
    g_window = wg.WindowGenerator(input_width=input_width,
                                  label_width=input_width,
                                  shift=input_width,
                                  train_ds=train_ds,
                                  val_ds=val_ds,
                                  test_ds=_,
                                  label_columns=label_columns,
                                  batch_size=512)

    # Arguments of the default NN. Use kernel increment values
    g_filter_size = [96, 192, 208]  # Neurons in a conv layer
    g_kernel_size = [5, 11, 24]  # The kernel will see a day of data
    g_pool_size = [2, 3]  # Pooling of the data to reduce the dimensions
    g_input_size = (input_width, dataset.shape[1])  # Input size of the model

    # *** Dataframe
    # Dataframe use to store the history of each training, then save it
    try:
        # Overwrites past results
        all_history = pd.read_csv(PATH + "/results/benchmark_labels_convolutional.csv",
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
    csv_file = PATH + "/results/benchmark_labels_convolutional.csv"
    with open(csv_file, mode='w') as file:
        all_history.to_csv(file)


def compile_and_fit(model, window, patience=10, learning_rate=0.0001,
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
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

        # Name used to identify its data in the history
        self.name = "generic"

        output_size = (g_input_size[0], len(g_window.label_columns))

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, g_window)


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

    def test_temp_out(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_high_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_low_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_out_hum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_wind_speed(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_hi_speed(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_pool_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_bar(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_rain(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_solar_rad(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_hi_solar_rad(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_in_temp(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_in_hum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_soil_moist(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_leaf_wet(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)

    def test_leaf_wet_accum(self):
        """
        Function that compiles and train the default CNN to use as baseline.
        """
        global g_filter_size, g_kernel_size, g_pool_size
        global g_window, g_input_size

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

        # Compiles the model with the default values
        model = mg.convolutional_model(g_filter_size,
                                       g_kernel_size,
                                       g_pool_size,
                                       g_input_size,
                                       output_size)

        # Train the model using the default window
        self.history = compile_and_fit(model, window)


if __name__ == '__main__':
    unittest.main()
