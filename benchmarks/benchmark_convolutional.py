import unittest

from IsCoffeeWet import model_generator

import copy
import pandas as pd
import tensorflow as tf

from IsCoffeeWet import window_generator as wg
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import config_file as cf

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
    g_config = cf.ConfigFile()
    g_config.training = 0.7
    g_config.validation = 0.2
    g_config.num_data, g_config.num_features = g_dataset.shape

    # *** Dataset preparation
    # Normalize the dataset
    g_dataset = nn.normalize(g_dataset)

    # Partition the dataset
    _, g_train, g_val, g_test = nn.split_dataset(g_dataset, g_config)

    # *** Window
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
    g_kernel_size = 24  # A day
    g_pool_size = 2
    g_input_size = (input_width, g_dataset.shape[1])
    g_output_size = (label_width, len(label_columns))
    max_epochs = 100

    # *** Dataframe
    # Dataframe use to store the history of each training, then save it
    all_history = pd.DataFrame()


def tearDownModule():
    global all_history

    # Save to csv:
    csv_file = PATH_BENCHMARK + "benchmark_convolutional.csv"
    with open(csv_file, mode='w') as file:
        all_history.to_csv(file)


def compile_and_fit(model, window, patience=5, learning_rate=0.001):
    # TODO documentation
    global max_epochs

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(learning_rate),
                  metrics=[tf.metrics.MeanAbsoluteError(),
                           tf.metrics.MeanAbsolutePercentageError()])

    # TODO: add tensorboard to the callback
    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


class Test_TestBase(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_generic_network(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "generic"

        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window)


class Test_TestFilters(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_filter_16(self):
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "filter_16"

        filter_size = 16

        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window)

    def test_filter_64(self):
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "filter_64"

        filter_size = 64

        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window)

    def test_filter_128(self):
        global g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "filter_128"

        filter_size = 128

        model = model_generator.convolutional_model(filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window)


class Test_TestDropout(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_no_dropout(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "dropout"

        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size,
                                                    None)

        self.history = compile_and_fit(model, g_window)


class Test_TestNoEncoding(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_no_encoding(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_test, g_val, g_train
        global g_output_size, g_window

        self.name = "no_encoding"

        # Copy the original window to modify only the sets
        window_ne = copy.deepcopy(g_window)

        # Drops the encoding for each set
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
        input_width = 24*7
        label_columns = window_ne.train_ds.shape[1]

        # Re-defines only the input size as it's the one that changed
        input_size = (input_width, label_columns)

        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, window_ne)


class Test_TestDay(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_day(self):
        global g_config

        self.name = "day"

        # *** Dataset. All parsing process for new dataset
        # Loads the dataset
        dataset_day = pd.read_csv(PATH_BENCHMARK + "benchmark_day.csv",
                                  engine="c", index_col="Datetime", parse_dates=True)

        # *** Dataset preparation
        # Normalize the dataset
        dataset_day = nn.normalize(dataset_day)

        # Copy config file
        config_day = copy.deepcopy(g_config)
        config_day.num_data = dataset_day.shape[0]

        # Partition the dataset
        _, train_day, val_day, test_day = nn.split_dataset(dataset_day, config_day)

        # *** Window
        input_width = 7
        label_width = input_width
        label_columns = dataset_day.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 7 days for testing the NN
        window_day = wg.WindowGenerator(input_width=input_width,
                                        label_width=label_width,
                                        shift=label_width,
                                        train_ds=train_day,
                                        val_ds=val_day,
                                        test_ds=test_day,
                                        label_columns=label_columns)

        kernel_size = 2
        pool_size = [2, 1]
        input_size = (input_width, g_dataset.shape[1])
        output_size = (label_width, len(label_columns))

        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, window_day)


class Test_TestWindow(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_window_14(self):
        global  g_train, g_val, g_test, g_dataset, g_filter_size

        self.name = "window_14"

        # *** Window
        input_width = 14 * 24
        label_width = input_width
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 7 days for testing the NN
        window_14 = wg.WindowGenerator(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_ds=g_train,
                                       val_ds=g_val,
                                       test_ds=g_test,
                                       label_columns=label_columns)

        # Arguments of the default NN
        kernel_size = [48, 24, 12]  # A day
        pool_size = 4
        input_size = (input_width, g_dataset.shape[1])
        output_size = (label_width, len(label_columns))

        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, window_14)

    def test_window_14x7(self):
        global  g_train, g_val, g_test, g_dataset, g_filter_size

        self.name = "window_14x7"

        # *** Window
        input_width = 14 * 24
        label_width = 7 * 24
        label_columns = g_dataset.columns.tolist()

        # Removes th sin/cos columns from the labels
        label_columns = label_columns[:-4]

        # Window of 7 days for testing the NN
        window_14x7 = wg.WindowGenerator(input_width=input_width,
                                       label_width=label_width,
                                       shift=label_width,
                                       train_ds=g_train,
                                       val_ds=g_val,
                                       test_ds=g_test,
                                       label_columns=label_columns)

        # Arguments of the default NN
        kernel_size = [48, 24, 12]  # A day
        pool_size = 4
        input_size = (input_width, g_dataset.shape[1])
        output_size = (label_width, len(label_columns))

        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, window_14x7)


class Test_TestModels(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_kernel_dynamic(self):
        global g_input_size, g_output_size, g_window, g_filter_size

        self.name = "kernel_reduction"

        # Arguments of the default NN
        kernel_size = [48, 24, 12]  # A day
        pool_size = [2, 3]

        model = model_generator.convolutional_model(g_filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window)

    def test_dense_15(self):
        global g_input_size, g_output_size, g_window

        self.name = "dense_15"

        # Arguments of the default NN
        kernel_size = [12, 48, 1]  # A day
        filter_size = [48, 24, 12]
        pool_size = [2, 2]

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
        x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                                   kernel_size=kernel_size.pop(0),
                                   activation="relu")(x)
        dense = tf.keras.layers.Dense(units=g_output_size[0],
                                      activation="linear")(x)
        outputs = tf.keras.layers.Reshape([g_output_size[0], g_output_size[1]])(dense)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv_model")

        self.history = compile_and_fit(model, g_window)


class Test_TestLearning(unittest.TestCase):
    def setUp(self):
        self.history: tf.keras.callbacks.History
        self.name: str

    def tearDown(self):
        global all_history

        history_df = pd.DataFrame(self.history.history)
        history_df["name"] = self.name

        all_history = all_history.append(history_df)

    def test_learning_small(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "learning_big"

        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window, 5, 0.0001)

    def test_learning_big(self):
        global g_filter_size, g_kernel_size, g_pool_size, g_input_size, g_output_size
        global g_window

        self.name = "learning_big"

        model = model_generator.convolutional_model(g_filter_size,
                                                    g_kernel_size,
                                                    g_pool_size,
                                                    g_input_size,
                                                    g_output_size)

        self.history = compile_and_fit(model, g_window, 5, 0.01)

# TODO: document all benchmarks

if __name__ == '__main__':
    unittest.main()
