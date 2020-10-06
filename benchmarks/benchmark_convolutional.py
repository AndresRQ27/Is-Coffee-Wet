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

dataset: pd.DataFrame
config_file: cf.ConfigFile
train_ds: pd.DataFrame
val_ds: pd.DataFrame
test_ds: pd.DataFrame
sliding_window: wg.WindowGenerator
filter_size: int
kernel_size: int
pool_size: int
input_size: tuple
output_size: tuple
max_epochs: int

all_history: pd.DataFrame


# ************************************

def setUpModule():
    global dataset, config_file, train_ds, val_ds, test_ds, sliding_window, all_history
    global filter_size, kernel_size, pool_size, input_size, output_size, max_epochs

    # *** Dataset
    # Loads the dataset
    dataset = pd.read_csv(PATH_BENCHMARK + "benchmark_hour.csv",
                          engine="c", index_col="Datetime", parse_dates=True)

    # Information of the dataset
    print(dataset.info(verbose=True))
    print(dataset.describe().transpose())

    # *** Config File
    config_file = cf.ConfigFile()
    config_file.training = 0.7
    config_file.validation = 0.2
    config_file.num_data, config_file.num_features = dataset.shape

    # *** Dataset preparation
    # Normalize the dataset
    dataset = nn.normalize(dataset)

    # Partition the dataset
    _, train_ds, val_ds, test_ds = nn.split_dataset(dataset, config_file)

    # *** Window
    input_width = 7 * 24
    label_width = input_width
    label_columns = dataset.columns.tolist()

    # Removes th sin/cos columns from the labels
    label_columns = label_columns[:-4]

    # Window of 7 days for testing the NN
    sliding_window = wg.WindowGenerator(input_width=input_width,
                                        label_width=label_width,
                                        shift=label_width,
                                        train_ds=train_ds,
                                        val_ds=val_ds,
                                        test_ds=test_ds,
                                        label_columns=label_columns)

    # Arguments of the default NN
    filter_size = 32  # Neurons in a conv layer
    kernel_size = 24  # A day
    pool_size = 2
    input_size = (input_width, dataset.shape[1])
    output_size = (label_width, len(label_columns))
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


def compile_and_fit(model, window, patience=5):
    # TODO documentation
    global max_epochs

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
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
        global filter_size, kernel_size, pool_size, input_size, output_size
        global sliding_window

        self.name = "generic"

        model = model_generator.convolutional_model(filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, sliding_window)


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
        global kernel_size, pool_size, input_size, output_size
        global sliding_window

        self.name = "filter_16"

        model = model_generator.convolutional_model(16,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, sliding_window)

    def test_filter_64(self):
        global kernel_size, pool_size, input_size, output_size
        global sliding_window

        self.name = "filter_64"

        model = model_generator.convolutional_model(64,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, sliding_window)

    def test_filter_128(self):
        global kernel_size, pool_size, input_size, output_size
        global sliding_window

        self.name = "filter_128"

        model = model_generator.convolutional_model(128,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, sliding_window)


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
        global filter_size, kernel_size, pool_size, input_size, output_size
        global sliding_window

        self.name = "dropout"

        model = model_generator.convolutional_model(filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size,
                                                    None)

        self.history = compile_and_fit(model, sliding_window)


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
        global filter_size, kernel_size, pool_size, test_ds, val_ds, train_ds
        global output_size

        self.name = "no_encoding"

        # Copy the original window to modify only the sets
        window_ne = copy.deepcopy(sliding_window)

        # Drops the encoding for each set
        window_ne.test_ds = test_ds.drop(["day sin",
                                    "day cos",
                                    "year sin",
                                    "year cos"], axis=1)

        window_ne.train_ds = train_ds.drop(["day sin",
                                    "day cos",
                                    "year sin",
                                    "year cos"], axis=1)

        window_ne.val_ds = val_ds.drop(["day sin",
                                    "day cos",
                                    "year sin",
                                    "year cos"], axis=1)

        # Re-define the input/output size for the model
        input_width = 24*7
        label_columns = window_ne.train_ds.shape[1]

        # Re-defines only the input size as it's the one that changed
        input_size = (input_width, label_columns)

        model = model_generator.convolutional_model(filter_size,
                                                    kernel_size,
                                                    pool_size,
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, sliding_window)


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
        global config_file

        self.name = "day"

        # *** Dataset. All parsing process for new dataset
        # Loads the dataset
        dataset_day = pd.read_csv(PATH_BENCHMARK + "benchmark_day.csv",
                                  engine="c", index_col="Datetime", parse_dates=True)

        # *** Dataset preparation
        # Normalize the dataset
        dataset_day = nn.normalize(dataset_day)

        # Copy config file
        config_day = copy.deepcopy(config_file)
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

        input_size = (input_width, dataset.shape[1])
        output_size = (label_width, len(label_columns))

        kernel_size = 2

        model = model_generator.convolutional_model(filter_size,
                                                    kernel_size,
                                                    [2, 1],
                                                    input_size,
                                                    output_size)

        self.history = compile_and_fit(model, window_day)

if __name__ == '__main__':
    unittest.main()
