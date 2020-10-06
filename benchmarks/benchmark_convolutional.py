import unittest

from IsCoffeeWet import model_generator

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
    dataset = pd.read_csv(PATH_BENCHMARK + "estXCompleta_parsed.csv",
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


if __name__ == '__main__':
    unittest.main()
