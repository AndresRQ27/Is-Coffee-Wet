import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet.tools.config_file import ConfigFile
from IsCoffeeWet.neural_network.models import filternet_module as flm
from IsCoffeeWet.neural_network.models import temporal_convolutional as tcn
from IsCoffeeWet.neural_network import utils

PATH = os.getcwd()


class Test_TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = pd.read_csv(PATH + "/resources/tests/database/test_parsed.csv",
                              engine="c", index_col=0, parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)
        cls.config_file = ConfigFile("resources/tests/configs/test.json",
                                     PATH)

        cls.predictions = pd.read_csv(
            PATH + "/resources/tests/database/predictions.csv",
            engine="c", index_col=0)
        cls.config_file.num_data = len(dataset)

        # Dummies model. The amount of units is given by the test labels
        x = tf.ones((3, 3, 3))  # Dummy input
        cls.basic_model = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=3)])
        cls.tcn_model = tf.keras.Sequential([tcn.ResidualBlock(filters=16,
                                                               kernel_size=3)])
        cls.conv_lstm_model = tf.keras.Sequential(
            [flm.FilternetModule(w_out=16)])

        # Initialize the input of the layers
        cls.basic_model(x)
        cls.tcn_model(x)
        cls.conv_lstm_model(x)

    def test_split_dataset(self):
        datetime_index, train_ds, val_ds, test_ds = utils.split_dataset(
            self.dataset, self.config_file)

        # Original size must remain
        self.assertEqual(len(train_ds) + len(val_ds) +
                         len(test_ds), len(datetime_index))

    def test_save_model(self):
        # Check for the created file with the basic model
        with self.subTest(msg="Check basic_model"):
            utils.save_model(model=self.basic_model,
                             path=self.config_file.nn_path,
                             name="basic_model")
            self.assertTrue(os.path.isfile(
                self.config_file.nn_path + "/basic_model.h5"))

        # Check for the created file with the tcn model
        with self.subTest(msg="Check tcn_model"):
            utils.save_model(model=self.tcn_model,
                             path=self.config_file.nn_path,
                             name="tcn_model")
            self.assertTrue(os.path.isfile(
                self.config_file.nn_path + "/tcn_model.h5"))

        # Check for the created file with the conv lstm model
        with self.subTest(msg="Check conv_lstm_model"):
            utils.save_model(model=self.conv_lstm_model,
                             path=self.config_file.nn_path,
                             name="conv_lstm_model")
            self.assertTrue(os.path.isfile(
                self.config_file.nn_path + "/conv_lstm_model.h5"))

    def test_load_model(self):
        # Dummy input to see if the models have an output
        x = tf.ones((3, 3, 3))

        # Check for the loaded model of the basic model
        with self.subTest(msg="Check basic_model"):
            basic_model = utils.load_model(path=self.config_file.nn_path,
                                           name="basic_model")
            print(basic_model.summary())
            self.assertTrue(basic_model(x) is not None)

        # Check for the loaded model of the tcn model
        with self.subTest(msg="Check tcn_model"):
            tcn_model = utils.load_model(path=self.config_file.nn_path,
                                         name="tcn_model",
                                         submodel="tcn")
            print(tcn_model.summary())
            self.assertTrue(tcn_model(x) is not None)

        # Check for the loaded model of the conv lstm model
        with self.subTest(msg="Check conv_lstm_model"):
            conv_lstm_model = utils.load_model(path=self.config_file.nn_path,
                                               name="conv_lstm_model",
                                               submodel="conv_lstm")
            print(conv_lstm_model.summary())
            self.assertTrue(conv_lstm_model(x) is not None)

    def test_mae(self):
        # Resets the index to be numeric, as the predictions DataFrame
        dataset = self.dataset[self.predictions.columns].iloc[-len(
            self.predictions):]
        dataset.reset_index(inplace=True, drop=True)
        result = utils.mae(dataset, self.predictions)
        # TODO: validate test result
        self.assertTrue(True)

    def test_analyze_loss(self):
        dataset = self.dataset[self.predictions.columns].iloc[-len(
            self.predictions):]
        dataset.reset_index(inplace=True)
        datetime_index = dataset.pop("Datetime")
        result = utils.analyze_metrics(
            dataset, self.predictions, datetime_index, np.mean)
        # TODO: validate test result
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
