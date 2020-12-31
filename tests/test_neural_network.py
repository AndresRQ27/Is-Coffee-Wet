import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet import config_file as cf
from IsCoffeeWet import filternet_module as flm
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import temporal_convolutional as tcn

PATH = os.getcwd() + "/resources/tests"


class Test_TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column

        dataset = pd.read_csv(PATH + "/database/test_parsed.csv",
                              engine="c", index_col="Datetime", parse_dates=True)
        cls.predictions = pd.read_csv(PATH + "/database/predictions.csv",
                                      engine="c", index_col=0)

        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)

        cls.config_file = cf.ConfigFile()
        cls.config_file.training = 0.7
        cls.config_file.validation = 0.2
        cls.config_file.num_data = len(dataset)
        cls.config_file.forecast = 168
        cls.config_file.labels = [
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]
        cls.config_file.path = PATH + "/neural-network"

        # Dummies model. The amount of units is given by the test labels
        x = tf.ones((3, 3, 3))  # Dummy input
        cls.basic_model = tf.keras.Sequential([tf.keras.layers.Dense(units=3)])
        cls.tcn_model = tf.keras.Sequential([tcn.ResidualBlock(filters=16,
                                                               kernel_size=3)])
        cls.conv_lstm_model = tf.keras.Sequential([flm.FilternetModule(w_out=16)])

        # Initialize the input of the layers
        cls.basic_model(x)
        cls.tcn_model(x)
        cls.conv_lstm_model(x)

    def test_standardize(self):
        standardize_ds, _, _ = nn.standardize(self.dataset)

        # Only max of these columns as the sin/cos won't change
        max_original = self.dataset[[
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()
        max_standardize = standardize_ds[[
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max standardize must always be less than the original
        with self.subTest(msg="compare vs original"):
            # noinspection PyTypeChecker
            self.assertTrue(all(max_original > max_standardize))

        # The standardize dataset must be less than 4 std to be correct
        # At least in this dataset
        with self.subTest(msg="check if less than 4"):
            self.assertTrue(all(4 > max_standardize))

    def test_de_standardize(self):
        # We first need to standardize a dataset
        standardize_ds, mean, std = nn.standardize(self.dataset)
        restored_ds = nn.de_standardize(standardize_ds, mean, std)

        # Calculates the smallest value non-zero representable and multiplies by 10
        least_value = np.finfo("float64").resolution * 10

        # Restored dataset must be equal to the original
        self.assertTrue(
            (self.dataset - restored_ds < least_value).all().all())

    def test_normalize(self):
        normalized_ds, _ = nn.normalize(self.dataset)

        # Only max of those columns as the sin/cos won't change
        max_normalize = normalized_ds[[
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max normalize must always be less than 1
        self.assertTrue(all(1 > max_normalize))

    def test_de_normalize(self):
        # We first need to normalize a dataset
        normalize_ds, max_value = nn.normalize(self.dataset)
        restored_ds = nn.de_normalize(normalize_ds, max_value)

        # Calculates the smallest value non-zero representable and multiplies by 10
        least_value = np.finfo("float64").resolution * 10

        # Restored dataset must be equal to the original
        self.assertTrue(
            (self.dataset - restored_ds < least_value).all().all())

    def test_split_dataset(self):
        datetime_index, train_ds, val_ds, test_ds = nn.split_dataset(
            self.dataset, self.config_file)

        # Original size must remain
        self.assertEqual(len(train_ds) + len(val_ds) +
                         len(test_ds), len(datetime_index))

    def test_save_model(self):
        # Check for the created file with the basic model
        with self.subTest(msg="Check basic_model"):
            nn.save_model(model=self.basic_model,
                          path=self.config_file.path + "/basic_model.h5")
            self.assertTrue(os.path.isfile(
                self.config_file.path + "/basic_model.h5"))

        # Check for the created file with the tcn model
        with self.subTest(msg="Check tcn_model"):
            nn.save_model(model=self.tcn_model,
                          path=self.config_file.path + "/tcn_model.h5")
            self.assertTrue(os.path.isfile(
                self.config_file.path + "/tcn_model.h5"))

        # Check for the created file with the conv lstm model
        with self.subTest(msg="Check conv_lstm_model"):
            nn.save_model(model=self.conv_lstm_model,
                          path=self.config_file.path + "/conv_lstm_model.h5")
            self.assertTrue(os.path.isfile(
                self.config_file.path + "/conv_lstm_model.h5"))

    def test_load_model(self):
        # Dummy input to see if the models have an output
        x = tf.ones((3, 3, 3))

        # Check for the loaded model of the basic model
        with self.subTest(msg="Check basic_model"):
            basic_model = nn.load_model(
                path=self.config_file.path + "/basic_model.h5")
            print(basic_model.summary())
            self.assertTrue(basic_model(x) is not None)

        # Check for the loaded model of the tcn model
        with self.subTest(msg="Check tcn_model"):
            tcn_model = nn.load_model(path=self.config_file.path + "/tcn_model.h5",
                                      submodel="tcn")
            print(tcn_model.summary())
            self.assertTrue(tcn_model(x) is not None)

        # Check for the loaded model of the conv lstm model
        with self.subTest(msg="Check conv_lstm_model"):
            conv_lstm_model = nn.load_model(path=self.config_file.path + "/conv_lstm_model.h5",
                                            submodel="conv_lstm")
            print(conv_lstm_model.summary())
            self.assertTrue(conv_lstm_model(x) is not None)

    def test_mae(self):
        # Resets the index to be numeric, as the predictions DataFrame
        dataset = self.dataset[self.predictions.columns].iloc[-len(self.predictions):]
        dataset.reset_index(inplace=True, drop=True)
        result = nn.mae(dataset, self.predictions)
        # TODO: validate test result
        self.assertTrue(True)

    def test_analyze_loss(self):
        dataset = self.dataset[self.predictions.columns].iloc[-len(self.predictions):]
        dataset.reset_index(inplace=True)
        datetime_index = dataset.pop("Datetime")
        result = nn.analyze_loss(dataset, self.predictions, datetime_index, np.mean)
        # TODO: validate test result
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
