import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet import config_file as cf
from IsCoffeeWet import neural_network as nn

PATH = os.getcwd() + "/resources"


class Test_TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column

        dataset = pd.read_csv(PATH + "/database/test_parsed.csv",
                              engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)

        cls.config_file = cf.ConfigFile()
        cls.config_file.training = 0.7
        cls.config_file.validation = 0.2
        cls.config_file.num_data = len(dataset)
        cls.config_file.forecast = 168
        cls.config_file.labels = ["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]

        # A dummy model. The amount of units is given by the test labels
        cls.model = tf.keras.Sequential([tf.keras.layers.Dense(units=3)])

    def test_normalize(self):
        normalized_ds, _, _ = nn.standardize(self.dataset)

        # Only max of those columns as the sin/cos won't change
        max_original = self.dataset[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()
        max_normalize = normalized_ds[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max standardize must always be less than the original
        with self.subTest(msg="compare vs original"):
            # noinspection PyTypeChecker
            self.assertTrue(all(max_original > max_normalize))

        # The standardize dataset must be less than 4 std to be correct
        # At least in this dataset
        with self.subTest(msg="check if less than 4"):
            self.assertTrue(all(4 > max_normalize))

    def test_split_dataset(self):
        datetime_index, train_ds, val_ds, test_ds = nn.split_dataset(self.dataset, self.config_file)

        # Original size must remain
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), len(datetime_index))

    def test_de_normalize(self):
        # We first need to normalize a dataset
        normalized_ds, mean, std = nn.standardize(self.dataset)
        restored_ds = nn.de_standardize(normalized_ds, mean, std)

        # Calculates the smallest value non-zero representable and multiplies by 10
        least_value = np.finfo("float64").resolution * 10

        # Restored dataset must be equal to the original
        self.assertTrue((self.dataset - restored_ds < least_value).all().all())

    def test_mape(self):
        nn.mape(self.dataset, self.config_file, self.model)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
