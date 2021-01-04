import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from IsCoffeeWet.tools.config_file import ConfigFile
from IsCoffeeWet.preprocess import normalize as norm

PATH = os.getcwd()


class Test_TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column
        dataset = pd.read_csv(PATH + "/resources/tests/database/test_parsed.csv",
                              engine="c", index_col=0, parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)
        cls.config_file = ConfigFile("resources/tests/configs/test.json",
                                     PATH)
        cls.config_file.num_data = len(dataset)

    def test_standardize(self):
        standardize_ds, _, _ = norm.standardize(self.dataset)

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
        standardize_ds, mean, std = norm.standardize(self.dataset)
        restored_ds = norm.de_standardize(standardize_ds, mean, std)

        # Calculates the smallest value non-zero representable and multiplies by 10
        least_value = np.finfo("float64").resolution * 10

        # Restored dataset must be equal to the original
        self.assertTrue(
            (self.dataset - restored_ds < least_value).all().all())

    def test_normalize(self):
        normalized_ds, _ = norm.normalize(self.dataset)

        # Only max of those columns as the sin/cos won't change
        max_normalize = normalized_ds[[
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max normalize must always be less than 1
        self.assertTrue(all(1 > max_normalize))

    def test_de_normalize(self):
        # We first need to normalize a dataset
        normalize_ds, max_value = norm.normalize(self.dataset)
        restored_ds = norm.de_normalize(normalize_ds, max_value)

        # Calculates the smallest value non-zero representable and multiplies by 10
        least_value = np.finfo("float64").resolution * 10

        # Restored dataset must be equal to the original
        self.assertTrue(
            (self.dataset - restored_ds < least_value).all().all())


if __name__ == '__main__':
    unittest.main()
