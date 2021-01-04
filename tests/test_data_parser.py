import os
import unittest
import pandas as pd

from IsCoffeeWet.tools.config_file import ConfigFile
from IsCoffeeWet.preprocess import data_parser

PATH = os.getcwd()


class Test_TestDataParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_dataset = pd.read_csv(PATH + "/resources/tests/database/test.csv")

        cls.config_file = ConfigFile("resources/tests/configs/test.json",
                                     PATH)

        # Uses a list of columns with the date as the config file
        cls.datetime_ds = data_parser.merge_datetime(
            raw_dataset, cls.config_file)

        cls.converted_ds = data_parser.convert_numeric(
            cls.datetime_ds, cls.config_file)

    def test_merge_datetime(self):
        result = pd.date_range("2010-04-07 00:00:00",
                               "2010-04-07 00:04:00",
                               freq="min")

        # Test to see if the resulting format is as the desired one
        # All the values must match to generate a True
        self.assertTrue(all(result == self.datetime_ds.head(5).index))

    def test_convert_numeric(self):
        # Test if there is a NaN value
        with self.subTest(msg="NaN test"):
            # notna() returns False in value is NaN
            # Use all() to detect if there is a single False value (NaN)
            # First all check for True in all columns
            # Second all check for True across all columns
            self.assertTrue((self.converted_ds.notna()).all().all())

        # Test if all values are converted
        with self.subTest(msg="dtypes test"):
            # Check if all columns are float64
            self.assertTrue((self.converted_ds.dtypes == "float64").all())

    def test_sample_series(self):
        sampled_series = data_parser.sample_series(
            self.converted_ds["Temp Out"], self.config_file)

        # Checks the new frequency of the series
        with self.subTest(msg="freq test"):
            self.assertEqual(sampled_series.index.freq, "1H")


if __name__ == '__main__':
    unittest.main()
