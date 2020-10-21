import unittest

import os
import pandas as pd

from IsCoffeeWet import data_parser
from IsCoffeeWet import config_file as cf

PATH = os.getcwd() + "/resources/database"


class Test_TestDataParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_dataset = pd.read_csv(PATH + "/test.csv")

        # Uses a list of columns with the date as the config file
        cls.config_file = cf.ConfigFile()
        cls.config_file.datetime = ["Date", "Time"]
        cls.config_file.columns = ["Date", "Time"]
        cls.config_file.datetime_format = "%d/%m/%Y %I:%M %p"

        cls.datetime_ds = data_parser.merge_datetime(raw_dataset, cls.config_file)

        cls.config_file.null = ["---"]
        cls.config_file.columns.extend(["Temp Out", "Leaf Wet 1"])
        cls.config_file.formats = {"Leaf Wet 1": "int"}
        cls.converted_ds = data_parser.convert_numeric(cls.datetime_ds, cls.config_file)

    def test_merge_datetime(self):
        result = pd.date_range("2010-04-07 00:00:00",
                               "2010-04-07 00:04:00",
                               freq="min")

        # Test to see if the resulting format is as the desired one
        # All the values must match to generate a True
        self.assertTrue(all(result == self.datetime_ds.head(5).index))

    def test_convert_numeric(self):
        # Test if there is a NaN value
        # noinspection SpellCheckingInspection
        with self.subTest(msg="NaN test"):
            # isna() returns False in value is NaN
            # Use all() to detect if there is a single False value (NaN)
            # First all check for True in all columns
            # Second all check for True across all columns
            self.assertTrue((self.converted_ds.notna()).all().all())

        # Test if all values are converted
        with self.subTest(msg="dtypes test"):
            # Check if all columns are float64
            self.assertTrue((self.converted_ds.dtypes == "float64").all())

    # noinspection DuplicatedCode
    def test_sample_dataset(self):
        self.config_file.freq = "15min"
        self.config_file.functions = {"Leaf Wet 1": "last"}

        sampled_ds = data_parser.sample_dataset(self.converted_ds, self.config_file)

        # Checks the new frequency of the dataset
        with self.subTest(msg="freq test"):
            self.assertEqual(sampled_ds.index.freq, "15T")

        with self.subTest(msg="check 'Leaf Wet Accum'"):
            self.assertTrue("Leaf Wet Accum" in sampled_ds.columns)

    def test_cyclical_encoder(self):
        self.config_file.encode = {"day": 86400}

        encoded_ds = data_parser.cyclical_encoder(self.datetime_ds, self.config_file)

        # Sin/Cos columns added to the new dataset
        with self.subTest(msg="check sin/cos test"):
            self.assertTrue(("day sin" in encoded_ds.columns)
                            and "day cos" in encoded_ds.columns)

        # Check if there values are between -1 and 1
        with self.subTest(msg="check between 0 and 1"):
            self.assertTrue((min(encoded_ds["day sin"]) >= -1)
                            and max((encoded_ds["day sin"]) <= 1))


if __name__ == '__main__':
    unittest.main()
