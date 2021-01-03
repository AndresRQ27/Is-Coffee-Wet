import os
import unittest
import pandas as pd

from IsCoffeeWet.utils.config_file import ConfigFile
from IsCoffeeWet.preprocess import data_derived as dd

PATH = os.getcwd() + "/resources"


class Test_TestDataDerived(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path_dataset = PATH + "/tests/database/test_parsed.csv"
        dataset = pd.read_csv(path_dataset,
                              engine="c", index_col="Datetime",
                              parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)
        cls.config_file = ConfigFile("tests/configs/test.json",
                                     PATH)

    def test_cyclical_encoder(self):
        encoded_ds = dd.create_cyclical_encoder(dataset_index=self.dataset.index,
                                                config_file=self.config_file)

        # Sin/Cos columns added to the new dataset
        with self.subTest(msg="check sin/cos test"):
            self.assertTrue(("day sin" in encoded_ds.columns) and
                            ("day cos" in encoded_ds.columns) and
                            ("year sin" in encoded_ds.columns) and
                            ("year cos" in encoded_ds.columns))

        # Check if there values are between -1 and 1
        with self.subTest(msg="check between 0 and 1"):
            self.assertTrue((min(encoded_ds["day sin"]) >= -1)
                            and max((encoded_ds["day sin"]) <= 1))

    def test_leaf_wet_accum(self):
        leaf_wet_accum = dd.create_leaf_wet_accum(dataset=self.dataset,
                                                  config_file=self.config_file)
        with self.subTest(msg="test series name"):
            # Return serie should have the correct name when added
            # to the dataset
            self.assertTrue(leaf_wet_accum.name == "Leaf Wet Accum")

        with self.subTest(msg="NaN test"):
            # In a dataset with complete values, none should be NaN
            self.assertTrue(leaf_wet_accum.notna().all())

        with self.subTest(msg="Index test"):
            # Index must be the same to concatenate the serie with
            # the dataset
            self.assertTrue(
                (leaf_wet_accum.index == self.dataset.index).all())
        pass


if __name__ == '__main__':
    unittest.main()
