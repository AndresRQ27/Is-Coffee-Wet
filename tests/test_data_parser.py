import unittest
import numpy as np
from pandas import read_csv
from IsCoffeeWet import data_parser


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("resources/test.csv")

    def test_merge_datetime(self):
        datetimeDS = data_parser.merge_datetime(
            self.dirtyDataset, "Date", "Time")

        # Original dataset has 5 rows: Datetime, Date, Time, Temp Out,
        # Leaf Wet 1 .New dataset has 2 rows: Temp Out, Leaf Wet 1.
        self.assertEqual(len(datetimeDS.columns) + 3,
                         len(self.dirtyDataset.columns))

    def test_convert_numeric(self):
        # INFO: Needs datetime index to interpolate by time
        dataset = data_parser.merge_datetime(self.dirtyDataset, "Date", "Time")

        convert_ds = data_parser.convert_numeric(dataset,
                                              [("Temp Out", "float"),
                                               ("Leaf Wet 1", "unsigned")],
                                              ["---"])

        print(convert_ds.info(verbose=True))

        # Checks the table for np.NaN
        # Uses any(0) to group all the rows in a value for each column
        # Uses any(0) again to group the True and False into a single result
        # If the result is false, all instances have been replaced
        self.assertFalse(convert_ds.loc[convert_ds.isna()].any(0).any(0))

    def test_sample_dataset(self):
        # Merge into datetime and uses it as index
        dataset = data_parser.merge_datetime(self.dirtyDataset, "Date", "Time")

        # Casts the dataset's columns into the needed dtypes
        dataset = data_parser.convert_numeric(dataset,
                                            [("Temp Out", "float"),
                                             ("Leaf Wet 1", "unsigned")],
                                            ["---"])

        sample_ds = data_parser.sample_dataset(dataset,
                                            [("Temp Out", np.mean),
                                             ("Leaf Wet 1", "last")],
                                            "15min")

        self.assertEqual(sample_ds.index.freq, "15T")

    def test_cyclical_encoder(self):
        # Merge into datetime and uses it as index
        dataset = data_parser.merge_datetime(self.dirtyDataset, "Date", "Time")
        num_columns = dataset.shape[1]

        # Encode the days into 2 additional column
        encoded_ds = data_parser.cyclical_encoder(dataset, [("Day", 60*60*24)])

        #Sin/Cos columns added to the new dataset
        with self.subTest:
            self.assertEquals(num_columns+2, encoded_ds.shape[1])

        #Check if there's a column
        with self.subTest:
            self.assertFalse(encoded_ds["Day sin"].empty)


if __name__ == '__main__':
    unittest.main()
