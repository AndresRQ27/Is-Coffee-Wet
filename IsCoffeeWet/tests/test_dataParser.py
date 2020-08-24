import unittest
import numpy as np
from pandas import read_csv, DataFrame
from IsCoffeeWet import dataParser


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("resources/est0Corta.csv")

    def test_mergeDateTime(self):
        datetimeDS = dataParser.mergeDateTime(
            self.dirtyDataset, "Date", "Time")

        # New dataset must have less rows as 2 of them have been merged
        self.assertNotEqual(len(datetimeDS.columns),
                            len(self.dirtyDataset.columns))

    def test_convertNumeric(self):
        # INFO: Needs datetime index to interpolate by time
        dataset = dataParser.mergeDateTime(self.dirtyDataset, "Date", "Time")

        convertDS = dataParser.convertNumeric(dataset,
                                              [("Temp Out", "float"),
                                               ("Leaf Wet 1", "unsigned")],
                                              "---")

        print(convertDS.info(verbose=True))

        # Checks the table for np.NaN
        # Uses any(0) to group all the rows in a value for each column
        # Uses any(0) again to group the True and False into a single result
        # If the result is false, all instances have been replaced
        self.assertFalse(convertDS.eq(np.NaN).any(0).any(0))

    def test_sampleDataset(self):
        # Merge into datetime and uses it as index
        dataset = dataParser.mergeDateTime(self.dirtyDataset, "Date", "Time")

        # Casts the dataset's columns into the needed dtypes
        dataset = dataParser.convertNumeric(dataset,
                                            [("Temp Out", "float"),
                                             ("Leaf Wet 1", "unsigned")],
                                            "---")

        sampleDS = dataParser.sampleDataset(dataset,
                                            [("Temp Out", np.mean),
                                             ("Leaf Wet 1", "last")],
                                            "15min")

        self.assertEquals(sampleDS.index.freq, "15T")


if __name__ == '__main__':
    unittest.main()
