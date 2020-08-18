import unittest
import numpy as np
from pandas import read_csv
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

    def test_cleanDataset(self):
        clearDS = dataParser.cleanDataset(self.dirtyDataset,
                                          ["Temp Out", "Leaf Wet 1"],
                                          "---")

        # Checks the table for "---" aka nullvalue
        # Uses any(0) to group all the rows in a value for each column
        # Uses any(0) again to group the True and False into a single result
        # If the result is false, all instances have been replaced
        self.assertFalse(clearDS.eq("---").any(0).any(0))

    def test_convertNumeric(self):
        # INFO: Needs datetime index to interpolate by time
        dataset = dataParser.mergeDateTime(self.dirtyDataset, "Date", "Time")
        # Cleans the dataset of null values
        dataset = dataParser.cleanDataset(dataset,
                                          ["Temp Out", "Leaf Wet 1"],
                                          "---")

        convertDS = dataParser.convertNumeric(dataset,
                                           [("Temp Out", "float32"),
                                            ("Leaf Wet 1", "float32")])

        # Checks the table for np.NaN
        # Uses any(0) to group all the rows in a value for each column
        # Uses any(0) again to group the True and False into a single result
        # If the result is false, all instances have been replaced
        self.assertFalse(convertDS.eq(np.NaN).any(0).any(0))

    def test_sampleDataset(self):
        # Merge into datetime and uses it as index
        dataset = dataParser.mergeDateTime(self.dirtyDataset, "Date", "Time")
        # Cleans the dataset of null values
        dataset = dataParser.cleanDataset(dataset,
                                          ["Temp Out", "Leaf Wet 1"],
                                          "---")
        # Casts the dataset's columns into the needed dtypes
        dataset = dataParser.convertNumeric(dataset,
                                            [("Temp Out", "float32"),
                                             ("Leaf Wet 1", "float32")])

        # TODO: cambiar np.max de Leaf Wet 1
        sampleDS = dataParser.sampleDataset(dataset,
                                            [("Temp Out", np.mean),
                                            ("Leaf Wet 1", np.max)],
                                            "15min")
        print(sampleDS.info(verbose=True))
        self.assertLess(sampleDS.size, dataset.size)


if __name__ == '__main__':
    unittest.main()

#import IsCoffeeWet.dataParser as dp
#import pandas as pd
#dataset = pd.read_csv("resources/est0Corta.csv")
#dataset = dp.cleanDataset(dataset,["Temp Out", "Leaf Wet 1"],"---")
#dataset = dp.convertNumeric(dataset,("Date","Time"),[("Temp Out","float32"),("Leaf Wet 1","int32")])
#dataset = dp.sampleDataset(dataset, "1H")
