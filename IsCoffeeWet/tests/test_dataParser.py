import unittest
import numpy as np
from pandas import read_csv
from IsCoffeeWet import dataParser


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("resources/est0Corta.csv")

    def test_cleanDataset(self):
        testDS = dataParser.cleanDataset(self.dirtyDataset,
                                         ["Temp Out", "Leaf Wet 1"],
                                         "---")

        self.assertLess(testDS.size, self.dirtyDataset.size)

    def test_setDataTypes(self):
        dataset = dataParser.cleanDataset(self.dirtyDataset,
                                          ["Temp Out",
                                           "Leaf Wet 1"],
                                          "---")
        testDS = dataParser.setDataTypes(dataset,
                                         ("Date", "Time"),
                                         [("Temp Out", "float32"),
                                          ("Leaf Wet 1", "float32")])

        self.assertLess(len(testDS.columns), len(dataset.columns))

    def test_sampleDataset(self):
        dataset = dataParser.cleanDataset(self.dirtyDataset,
                                          ["Temp Out",
                                           "Leaf Wet 1"],
                                          "---")
        dataset = dataParser.setDataTypes(dataset,
                                          ("Date", "Time"),
                                          [("Temp Out", "float32"),
                                           ("Leaf Wet 1", "float32")])

        # Check warning output
        with self.subTest():
            with self.assertWarns(Warning):
                dataParser.sampleDataset(dataset)

        with self.subTest():
            # TODO: cambiar np.max de Leaf Wet 1
            testDS = dataParser.sampleDataset(dataset,
                                              [("Temp Out", np.mean),
                                               ("Leaf Wet 1", np.max)],
                                              "15min")
            print(testDS.info(verbose=True))
            self.assertLess(testDS.size, dataset.size)


if __name__ == '__main__':
    unittest.main()

#import IsCoffeeWet.dataParser as dp
#import pandas as pd
#dataset = pd.read_csv("resources/est0Corta.csv")
#dataset = dp.cleanDataset(dataset,["Temp Out", "Leaf Wet 1"],"---")
#dataset = dp.setDataTypes(dataset,("Date","Time"),[("Temp Out","float32"),("Leaf Wet 1","int32")])
#dataset = dp.sampleDataset(dataset, "1H")
