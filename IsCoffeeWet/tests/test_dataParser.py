import unittest
from pandas import read_csv
from IsCoffeeWet import dataParser


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("resources/est0Corta.csv")

    def test_cleanDataset(self):
        filteredDataset = dataParser.cleanDataset(self.dirtyDataset,
                                                  ["Temp Out", "Leaf Wet 1"],
                                                  "---")

        self.assertLess(filteredDataset.size, self.dirtyDataset.size)

    def test_setDataTypes(self):
        filteredDataset = dataParser.cleanDataset(self.dirtyDataset,
                                                  ["Temp Out", 
                                                  "Leaf Wet 1"],
                                                  "---")
        typedDataset = dataParser.setDataTypes(filteredDataset,
                                               ("Date", "Time"),
                                               [("Temp Out", "float32"),
                                                ("Leaf Wet 1", "int32")])

        self.assertLess(len(typedDataset.columns), len(filteredDataset.columns))
        
    def test_sampleDataset(self):
        pass


if __name__ == '__main__':
    unittest.main()
