import unittest
from pandas import read_csv
from IsCoffeeWet import dataParser


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("../../resources/est0Corta.csv")

    def test_cleanDataset(self):
        filteredDataset = dataParser.cleanDataset(self.dirtyDataset)
        self.assertLess(filteredDataset.size, self.dirtyDataset.size)


if __name__ == '__main__':
    unittest.main()
