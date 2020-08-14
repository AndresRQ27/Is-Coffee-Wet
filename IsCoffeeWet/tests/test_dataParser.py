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
        filteredDataset = dataParser.cleanDataset(self.dirtyDataset)
        typedDataset = dataParser.setDataTypes(filteredDataset) #Columns reduced

        self.assertLess(newColumns, oldColumns)


if __name__ == '__main__':
    unittest.main()
