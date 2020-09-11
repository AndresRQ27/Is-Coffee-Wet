import unittest
from pandas import read_csv
from IsCoffeeWet import dataAnalysis


class Test_TestDataAnalysis(unittest.TestCase):
    def setUp(self):
        # Sets the index using Datetime column
        self.dataset = read_csv("resources\test_1H_Day_Trimester.csv",
                                engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        self.dataset = self.dataset.asfreq(self.dataset.index.inferred_freq)

    def test_graphData(self):
        dataAnalysis.graphData(self.dataset, self.dataset.columns[:2])
        self.assertTrue(True)

    def test_freqDomain(self):
        dataAnalysis.freqDomain(self.dataset, [("Temp Out"), "Leaf Wet 1"])
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
