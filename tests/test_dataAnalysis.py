import unittest
from pandas import read_csv
from IsCoffeeWet import dataAnalysis


class Test_TestDataAnalysis(unittest.TestCase):
    def setUp(self):
        # Sets the index using Datetime column
        self.dataset = read_csv("resources/test_1H_encodedDays_encodedHours.csv",
                                engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        self.dataset = self.dataset.asfreq(self.dataset.index.inferred_freq)

    def test_graphData(self):
        dataAnalysis.graphData(self.dataset, self.dataset.columns[:2])
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
