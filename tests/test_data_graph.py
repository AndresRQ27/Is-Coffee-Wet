import os
import unittest
import pandas as pd

from IsCoffeeWet.tools.config_file import ConfigFile
from IsCoffeeWet.preprocess import data_graph

PATH = os.getcwd() + "/resources/tests"


class Test_TestDataGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column
        dataset = pd.read_csv(PATH + "/database/test_parsed.csv",
                              engine="c", index_col=0, parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)

        cls.config_file = cf.ConfigFile()
        cls.config_file.columns = [
            "Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]

    def test_graph_data(self):
        data_graph.graph_data(
            self.dataset, self.config_file, PATH + "/images")
        self.assertTrue(True)

    def test_freq_domain(self):
        data_graph.freq_domain(
            self.dataset, self.config_file, PATH + "/images")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
