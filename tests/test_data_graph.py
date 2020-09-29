import unittest
from pandas import read_csv
from IsCoffeeWet import data_graph
from IsCoffeeWet import config_file as cf


class Test_TestDataGraph(unittest.TestCase):
    def setUp(self):
        # Sets the index using Datetime column
        self.dataset = read_csv("resources/test_parsed.csv",
                                engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        self.dataset = self.dataset.asfreq(self.dataset.index.inferred_freq)

    def test_graph_data(self):
        config_file = cf.ConfigFile()
        config_file.columns = ["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]
        
        data_graph.graph_data(self.dataset, config_file)
        self.assertTrue(True)

    def test_freq_domain(self):
        config_file = cf.ConfigFile()
        config_file.columns = ["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]

        data_graph.freq_domain(self.dataset, config_file)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
