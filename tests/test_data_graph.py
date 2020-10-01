import unittest

from pandas import read_csv

from IsCoffeeWet import data_graph
from IsCoffeeWet import config_file as cf

# Path for Linux
# PATH_TEST = "/media/andres/DATA/Code-Projects/Is-Coffee-Wet/resources/"
# Path for Windows
# PATH_TEST = "D:/VMWare/Shared/Is-Coffee-Wet/resources/"
# Path for Docker
PATH_TEST = "/opt/project/resources/"


class Test_TestDataGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column
        cls.dataset = read_csv(PATH_TEST + "test_parsed.csv",
                                engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        cls.dataset = cls.dataset.asfreq(cls.dataset.index.inferred_freq)

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
