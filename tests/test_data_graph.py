import unittest

from pandas import read_csv

from IsCoffeeWet import config_file as cf
from IsCoffeeWet import data_graph

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
        dataset = read_csv(PATH_TEST + "test_parsed.csv",
                           engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        cls.dataset = dataset.asfreq(dataset.index.inferred_freq)

        cls.config_file = cf.ConfigFile()
        cls.config_file.columns = ["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]

    def test_graph_data(self):
        data_graph.graph_data(self.dataset, self.config_file)
        self.assertTrue(True)

    def test_freq_domain(self):
        data_graph.freq_domain(self.dataset, self.config_file)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
