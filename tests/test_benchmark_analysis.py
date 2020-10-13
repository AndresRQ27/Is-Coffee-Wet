import unittest

import pandas as pd

from benchmarks import benchmark_analysis as ba

# Path for Docker
PATH_TEST = "/opt/project/resources/benchmark/"


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Name of the columns in the dataset
        column_names = ["Number", "Loss", "MAE", "MAPE", "Val_Loss",
                        "Val_MAE", "Val_MAPE", "Name"]

        # Loads the dataset
        dataset = pd.read_csv(PATH_TEST + "benchmark_convolutional.csv",
                              engine="c", header=0, names=column_names)

        # Drops the previous index
        cls.dataset = dataset.drop("Number", axis=1)

    def test_benchmark_graph_all(self):
        ba.benchmark_graph_all(self.dataset)
        self.assertTrue(True)

    def test_benchmark_graph_minmax(self):
        ba.benchmark_graph_minmax(self.dataset)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
