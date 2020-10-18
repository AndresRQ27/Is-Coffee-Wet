import unittest

import pandas as pd

from benchmarks import cross_validation as cv

# Path for Docker
PATH_TEST = "/opt/project/resources/benchmark/results/"


class Test_CrossValidation(unittest.TestCase):
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
        cv.benchmark_graph_all(self.dataset, 30)
        self.assertTrue(True)

    def test_benchmark_graph_summary(self):
        cv.benchmark_graph_summary(self.dataset)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
