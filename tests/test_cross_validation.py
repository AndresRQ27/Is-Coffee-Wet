import unittest

import os
import pandas as pd

from benchmarks import cross_validation as cv

PATH = os.getcwd() + "/resources"


class Test_CrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Name of the columns in the dataset
        column_names = ["Number", "Loss", "MAE", "MAPE", "Val_Loss",
                        "Val_MAE", "Val_MAPE", "Name"]

        # Loads the dataset
        dataset = pd.read_csv(PATH + "/benchmark/results/benchmark_convolutional.csv",
                              engine="c", header=0, names=column_names)

        # Drops the previous index
        cls.dataset = dataset.drop("Number", axis=1)

    def test_benchmark_graph_all(self):
        cv.benchmark_graph_all(self.dataset, PATH + "/images", 100)
        self.assertTrue(True)

    def test_benchmark_graph_summary(self):
        cv.benchmark_graph_summary(self.dataset, PATH + "/images")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
