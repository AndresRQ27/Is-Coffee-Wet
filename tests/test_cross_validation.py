import os
import unittest
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
        dataset_convolutional = pd.read_csv(PATH + "/benchmark/results/benchmark_convolutional.csv",
                                            engine="c", header=0, names=column_names)

        # Drops the previous index
        cls.dataset_convolutional = dataset_convolutional.drop("Number", axis=1)

        try:
            os.mkdir(PATH + "/images/cross-validation")
        except FileExistsError:
            print("{} already exists".format(PATH + "/images/cross-validation"))

    def test_benchmark_graph_all(self):
        cv.benchmark_graph_all(self.dataset_convolutional,
                               PATH + "/images/cross-validation",
                               "convolutional",
                               100)
        self.assertTrue(True)

    def test_benchmark_graph_summary(self):
        cv.benchmark_graph_summary(self.dataset_convolutional,
                                   PATH + "/images/cross-validation",
                                   "convolutional")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
