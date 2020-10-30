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
        dataset_temporal = pd.read_csv(PATH + "/benchmark/results/benchmark_temporal.csv",
                                       engine="c", header=0, names=column_names)
        labels_convolutional = pd.read_csv(PATH + "/benchmark/results/benchmark_labels_convolutional.csv",
                                           engine="c", header=0, names=column_names)
        labels_temporal = pd.read_csv(PATH + "/benchmark/results/benchmark_labels_temporal.csv",
                                      engine="c", header=0, names=column_names)

        # Drops the previous index
        cls.dataset_convolutional = dataset_convolutional.drop("Number", axis=1)
        cls.dataset_temporal = dataset_temporal.drop("Number", axis=1)
        cls.labels_convolutional = labels_convolutional.drop("Number", axis=1)
        cls.labels_temporal = labels_temporal.drop("Number", axis=1)

        try:
            os.mkdir(PATH + "/images/cross-validation")
        except FileExistsError:
            print("{} already exists".format(PATH + "/images/cross-validation"))

    def test_dataset_convolutional(self):
        # Makes a graph of all the values from the convolutional benchmarks
        cv.benchmark_graph_all(self.dataset_convolutional,
                               PATH + "/images/cross-validation",
                               "convolutional",
                               100)
        # Makes a bar graph from the convolutional benchmarks
        cv.benchmark_graph_summary(self.dataset_convolutional,
                                   PATH + "/images/cross-validation",
                                   "convolutional")
        self.assertTrue(True)

    def test_dataset_temporal(self):
        # Makes a graph of all the values from the temporal benchmarks
        cv.benchmark_graph_all(self.dataset_temporal,
                               PATH + "/images/cross-validation",
                               "temporal",
                               100)
        # Makes a bar graph from the temporal benchmarks
        cv.benchmark_graph_summary(self.dataset_temporal,
                                   PATH + "/images/cross-validation",
                                   "temporal")
        self.assertTrue(True)

    def test_dataset_labels_convolutional(self):
        # Makes a graph of all the values from the labels convolutional benchmarks
        cv.benchmark_graph_all(self.labels_convolutional,
                               PATH + "/images/cross-validation",
                               "labels_convolutional",
                               100)
        # Makes a bar graph from the labels convolutional benchmarks
        cv.benchmark_graph_summary(self.labels_convolutional,
                                   PATH + "/images/cross-validation",
                                   "labels_convolutional")
        self.assertTrue(True)

    def test_dataset_labels_temporal(self):
        # Makes a graph of all the values from the labels temporal benchmarks
        cv.benchmark_graph_all(self.labels_temporal,
                               PATH + "/images/cross-validation",
                               "labels_temporal",
                               100)
        # Makes a bar graph from the labels temporal benchmarks
        cv.benchmark_graph_summary(self.labels_temporal,
                                   PATH + "/images/cross-validation",
                                   "labels_temporal")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
