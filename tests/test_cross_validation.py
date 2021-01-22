from benchmark import cross_validation as cv
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "benchmark"))


PATH_RESOURCES = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "resources")


class Test_CrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.performance_ds,
         cls.min_max) = performance_analysis()

        (cls.labels,
         cls.prediction_group,
         cls.prediction_indiv) = prediction_analysis()

    def test_convolutional_ds(self):
        test_name = "convolutional"

        # Makes a graph of all the values from the convolutional benchmarks
        cv.graph_epochs(self.performance_ds[test_name],
                        PATH_RESOURCES + "/benchmark/images/performance",
                        test_name,
                        100)
        # Makes a bar graph from the convolutional benchmarks
        cv.graph_performance(self.min_max[test_name],
                             self.performance_ds[test_name].columns,
                             PATH_RESOURCES + "/benchmark/images/performance",
                             test_name)
        self.assertTrue(True)

    def test_temporal_ds(self):
        test_name = "temporal"

        # Makes a graph of all the values from the temporal benchmarks
        cv.graph_epochs(self.performance_ds[test_name],
                        PATH_RESOURCES + "/benchmark/images/performance",
                        test_name,
                        100)
        # Makes a bar graph from the temporal benchmarks
        cv.graph_performance(self.min_max[test_name],
                             self.performance_ds[test_name].columns,
                             PATH_RESOURCES + "/benchmark/images/performance",
                             test_name)
        self.assertTrue(True)

    def test_conv_lstm_ds(self):
        test_name = "conv_lstm"

        # Makes a graph of all the values from the conv_lstm benchmarks
        cv.graph_epochs(self.performance_ds[test_name],
                        PATH_RESOURCES + "/benchmark/images/performance",
                        test_name,
                        100)
        # Makes a bar graph from the conv_lstm benchmarks
        cv.graph_performance(self.min_max[test_name],
                             self.performance_ds[test_name].columns,
                             PATH_RESOURCES + "/benchmark/images/performance",
                             test_name)
        self.assertTrue(True)

    def test_individual_conv_ds(self):
        test_name = "individual_conv"

        # Makes a graph of all the values from the individual_conv benchmarks
        cv.graph_epochs(self.performance_ds[test_name],
                        PATH_RESOURCES + "/benchmark/images/performance",
                        test_name,
                        100)
        # Makes a bar graph from the individual_conv benchmarks
        cv.graph_performance(self.min_max[test_name],
                             self.performance_ds[test_name].columns,
                             PATH_RESOURCES + "/benchmark/images/performance",
                             test_name)
        self.assertTrue(True)

    def test_individual_temp_ds(self):
        test_name = "individual_temp"

        # Makes a graph of all the values from the individual_temp benchmarks
        cv.graph_epochs(self.performance_ds[test_name],
                        PATH_RESOURCES + "/benchmark/images/performance",
                        test_name,
                        100)
        # Makes a bar graph from the individual_temp benchmarks
        cv.graph_performance(self.min_max[test_name],
                             self.performance_ds[test_name].columns,
                             PATH_RESOURCES + "/benchmark/images/performance",
                             test_name)
        self.assertTrue(True)

    def test_prediction_conv(self):
        test_name = "convolutional"

        # Group the label with the datasets of the convolutional predictions
        datasets = {"labels": self.labels,
                    "group": self.prediction_group[test_name],
                    "individual": self.prediction_indiv[test_name]}

        cv.graph_predictions(datasets,
                             self.labels.columns,
                             PATH_RESOURCES + "/benchmark/images/predictions",
                             "prediction_" + test_name)
        self.assertTrue(True)

    def test_prediction_temp(self):
        test_name = "temporal"

        # Group the label with the datasets of the temporal predictions
        datasets = {"labels": self.labels,
                    "group": self.prediction_group[test_name],
                    "individual": self.prediction_indiv[test_name]}

        cv.graph_predictions(datasets,
                             self.labels.columns,
                             PATH_RESOURCES + "/benchmark/images/predictions",
                             "prediction_" + test_name)
        self.assertTrue(True)


def performance_analysis():
    """
    Function that loads and parses the datasets of the benchmarks done for
    each neural network tested. These are later graphed for easier analysis.

    Returns
    -------
    (list, dict, dict)
        Returns a list of the column names in the datasets, a dictionary of
        the datasets with the results of the performance from the benchmark
        and another dictionary with the important values extracted for a
        bar chart.

    """
    # Name of the columns of the datasets
    column_names = ["Number", "Loss", "MAE", "MAPE", "Val_Loss",
                    "Val_MAE", "Val_MAPE", "Name"]

    # Datasets with the performance metrics for the benchmark tests
    convolutional = pd.read_csv(PATH_RESOURCES
                                + "/benchmark/performance/benchmark_convolutional.csv",
                                engine="c", header=0, names=column_names)
    temporal = pd.read_csv(PATH_RESOURCES +
                           "/benchmark/performance/benchmark_temporal.csv",
                           engine="c", header=0, names=column_names)
    conv_lstm = pd.read_csv(PATH_RESOURCES +
                            "/benchmark/performance/benchmark_conv_lstm.csv",
                            engine="c", header=0, names=column_names)
    individual_conv = pd.read_csv(PATH_RESOURCES
                                  + "/benchmark/performance/benchmark_prediction_convolutional.csv",
                                  engine="c", header=0, names=column_names)
    individual_temp = pd.read_csv(PATH_RESOURCES
                                  + "/benchmark/performance/benchmark_prediction_temporal.csv",
                                  engine="c", header=0, names=column_names)

    # Group all the datasets in a dictionary
    datasets = {"convolutional": convolutional,
                "temporal": temporal,
                "conv_lstm": conv_lstm,
                "individual_conv": individual_conv,
                "individual_temp": individual_temp}

    # Dictionary of the datasets parsed to contain the values used to graph
    min_max = {}

    # Pre-process each DataFrames into a list of DataFrames of values to
    # graph. Analyses the model metrics and take the important values
    for key, value in datasets.items():
        # Drops the previous index
        value.drop("Number", axis=1, inplace=True)

        # Retrieves the unique test names in the dataset
        # Assign an id to a test name. Use for colors and markers
        names = [name for name in value["Name"].drop_duplicates()]

        # Obtains the list of columns to graph with numeric value
        # Removes the last columns ("Name")
        columns = value.columns.tolist()[:-1]

        # Initialize the values in the dictionary
        min_max[key] = {}

        # DataFrames are in order: 1-min, 2-max, 3-avg, 4-last
        min_max[key]["min"] = pd.DataFrame()
        min_max[key]["max"] = pd.DataFrame()
        min_max[key]["avg"] = pd.DataFrame()
        min_max[key]["last"] = pd.DataFrame()

        # Obtains the values to plot from each test
        # There is a DataFrame for each measurement
        for name in names:
            # Gets the data from a sub-test
            data = value.loc[value["Name"] == name]

            # Checks if a single value has NaN
            if data.isna().any().any():
                data = data.dropna(axis=0)
                # If the data was all NaNs, don't add ir
                if data.empty:
                    continue

            # 1. Obtains the operation only from the numeric columns (that's why se use "columns")
            # The columns generated by the operation will only be the ones in "columns" and is a pd.Series
            # 2. Adds the name of the sub-test in the Series as a new row. Must be added as as pd.Series
            # 3. Converts the pd.Series with the name into a pd.DataFrame
            # 4. Transpose the Frame and append to the rest of the DataFrame's data
            min_max[key]["min"] = min_max[key]["min"].append(data[columns]
                                                             .min()
                                                             .append(pd.Series({"Name": name}))
                                                             .to_frame()
                                                             .transpose(), ignore_index=True)
            min_max[key]["avg"] = min_max[key]["avg"].append(data[columns]
                                                             .mean()
                                                             .append(pd.Series({"Name": name}))
                                                             .to_frame()
                                                             .transpose(), ignore_index=True)
            min_max[key]["max"] = min_max[key]["max"].append(data[columns]
                                                             .max()
                                                             .append(pd.Series({"Name": name}))
                                                             .to_frame()
                                                             .transpose(), ignore_index=True)
            min_max[key]["last"] = min_max[key]["last"].append(data[columns]
                                                               .iloc[-1]
                                                               .append(pd.Series({"Name": name}))
                                                               .to_frame()
                                                               .transpose(), ignore_index=True)

        return datasets, min_max

    if __name__ == '__main__':
        unittest.main()


def prediction_analysis():
    # Dictionary of the datasets with the predictions from the benchmarks
    labels = pd.read_csv(PATH_RESOURCES
                         + "/benchmark/database/dataset_hour.csv",
                         engine="c", index_col="Datetime", parse_dates=True)
    convolutional = pd.read_csv(PATH_RESOURCES
                                + "/benchmark/prediction/prediction_convolutional.csv",
                                engine="c", index_col=0)
    temporal = pd.read_csv(PATH_RESOURCES
                           + "/benchmark/prediction/prediction_temporal.csv",
                           engine="c", index_col=0)

    # Stores only the last 168 labels as the ground truth
    labels.drop(["day sin", "day cos", "year sin", "year cos"],
                axis=1, inplace=True)
    labels = labels.tail(168)
    labels.reset_index(drop=True, inplace=True)

    # Dictionary with the datasets of the predictions
    datasets = {"convolutional": convolutional,
                "temporal": temporal}

    # Dictionaries with the datasets divided into group or individual
    group = {}
    individual = {}

    # Divide predictions into a group and individual prediction
    # The group predictions will always be the first 168 rows
    for key, value in datasets.items():
        group[key] = value.head(168)
        individual[key] = value.tail(-168)

    return labels, group, individual


if __name__ == '__main__':
    unittest.main()
