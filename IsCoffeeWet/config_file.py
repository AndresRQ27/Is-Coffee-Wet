import json
import warnings

import numpy as np
import pandas as pd


class ConfigFile:
    """
    Object that contains the values of the configuration file in the JSON
    for the corresponding dataset. This helps to not have the file open
    more than the necessary time, while retaining its data during the
    execution of the program.
    """

    def __init__(self, path_config=None, path_resources=None):
        """
        Parameters
        ----------
        path_config : string, optional
            Path of the JSON that has the configuration of the dataset to
            use. Can be relative to the path where the program is running
            or absolute.
        path_resources: string, optional
            Path to the resources folder where all the persistent files 
            are stored.
        """
        # Number of rows (data) in the dataset. Not found in the
        # configuration file
        self.num_data = 0

        # When initialize without arguments
        if path_config is None:
            return

        # When initialize with a path
        else:
            with open(path_config, 'r') as file:
                # Loads json file
                config_file = json.load(file)

                # TODO: use a function to receive a folder path
                # and return the most recent file in it

                # Path of the file dataset to use
                self.path = config_file["dataset_path"]

                # Frequency of the dataset
                self.freq = config_file["frequency"]

                # Amount of days to forecast
                forecast = config_file["forecast_window"]

                # Analyzes if the forecast window is bigger than the
                # frequency of the dataset

                # Transforms the frequency from string to Timedelta
                delta_freq = pd.Timedelta(self.freq)
                # Threshold that represent the amount of days in int
                threshold = delta_freq.days + delta_freq.seconds / 86400

                if threshold >= forecast:
                    # If the frequency of the dataset is bigger than the forecast window, NN won't work
                    raise ValueError(
                        "Frequency can't be bigger than forecast window")
                else:
                    # Always round down. Forecast will always be in hours IN THE CODE!!!!
                    self.forecast = int(np.floor(forecast / threshold))

                # Boolean to graph the col
                self.graph = config_file["graph_data"]

                # *****************************
                # *** Preprocess parameters ***
                # *****************************

                # List of the columns to merge into datetime
                self.datetime = config_file["pre-process"]["datetime"]

                self.datetime_format = config_file["pre-process"]["datetime_format"]

                # List of strings to use as null
                self.null = config_file["pre-process"]["null_list"]

                # Read the list of columns to delete
                delete_columns = config_file["pre-process"]["delete_columns"]

                # List of columns to use
                self.columns = pd.read_csv(path_resources + self.path,
                                           engine="c",
                                           nrows=1).columns.tolist()

                # Removes the unwanted columns
                for column_name in delete_columns:
                    try:
                        self.columns.remove(column_name)
                    except ValueError:
                        warnings.warn("Some column names to remove don't exist. "
                                      "Check that the names in the Config File match the "
                                      "ones on the dataset")

                # Creates an empty dictionary for the column encoding
                self.encode = {}
                # Fill the dictionary
                for name, time in config_file["pre-process"]["encode_time"].items():
                    self.encode[name] = time

                # Creates an empty dictionary for the special functions
                self.functions = {}
                # Fill the dictionary
                for name, functions in config_file["pre-process"]["special_functions"].items():
                    self.functions[name] = functions

                # Creates an empty dictionary for the special format
                self.formats = {}
                # Fill the dictionary
                for name, formats in config_file["pre-process"]["special_format"].items():
                    self.formats[name] = formats

                # *********************************
                # *** Neural Network parameters ***
                # *********************************

                self.training = config_file["neural_network"]["training_set"]
                self.validation = config_file["neural_network"]["validation_set"]

                labels = config_file["neural_network"]["labels"]

                # If no labels are provided, use the columns as labels
                self.labels = labels if labels else self.columns
