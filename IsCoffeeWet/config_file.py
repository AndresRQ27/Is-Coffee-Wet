import json

from pandas import read_csv


class ConfigFile:
    """
    Object that contains the values of the configuration file in the JSON
    for the corresponding dataset. This helps to not have the file open
    more than the necessary time, while retaining its data during the
    execution of the program.

    Parameters
    ----------
    - config_path : string. Path of the JSON that has the configuration of
        the dataset to use. Can be relative to the path where the program
        is running or absolute.
    """

    def __init__(self, config_path=None):
        # When initialize without arguments
        if config_path is None:
            return

        # When initialize with a path
        else:
            with open(config_path, 'r') as file:
                # Loads json file
                config_file = json.load(file)

                # TODO: use a function to receive a folder path
                # and return the most recent file in it

                # Path of the file dataset to use
                self.path = config_file["dataset_path"]

                # Frequency of the dataset
                self.freq = config_file["frequency"]

                # Amount of days to forecast
                self.forecast = config_file["forecast_window"]

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
                self.columns = read_csv(self.path,
                                        engine="c",
                                        nrows=1).columns.tolist()

                # Removes the unwanted columns
                for column_name in delete_columns:
                    self.columns.remove(column_name)

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
