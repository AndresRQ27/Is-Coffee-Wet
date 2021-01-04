import copy
import glob
import json
import os
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

    def __init__(self, path_config=None, parent_path=None):
        """
        Parameters
        ----------
        path_config : string, optional
            Path of the JSON that has the configuration of the dataset to
            use. Can be relative to the path where the program is running
            or absolute.
        parent_path: string, optional
            Parent path used to get to the resources folder and the
            checkpoints folder. It's assumed that this is the bound
            path.
        """
        # Initialize all the values in the config file
        self._init_values()

        # Is empty, return a generic config file
        if path_config:
            path_config = os.path.join(parent_path, path_config)
            # Opens the file
            with open(path_config, 'r') as file:
                # Loads json file
                config_file = json.load(file)

            # Look for values in the json and overwrites default
            self._overwrite_values(config_file, parent_path)

    def _init_values(self):
        """
        Initialize all the values of the ConfigFile with generic
        data.

        Parameters
        ----------
        num_data: int
            Number of data contained in th dataset. This property 
            isn't in the config file, but added later on when the
            dataset is loaded (manually)
        train_ratio: int
            Number of data to use in the training set. This property 
            isn't in the config file, but added later on when the
            dataset is split (manually)
        ds_path: string
            Read from dataset_path. Relative path of the dataset from
            the bound volume in the container. Starts from the parent
            path.
        freq: string
            Read from frequency. Frequency (steps) to sample the 
            dataset.
        forecast: int
            Read from forecast_window. Amount of days to make a 
            prediction.
        columns: list(string)
            Columns in the dataset. Loaded if a dataset_path is provided.
        datetime: list(string)
            Read from datetime. Columns from the dataset to merge into a
            single datetime column, which will be the index.
        datetime_format: string
            Read from datetime_format. Format of the datetime when all the
            columns are merge. Greatly speeds the parsing of the dates, so
            it's necessary
        null_list: list(string)
            Read from null_list. List of values to be interpreted as empty
            or null values.
        encode: dictionary
            Read from encode_time. Derivated time generated from the 
            datetime index in order to reflect the cyclical behavior
            of the days of the year and time.
        functions: dictionary
            Read from special_functions. Name of the column with the
            function to use when sampling. 
        formats: dictionary
            Read from special_format. Name of the column with the 
            special type of values other than the default float64
            that mey required special handling.
        model_name: string
            Read from model_name. Name of the model to load/save the
            weights of the neural network.
        labels: list(string)
            Read from labels. Labels of the neural network (output
            columns)
        nn_path: string
            Read from nn_path. Relative path of the folder where to
            load/save the neural network from the bound volume in the
            container. Starts from the parent path.
        submodel: string
            Read from submodel. Name of the type of model to build.
            There are 3 types: cnn, tcn and conv_lstm.
        max_epochs: int
            Read from max_epochs. Maximum number of epochs to train
            the neural network.
        lr: float
            Read from learning_rate. Learning rate of the optimizer
            for the neural network.
        patience: int
            Read from patience. Number of epochs that can pass without
            improving the accuracy of the neural network before it
            early stops
        batch_size: int
            Read from batchsize. Number of data in a batch of training.
        """
        self.num_data = 0
        self.train_ratio = 0
        self.ds_path = ""
        self.freq = "1H"
        self.forecast = 7
        self.columns = []

        # *****************************
        # *** Preprocess parameters ***
        # *****************************
        self.datetime = []
        self.datetime_format = "%d/%m/%Y %I:%M %p"
        self.null_list = []
        # Creates an empty dictionary for the column encoding
        self.encode = {}
        # Creates an empty dictionary for the special functions
        self.functions = {}
        # Creates an empty dictionary for the special format
        self.formats = {}

        # *********************************
        # *** Neural Network parameters ***
        # *********************************
        self.model_name = "generic"
        self.labels = []
        self.nn_path = "checkpoints"
        self.submodel = ""
        self.max_epochs = 1
        self.lr = 0.01
        self.patience = 5
        self.batch_size = 64

    def _overwrite_values(self, config_json, parent_path):
        """
        Function that overwrites the generic values of the
        ConfigFile with the ones specified in the config.json.
        Overwrites the general values.
        """
        if "dataset_path" in config_json:
            path = os.path.join(parent_path,
                                config_json["dataset_path"])
            # If a directory, look for the most recent file
            if os.path.isdir(path):
                # * means all if need specific format then *.csv
                list_of_files = glob.glob(config_json["dataset_path"]
                                          + "/*.csv")
                # Path of lastest DB from the provided folder
                self.ds_path = max(list_of_files, key=os.path.getctime)
            else:
                # Path of the specific dataset file to use
                self.ds_path = path

            # List of columns in the dataset
            self.columns = pd.read_csv(self.ds_path,
                                       engine="c",
                                       nrows=1).columns.tolist()

        if "frequency" in config_json:
            # Frequency of the dataset
            self.freq = config_json["frequency"]

        if "forecast_window" in config_json:
            # Amount of days to forecast
            forecast = config_json["forecast_window"]

            # Analyzes if the forecast window is bigger than the
            # frequency of the dataset. Useful to dinamically
            # create window sizes for prediction (e.g. predicting
            # the next week when the frequency is 3 would give 3
            # predictions [6 days] vs frequency of 1 would give
            # 7 predictions [one for each day])

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

        # Loads the parameters relative to the preprocessing
        if "preprocess" in config_json:
            self._preprocess(config_json=config_json["preprocess"],
                             parent_path=parent_path)

        # Loads the parameters relative to the neural network
        if "neural_network" in config_json:
            self._neural_network(config_json=config_json["neural_network"],
                                 parent_path=parent_path)

    def _preprocess(self, config_json, parent_path):
        """
        Function that overwrites the generic values of the
        ConfigFile with the ones specified in the config.json.
        Overwrites the values from the preprocess.
        """
        if "datetime" in config_json:
            # List of the columns to merge into datetime
            # (e.g. date + time)
            self.datetime = config_json["datetime"]

        if "datetime_format" in config_json:
            self.datetime_format = config_json["datetime_format"]

        if "null_list" in config_json:
            # List of strings to use as null
            self.null_list = config_json["null_list"]

        # It's expected to be given a dataset to erase columns
        if "delete_columns" in config_json and self.ds_path:
            # Read the list of columns to delete
            delete_columns = config_json["delete_columns"]

            # Removes the unwanted columns
            for column_name in delete_columns:
                try:
                    self.columns.remove(column_name)
                except ValueError:
                    warnings.warn("Some column names to remove don't exist. "
                                  "Check that the names in the Config File match the "
                                  "ones on the dataset")

        if "encode_time" in config_json:
            # Fill the dictionary
            for name, time in config_json["encode_time"].items():
                self.encode[name] = time

        if "special_functions" in config_json:
            # Fill the dictionary
            for name, functions in config_json[
                    "special_functions"].items():
                self.functions[name] = functions

        if "special_format" in config_json:
            # Fill the dictionary
            for name, formats in config_json[
                    "special_format"].items():
                self.formats[name] = formats

    def _neural_network(self, config_json, parent_path):
        """
        Function that overwrites the generic values of the
        ConfigFile with the ones specified in the config.json.
        Overwrites the values from the neural network.
        """
        if "model_name" in config_json:
            # Name to give the model
            self.model_name = config_json["model_name"]

        if "labels" in config_json:
            self.labels = config_json["labels"]

        # If no labels are provided, use the columns as labels
        self.labels = (self.labels if self.labels else
                       copy.deepcopy(self.columns))

        if "submodel" in config_json:
            self.submodel = config_json["submodel"]

        if "nn_path" in config_json:
            # Checks if the path to save the neural network exists
            self.nn_path = os.path.join(parent_path, config_json["nn_path"])
            try:
                os.makedirs(self.nn_path)
                print("Path to save the neural networks was created")
            except FileExistsError:
                print("Path to save the neural networks was found")

        # Values for the neural network compile and train
        if "max_epochs" in config_json:
            self.max_epochs = config_json["max_epochs"]
        if "learning_rate" in config_json:
            self.lr = config_json["learning_rate"]
        if "patience" in config_json:
            self.patience = config_json["patience"]
        if "batch_size" in config_json:
            self.batch_size = config_json["batch_size"]
