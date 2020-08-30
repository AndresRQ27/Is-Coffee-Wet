import json


class configFile:
    """
    Object that contains the values of the configuration file in the
    JSON for the corresponding dataset. This helps to not have the file
    open more than the necessary time, while retaining its data during 
    the execution of the program.
    
    Parameters
    ----------
    - configPath : string.
        Path of the JSON that has the configuration of the dataset to
        use. Can be relative to the path where the program is running 
        or absolute.
    """
    def __init__(self, configPath):
        with open(configPath, 'r') as file:
            # Loads json file
            configFile = json.load(file)

            # Name of the Date column
            self.date = configFile["dateName"]
            # Name of the Time column
            self.time = configFile["timeName"]

            # List of the type of each column
            self.cType = []
            # List of the function for each column
            self.cFunction = []

            # Construct the list of tuples to use
            for columnName in configFile["columns"]:
                self.cType.append(
                    (columnName, configFile["columnAndType"][columnName]))
                self.cFunction.append(
                    (columnName, configFile["columnAndFunction"][columnName]))

            # List of strings to use as null
            self.null = configFile["nullList"]

            # Frequency of the dataset
            self.freq = configFile["frequency"]

            # Bool values to whether encode days/hours or not
            self.days = configFile["encodeDays"]
            self.hours = configFile["encodeHours"]
