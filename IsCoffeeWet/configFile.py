import json


class configFile:

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
