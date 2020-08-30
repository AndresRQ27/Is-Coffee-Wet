import matplotlib as plt
import pandas as pd
import numpy as np
import json
from IsCoffeeWet import dataParser as dp

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")
dataPath = input("- Please type the path to your dataset: ")

isDataParsed = input(
    "- Has your dataset been previously parsed? (Yes/No): ")

if isDataParsed == "No":
    dataset = pd.read_csv(dataPath, engine="c")

    # Remove trailing and leading spaces from column names
    dataset.columns = dataset.columns.str.strip()

    configPath = input("- Path to your config file (JSON): ")
    with open(configPath, 'r') as file:
        
        # Loads json file
        configFile = json.load(file)

        # Merge Date and Time into a single column
        dataset = dp.mergeDateTime(
            dataset, configFile["dateName"], configFile["timeName"])

        # List to construct
        columnAndType = []
        columnAndFunction = []

        # Construct the list of tuples to use
        for columnName in configFile["columns"]:
            columnAndType.append(
                (columnName, configFile["columnAndType"][columnName]))
            columnAndFunction.append(
                (columnName, configFile["columnAndFunction"][columnName]))

        # Cleans the dataset and infers the type of the columns
        dataset = dp.convertNumeric(
            dataset, columnAndType, configFile["nullList"])

        # Samples the dataset and removes the unused columns
        dataset = dp.sampleDataset(
            dataset, columnAndFunction, configFile["frequency"])

        # Fill empty values after sampling
        dataset = dp.convertNumeric(
            dataset, columnAndType, configFile["nullList"])

        # Encodes tha day/hours for the NN
        dataset = dp.cyclicalEncoder(
            dataset, configFile["encodeDays"], configFile["encodeHours"])

        # Value used for filename of new dataset
        if configFile["encodeDays"]:
            encodeDays = "_encodedDays"
        else:
            encodeDays = ""

        # Value used for filename of new dataset
        if configFile["encodeHours"]:
            encodeHours = "_encodedHours"
        else:
            encodeHours = ""

        dataPath = dataPath.replace(".csv", "")
        dataPath = dataPath + "_" + \
            configFile["frequency"] + encodeDays + encodeHours + ".csv"
        
        dataset.to_csv(dataPath)
        print("A copy of your dataset has been save into: " + dataPath)

else:
    # Sets the index using Datetime column
    dataset = pd.read_csv(dataPath, engine="c",
                          index_col="Datetime", parse_dates=True)
    # Infers the frequency
    dataset = dataset.asfreq(dataset.index.inferred_freq)
    pass