import matplotlib as plt
import pandas as pd
import numpy as np
from IsCoffeeWet import dataParser as dp

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")
dataPath = input("- Please type the path to your dataset: ")

dataset = pd.read_csv(dataPath, engine="c")
print("Dataset loaded... \n")

isDataParsed = input(
    "- Has your dataset been previously parsed?: (Yes/No) ")

if isDataParsed == "No":
    print("\n***Dataset parse started***")

    date = input("- What's the column name with the dates?: ")
    time = input("- What's the column name with the hours?: ")
    dataset = dp.mergeDateTime(dataset, date, time)
    print("Date & Time merge completed...\n")

    nullValue = input("- How your dataset manage null values?: ")

    print("- Write the column name and the data type for it: ")
    print("\t+ Separate using a coma (,)")
    print("\t+ Data types available: signed, unsigned, float")
    print("\t+ When finished, leave empty and press enter")

    columnAndType = []
    # Stops the function when not enough inputs are received
    while True:
        try:
            inputA, inputB = input("Column-name,data-type: ").split(",")
            columnAndType.append((inputA, inputB))
        except ValueError:
            break

    dataset = dp.convertNumeric(dataset, columnAndType, nullValue)
    print("Conversion of columns to numbers completed...")

    print("- What's the frequency to use for the dataset?")
    print("Examples of correct inputs:")
    print("\t15min")
    print("\t1H (hour)")
    print("\t1h30min")
    print("\t1D (day)")
    print("\t1M (month)")
    frequency = input("freq: ")

    print("- Write the column name and the function to apply during resampling: ")
    print("\t+ Separate using a coma (,)")
    print("\t+ Some available functions: mean, last, max, etc.")
    print("\t+ When finished, leave empty and press enter")

    columnAndFunction = []
    # Stops the function when not enough inputs are received
    while True:
        try:
            inputA, inputB = input("Column-name,function: ").split(",")
            columnAndFunction.append((inputA, inputB))
        except ValueError:
            break

    dataset = dp.sampleDataset(dataset, columnAndFunction, frequency)
    print("Resampling completed...")

    #Do again this function in case a sampling operation has turne
    #integers in floats
    dataset = dp.convertNumeric(dataset, columnAndType, nullValue)

    encodeDays = input(
        "- Do you want to encode days into a pair of sin/cos? (Yes/No) ")
    encodeHours = input(
        "- Do you want to encode hours into a pair of sin/cos? (Yes/No) ")

    dataset = dp.cyclicalEncoder(
        dataset, encodeDays == "Yes", encodeHours == "Yes")
    print("Encoding complete completed...")

    if encodeDays == "Yes":
        encodeDays = "_encodedDays"
    else:
        encodeDays = ""

    if encodeHours == "Yes":
        encodeHours = "_encodedHours"
    else:
        encodeHours = ""

    dataPath = dataPath.replace(".csv", "")
    dataName = dataPath + "_" + frequency + encodeDays + encodeHours + ".csv"
    dataset.to_csv(dataName)
    print("A copy of your dataset has been save into: " + dataName)
