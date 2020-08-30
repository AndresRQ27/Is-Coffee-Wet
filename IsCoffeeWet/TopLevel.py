from pandas import read_csv
from IsCoffeeWet import (dataParser as dp,  
                        configFile as cf, 
                        dataAnalysis as da)

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")

dataPath = input("- Please type the path to your dataset: ")

configPath = input("- Path to your config file (JSON): ")

isDataParsed = input("- Has your dataset been previously parsed? (yes/no): ")

# Dataset configuration extracted from the JSON
dsConfig = cf.configFile(configPath)

# Initialize the dataset
if isDataParsed == "no":
    # Loads the dataset
    dataset = read_csv(dataPath, engine="c")

    # Remove trailing and leading spaces from column names
    dataset.columns = dataset.columns.str.strip()

    # Parse the dataset
    # convertNumeric twice to fill empty values after sampling
    dataset = (dataset.pipe(dp.mergeDateTime, dateName=dsConfig.date, timeName=dsConfig.time)
                      .pipe(dp.convertNumeric, columnAndType=dsConfig.cType, nullList=dsConfig.null)
                      .pipe(dp.sampleDataset, columnAndFunction=dsConfig.cFunction, frequency=dsConfig.freq)
                      .pipe(dp.convertNumeric, columnAndType=dsConfig.cType, nullList=dsConfig.null)
                      .pipe(dp.cyclicalEncoder, encodeDays=dsConfig.days, encodeHours=dsConfig.hours)
               )

    # Value used for filename of new dataset
    if dsConfig.days:
        printDays = "_encodedDays"
    else:
        printDays = ""

    # Value used for filename of new dataset
    if dsConfig.time:
        printHours = "_encodedHours"
    else:
        printHours = ""

    dataPath = dataPath.replace(".csv", "")
    dataPath = dataPath + "_" + dsConfig.freq + printDays + printHours + ".csv"

    # Saves the parsed dataset
    dataset.to_csv(dataPath)
    print("A copy of your dataset has been save into: " + dataPath)

else:
    # Sets the index using Datetime column
    dataset = read_csv(dataPath, engine="c",
                          index_col="Datetime", parse_dates=True)
    # Infers the frequency
    dataset = dataset.asfreq(dsConfig.freq)

print("Do a graphical analysis of the dataset?")
print("(If yes, NN training must be done in another execution)")
graphData = input("- (yes/no): ")

# Execution ends with the graph as it requires a lot of memory
# to have the graphs with the NN training
if graphData == "yes":    
    #TODO: change to False when more code is written
    da.graphData(dataset, dsConfig.graphColumns)
    exit()