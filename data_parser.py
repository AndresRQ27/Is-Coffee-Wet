import pandas as pd
import numpy as np

#Checks pandas version
print("Pandas version:", pd.__version__)

#Loads the dataset
dataset = pd.read_csv("resources/est0Corta.csv")

#TEST: prints
#print(dataset.iloc[4992])
#print(dataset.filter(items=["---"], axis=1))
print(dataset.info(verbose=True))
old = dataset.size

#*** Manages missing values
#TODO: Removes the columns with almost all missing values
#! dataset = dataset.drop(['column1', 'column2'], axis=1)
#Sets al the missing values (represented as "---") to NaN in the selected columns
dataset = dataset.replace({"Temp Out": "---", 
                           "Leaf Wet 1": "---"}, 
                           np.NaN)
dataset = dataset.dropna(axis = 0, how = "any") #Removes the NaN values in the rows

#*** Parse the dtypes of the columns
#Unifies "Date" and "Time" into a "Date" as pandas uses "datetime64[ns]"
dataset['Date'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'])
#Changes the data to types that use less memory
dataset = dataset.astype({"Temp Out": "float32", 
                          "Leaf Wet 1": "int32"}) 
#Drops the "Time" columns as it has been combine into "Date"
dataset = dataset.drop("Time", axis=1)

#TEST: prints
#print(dataset.iloc[4992])
print(dataset.info(verbose=True))
new = dataset.size
print("Eliminated data:", old-new)