import pandas as pd
import numpy as np

def cleanDataset(dataset):
  """Removes missing or empty values that can 
  generate error in the parse of the dataset

  Args:
      dataset (pd.DataFrame): Dataset to clean

  Returns:
      pd.DataFrame: Cleaned dataset
  """  
  #Removes the columns with almost all missing values
  #TODO: when de data set is complete
  #! dataset = dataset.drop(['column1', 'column2'], axis=1)
  
  #Sets al the missing values (represented as "---") to NaN in the selected columns
  dataset = dataset.replace({"Temp Out": "---", #Columns where missing data could be
                             #?"columnName"": "---",
                             "Leaf Wet 1": "---"}, 
                             np.NaN) #New value
  dataset = dataset.dropna(axis = 0, #Drop only the rows
                           how = "any") #Removes the NaN values in the rows
  return dataset

def setDataTypes(dataset):
  """Reorders the dataset by merging Date & Time into a pd.datetime64[ns]
  Sets the types of the rest of the columns

  Args:
      dataset (pd.DataFrame): Dataset with the columns to change the type

  Returns:
      pd.DataFrame: dataset with all the columns set to a known type 
                    (not just "object")
  """  
  #Unifies "Date" and "Time" into a "Date" as pandas uses "datetime64[ns]"
  dataset['Date'] = pd.to_datetime(dataset['Date'] + ' ' + dataset['Time'])
  #Changes the data to types that use less memory
  dataset = dataset.astype({"Temp Out": "float32", 
                            #?"columnName"": "---",
                            "Leaf Wet 1": "int32"}) 
  #Drops the "Time" columns as it has been combine into "Date"
  dataset = dataset.drop("Time", axis=1)

  return dataset

def unifyTime(dataset):
  #***Group the rows in sequences of every 15 minutes
  #Generates an array of all the hours in a day
  hour_array = pd.date_range('2010-01-01', periods=24, freq='1H')
  minute_array = pd.date_range('2010-01-01', periods=5, freq='15min')

#TEST: prints
#print(dataset.info(verbose=True))