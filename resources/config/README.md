__Table of contents__

- [JSON Format](#json-format)
  - [General values](#general-values)
    - [`dataset_path`](#dataset_path)
    - [`frequency`](#frequency)
    - [`forecast_window`](#forecast_window)
    - [`graph_data`](#graph_data)
  - [pre-process](#pre-process)
    - [`datetime`](#datetime)
    - [`datetime_format`](#datetime_format)
    - [`null_list`](#null_list)
    - [`encode_time`](#encode_time)
    - [`delete_columns`](#delete_columns)
    - [`special_functions`](#special_functions)
    - [`special_format`](#special_format)
  - [neural_network](#neural_network)
    - [`training_set`](#training_set)
    - [`validation_set`](#validation_set)
    - [test_set](#test_set)
    - [`path`](#path)
    - [`labels`](#labels)

# JSON Format

## General values

### `dataset_path`

(_string_) path of the dataset file to use.

  - Relative path to the resources folder.

### `frequency`

(_string_) Frequency to resample the dataset. It also includes the minimal
timestamp to use when predicting the values.

- Possible values are:
  - `15min` for 15 minutes
  - `1h` for 1 hour
  - `1h30min` for every hour and a half
  - `1d` for 1 day
  - `1m` for 1 month


### `forecast_window`

(_int_) Amount of time to forecast. Its unit are days.

### `graph_data`

(_boolean_). Indicates if it should show the graph of the data used.

- Recommended in the first run to view the behavior of the data and find
  strange values.
- Otherwise, recommended off (`False`)


## pre-process


### `datetime`

List (_string_) of column names to merge into a datetime index.

- With the current format:
  - The column "Date" has the date in DD/MM/YYYY
  - The column "Time" has the time in HH:MM AM/PM
- Both columns are string at first, merge together and converted into
  datetime objects

### `datetime_format`

(_string_) Format the datetime will have after merging all the column.

### `null_list`

List (_any_) of possible parameters to represent a bad value. They are
change into NaN (Not a Number) and later interpolated.

### `encode_time`

Dictionary (_string: string_) of key-value with __extra__ features to add
into the dataset. Time __must__ be in seconds.
  
- This is the result of analyzing the trends in the data by graphing the
  Fast-Fourier Transform and choosing the higher values.
- The idea is that the higher the value, the more representative is the
  feature for the dataset, making easier the NN training.

### `delete_columns`

List (_string_) Columns __NOT__ to use for the forecasting. They will be
eliminated during resampling.

- The names of the columns to delete must be exactly as they appear in the
  dataset, with white trail spaces and everything.
- It's' imperative to always delete the columns that use text (e.g. "Wind
  Direction")

### `special_functions`

Dictionary (_string: string_). In general, all the columns are resample
using the mean function. If a different function must be applied to a
specific column, it must be specified here (e.g. `"Rain": "sum"` as we want
the total amount of rain in a given interval)

- The names of the columns must be exactly as they appear in the
  dataset, with white trail spaces and everything.

### `special_format`
Dictionary (_string: string_). In general, all the numeric values are
float. Certain column may want to stay as integers (no decimals) after the
resampling process, so specify the columns and the format here (e.g. `"Leaf
Wet 1": "int"`). 

- The names of the columns must be exactly as they appear in the dataset,
  with white trail spaces and everything.


## neural_network


### `training_set`

(_float_) Percentage in decimals of the data to use in the neural network
training. This set will be used for traning.

### `validation_set`

(_float_) Percentage in decimals of the data to use in the neural network
training. This set will be used to check for accuracy and other metrics
__while__ traning. It's generally small or it can be $0$ if desired.

- The validation set is taken from the training set, so if the validation
    is $0.2$, the real training set will be $0.7-0.2=0.5$ of the dataset.

### test_set

Set that will be use to check for accuracy and other metrics __after__
traning. Used to compute the final performance of the neural network.

- It is not explicitly stated in the JSON, as it's calculated in the
  program like $\text{test set} = 1-\text{training set}$. But it's good to
  know.

Example: we want a 0.7 train and 0.3 test, that uses 0.2 for validation.
That means that train will go from 0 to 0.5, validation will go from 0.5 to
0.7 and test from 0.7 to 1.

### `path`

(_string_) Path to load/save the neural network model.

- File extension is `.h5` that keras uses for this purpose.

### `labels`

List of (_string_) of the column names to predict. If left empty,
all the columns, except the ones removed, are predicted.