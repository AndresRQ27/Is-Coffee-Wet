__Table of contents__

- [JSON Format](#json-format)
  - [General values](#general-values)
    - [`dataset_path`](#dataset_path)
    - [`frequency`](#frequency)
    - [`forecast_window`](#forecast_window)
    - [`output_path`](#output_path)
  - [preprocess](#preprocess)
    - [`datetime`](#datetime)
    - [`datetime_format`](#datetime_format)
    - [`null_list`](#null_list)
    - [`encode_time`](#encode_time)
    - [`delete_columns`](#delete_columns)
    - [`special_functions`](#special_functions)
    - [`special_format`](#special_format)
  - [neural_network](#neural_network)
    - [`model_name`](#model_name)
    - [`nn_path`](#nn_path)
    - [`submodel`](#submodel)
    - [`max_epochs`](#max_epochs)
    - [`learning_rate`](#learning_rate)
    - [`patience`](#patience)
    - [`batch_size`](#batch_size)

# JSON Format

## General values

### `dataset_path`

(_string_) path of the dataset file to use.

  - Relative path to the resources folder.
  - If it's a file, use that specific file to train/predict
  - If it's a directory, look for the most recent `.csv` to use ut

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

### `output_path`

(_str_) Path to save all the generated files by the system.


## preprocess


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

### `model_name`

(_string_) Name to given the model when saving its state. Also used to look
for a previously trained model in the [`path`](#path).

### `nn_path`

(_string_) Path to load/save the neural network model.

- File extension is `.h5` that keras uses for this purpose.

### `submodel`

(_string_) Model architecture to create and use during the execution. There currently are 3 options:

- "tcn"
- "cnn"
- "conv-lstm"

### `max_epochs`

(_int_) Maximum number of epochs to train the network.

### `learning_rate`

(_float_) Learning rate used when updating the weights of the model after a batch is done training.

### `patience`

(_int_) Number of epochs to wait before stopping the train, if the validation error has not improved.

### `batch_size`

(_int_) Number of data in a batch used for training.