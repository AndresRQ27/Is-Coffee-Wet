# JSON Format

- _datasetPath_: string. path of the dataset file to use.
  - Recommended to use an absolute path.
- _frequency_: string. Frequency to resample the dataset. It also
  includes the minimal timestep to use when predicting the values.
  - Possible values are:
    - `15min` for 15 minutes
    - `1h` for 1 hour
    - `1h30min` for every hour and a half
    - `1d` for 1 day
    - `1m` for 1 month
- _forecastWindow_: int. amount of __days__ to forecast.
- _graphData_: boolean. Indicates if it should show the graph of the data
  used.
  - Recommended in the first run to view the behavior of the data and find
    strange values.
  - Otherwise, recommended off (`False`)

## Preprocess

- _datetime_: list of column names (string) to merge into a datetime index.
  - With the current format:
    - The column "Date" has the date in DD/MM/YYYY
    - The column "Time" has the time in HH:MM AM/PM
  - Both columns are string at first, merge together and converted into
    datetime objects
- _datetime\_format_: format (string) the datetime will have after merging
  all the column.
- _nullList_: list of possible parameters to represent a bad value. They
  are change into NaN (Not a Number) and later interpolated
- _encodeTime_: dictionary of key-value with __extra__ features to add into
  the dataset. Time __must__ be in seconds,
  - This is the result of analyzing the trends in the data by graphing the
    Fast-Fourier Transform and choosing the higher values.
  - The idea is that the higher the value, the more representative is the
    feature for the dataset, making easier the NN training.
- _deleteColumns_: columns (string) to __NOT__ use for the forecasting.
  They will be eliminated during resampling.
  - It's' imperative to always delete the columns that use text (e.g. "Wind
    Direction")
- _specialFunctions_: in general, all the columns are resample using the
  mean function. If a different function must be applied to a specific
  column, it must be specified here (e.g. `"Rain": "sum"` as we want the
  total amount of rain in a given interval)
- _specialFormat_: in general, all the numeric values are float. Certain
  column may want to stay as integers (no decimals) after the resampling
  process, so specify the columns and the format here (e.g. `"Leaf Wet 1": "int"`).
