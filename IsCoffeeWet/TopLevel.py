from pandas import read_csv
#from IsCoffeeWet import config_file as cf
#from IsCoffeeWet import data_graph as dg
#from IsCoffeeWet import data_parser as dp
import config_file as cf
import data_graph as dg
import data_parser as dp
import neural_network as nn
import window_generator as wg

JSON_TEST = "resources/test.json"
PARSE_TEST = True

# JSON_TEST = "resources\estXCompleta_hours.json"

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")

# Dataset configuration extracted from the JSON
ds_config = cf.ConfigFile(JSON_TEST)

# Initialize the dataset
if PARSE_TEST:
    # Loads the dataset
    dataset = read_csv(ds_config.path, engine="c")

    # Remove trailing and leading spaces from column names
    dataset.columns = dataset.columns.str.strip()

    # Parse the dataset
    # convert_numeric twice to fill empty values after sampling
    dataset = (dataset.pipe(dp.merge_datetime, config_file=ds_config)
                      .pipe(dp.convert_numeric, config_file=ds_config)
                      .pipe(dp.sample_dataset, config_file=ds_config)
                      .pipe(dp.convert_numeric, config_file=ds_config)
                      .pipe(dp.cyclical_encoder, config_file=ds_config)
               )

    new_path = ds_config.path
    new_path = new_path.replace(".csv", "_parsed.csv")

    # Saves the parsed dataset
    dataset.to_csv(new_path)
    print("A copy of your dataset has been save into: " + new_path)

else:
    # Sets the index using Datetime column
    dataset = read_csv(ds_config.path, engine="c",
                       index_col="Datetime", parse_dates=True)
    # Infers the frequency
    dataset = dataset.asfreq(ds_config.freq)

# Execution ends with the graph as it requires a lot of memory
# to have the graphs with the NN training
if ds_config.graph:
    dataset.describe().transpose()
    dg.graph_data(dataset, ds_config.graph_columns)
    dg.freq_domain(dataset, ds_config.columns[:2])

# Split of the data
column_indices = {name: i for i, name in enumerate(dataset.columns)}

# Number of rows in the dataset
num_data, num_features = dataset.shape

if ds_config.graph:
    dg.graph_normalize(dataset, dataset.keys())
