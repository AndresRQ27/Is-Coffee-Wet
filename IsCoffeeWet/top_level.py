from pandas import read_csv
from IsCoffeeWet import config_file as cf
from IsCoffeeWet import data_graph as dg
from IsCoffeeWet import data_parser as dp
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import window_generator as wg

# import config_file as cf
# import data_graph as dg
# import data_parser as dp
# import neural_network as nn
# import window_generator as wg

# Path to the configuration file
JSON_TEST = "D:/VMWare/Shared/Is-Coffee-Wet/resources/estXCompleta_hours.json"
# JSON_TEST = "D:/VMWare/Shared/Is-Coffee-Wet/resources/test.json"

# Amount of neurons in a convolutional layer
FILTER_SIZE = 16

# A day
CONV_WIDTH = 24

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")

# TODO: add read input for the config file

# Dataset configuration extracted from the JSON
config_ds = cf.ConfigFile(JSON_TEST)

# Loads the dataset
dataset = read_csv(config_ds.path, engine="c")

# *********************************
# ******** Dataset Parsing ********
# *********************************
# convert_numeric twice to fill empty values after sampling
dataset = (dataset.pipe(dp.merge_datetime, config_file=config_ds)
           .pipe(dp.convert_numeric, config_file=config_ds)
           .pipe(dp.sample_dataset, config_file=config_ds)
           .pipe(dp.convert_numeric, config_file=config_ds)
           .pipe(dp.cyclical_encoder, config_file=config_ds)
           )

new_path = config_ds.path
new_path = new_path.replace(".csv", "_parsed.csv")

# Saves the parsed dataset
# ! Can be change to save file in a new format
dataset.to_csv(new_path)
print("A copy of your dataset has been save into: " + new_path)

# Information of the dataset
print(dataset.info(verbose=True))
print(dataset.describe().transpose())

# ! Program will stop until the graphs are checked
if config_ds.graph:
    dg.graph_data(dataset, config_ds)
    dg.freq_domain(dataset, config_ds)

# **********************************
# *** Neural Network preparation ***
# **********************************

column_indices = {name: i for i, name in enumerate(dataset.columns)}

# Number of rows in the dataset. Added in config_file only during execution
config_ds.num_data, config_ds.num_features = dataset.shape

# Normalize the dataset
dataset = nn.normalize(dataset)
print(dataset.describe().transpose())

# Show the graph of the normalize data
if config_ds.graph:
    dg.graph_normalize(dataset, dataset.keys())

datetime_index, train_ds, val_ds, test_ds = nn.split_dataset(dataset, config_ds)

# INFO: Input width and shift size are the same. Could be change
window = wg.WindowGenerator(input_width=config_ds.forecast*24, label_width=config_ds.forecast*24,
                            shift=config_ds.forecast*24, train_ds=train_ds,
                            val_ds=val_ds, test_ds=test_ds,
                            label_columns=config_ds.columns)

window.plot("Temp Out")

# **********************************
# **** Neural Network training *****
# **********************************

# Constructs the convolutional model
conv_model = nn.convolutional_model(FILTER_SIZE,
                                    CONV_WIDTH,
                                    (window.input_width, dataset.shape[1]),
                                    (window.label_width, len(window.label_columns))
                                    )

# Show the model summary (layers, i/o data, etc.)
conv_model.summary()

history = nn.compile_and_fit(conv_model, window, 5)

# Shows graphs for each column
for name in window.label_columns:
    window.plot(name, model=conv_model)

input("Stop?")