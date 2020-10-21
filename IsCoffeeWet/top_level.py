import os
import pandas as pd

from IsCoffeeWet import config_file as cf
from IsCoffeeWet import data_graph as dg
from IsCoffeeWet import data_parser as dp
from IsCoffeeWet import model_generator as mg
from IsCoffeeWet import neural_network as nn
from IsCoffeeWet import window_generator as wg

# Gets directory parent of the file
PATH_RESOURCES = os.getcwd() + "/resources"
PATH_IMAGES = PATH_RESOURCES + "/images"

# TEST
# Path to the configuration file
JSON_TEST = PATH_RESOURCES + "/config/test.json"

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")

# TODO: add read input for the config file

# Dataset configuration extracted from the JSON
config_ds = cf.ConfigFile(JSON_TEST)

# Loads the dataset
dataset = pd.read_csv(config_ds.path, engine="c")

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

# Information of the dataset
print(dataset.info(verbose=True))
print(dataset.describe().transpose())

# ! Program will stop until the graphs are checked
if config_ds.graph:
    dg.graph_data(dataset, config_ds, PATH_IMAGES)
    dg.freq_domain(dataset, config_ds, PATH_IMAGES)

# **********************************
# *** Neural Network preparation ***
# **********************************

# Number of rows in the dataset. Added in config_file only during execution
config_ds.num_data = dataset.shape[0]

# Normalize the dataset
dataset, mean, std = nn.standardize(dataset)
print(dataset.describe().transpose())

# Datetime index still useful for graphs
datetime_index, train_ds, val_ds, test_ds = nn.split_dataset(dataset, config_ds)

# Generates a window for the training of the neural network
window = wg.WindowGenerator(input_width=config_ds.forecast * 24,
                            label_width=config_ds.forecast * 24,
                            shift=config_ds.forecast * 24,
                            train_ds=train_ds,
                            val_ds=val_ds,
                            test_ds=test_ds,
                            label_columns=config_ds.columns)

# Plots a random window from the test dataset to show the label
if config_ds.graph:
    for name in config_ds.columns:
        window.plot(name, PATH_IMAGES)

# **********************************
# **** Neural Network training *****
# **********************************

# Constructs the  model
# model = mg.

# Show the model summary (layers, i/o data, etc.)
# model.summary()

# TODO: implement compile and fitting
# history = nn.compile_and_fit(conv_model, window, 5)

# Shows graphs for each column
# for name in window.label_columns:
#     window.plot(name, model=model)

input("Stop?")
