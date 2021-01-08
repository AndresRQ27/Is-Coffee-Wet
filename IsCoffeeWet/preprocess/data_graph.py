import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow._api.v2.signal as signal
from IsCoffeeWet.neural_network.window_generator import WindowGenerator

# Size of the graphs to make
FIG_SIZE = (30, 10)


def graph_data(dataset, config_file, output_path):
    """
    Function that graph the data in a column vs the time when it was
    taken. Help visualizing big gaps where the data was interpolated
    due to missing values.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the columns to graph
    config_file: config_file.ConfigFile
        Configuration file with the name of all the columns to graph
    output_path: string
        Path to the folder to save the graphs
    """
    print(">>> Creating graphs from dataset values...")

    # Creates the folder to save the graphs
    output_path = os.path.join(output_path, "graphs", "data_values")
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print("\n{} already exists".format(output_path))

    for name in config_file.columns:
        # Creates a new figure
        plt.figure(figsize=FIG_SIZE)
        dataset[name].plot().set_ylabel(name)
        # Saves the image
        plt.savefig("{}/{}.png".format(output_path, name))
        plt.close()


def graph_model(model, model_name, output_path):
    """
    # TODO: documentation
    """
    print(">>> Creating model architecture graph...")

    # Creates the folder to save the graphs
    output_path = os.path.join(output_path, "graphs",
                               "{}.png".format(model_name))

    tf.keras.utils.plot_model(model, output_path, show_shapes=True)


def graph_labels(dataset, config_file, output_path, model):
    """
    docstring
    """
    print(">>> Creating labels graphs of the last predictions...")

    # Creates the folder to save the graphs
    output_path = os.path.join(output_path, "graphs", "labels")
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print("\n{} already exists".format(output_path))

    # Creates an empty Window, used for graphing
    # Added a small train_ds to initialize column index
    graph_window = WindowGenerator(input_width=config_file.forecast,
                                   label_width=config_file.forecast,
                                   shift=config_file.forecast,
                                   train_ds=dataset.head(),
                                   label_columns=config_file.labels)

    # Prediction data is the info of the last week
    prediction_data = dataset[
        -config_file.forecast*2:-config_file.forecast
        ].to_numpy().reshape((1, 168, 19))

    # Prediction label is the info of this week
    prediction_label = dataset[
        -config_file.forecast:
        ].to_numpy().reshape((1, 168, 19))
    
    for label in graph_window.label_columns:
        # Make a graph for each label
        graph_window.plot(plot_col=label,
                          path=output_path,
                          data=(prediction_data,
                                prediction_label),
                          model=model)
