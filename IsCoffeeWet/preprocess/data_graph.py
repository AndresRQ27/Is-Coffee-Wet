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


def freq_domain(dataset, config_file, output_path):
    """
    Function that graph the data in the frequency domain by using Fourier Transform.
    Useful when analyzing the information to see which frequencies are the 
    most important in the dataset. They can be added as features by using
    `cyclical_encoder` to help the NN convergence

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset where to extract the columns to graph
    config_file: config_file.ConfigFile
        Configuration file with the name of all the columns to apply the
        Real-valued Fast Fourier Transformation.
    output_path: string
        Path to the folder to save the graphs

    See Also
    --------
    tensorflow._api.v2.signal.rfft: Real-valued Fast Fourier Transformation.
    """
    print(">>> Creating Fourier graphs...")

    # Creates the folder to save the graphs
    output_path = os.path.join(output_path, "graphs", "fourier")
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print("\n{} already exists".format(output_path))

    for name in config_file.columns:
        # Creates a new figure
        plt.figure(figsize=FIG_SIZE)

        fft = signal.rfft(dataset[name])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(dataset[name])
        hours_per_year = 24 * 365.2524
        years_per_dataset = n_samples_h / hours_per_year

        f_per_year = f_per_dataset / years_per_dataset
        plt.step(f_per_year, np.abs(fft))
        plt.xscale('log')
        plt.ylim(0, 1000000)
        plt.xlim([0.1, max(plt.xlim())])

        # Show some normals data points
        plt.xticks([1, 365.2524, 365.2524 * 24],
                   labels=['1/Year', '1/day', '1/hour'])
        _ = plt.xlabel('Frequency (log scale)')
        _ = plt.ylabel(name)

        # Saves the image
        plt.savefig("{}/{}.png".format(output_path, name))
        plt.close()


def graph_model(model, output_path):
    """
    # TODO: documentation
    """
    print(">>> Creating model architecture graph...")

    # Creates the folder to save the graphs
    output_path = os.path.join(output_path, "graphs", "model")
    try:
        os.makedirs(output_path)
    except FileExistsError:
        print("\n{} already exists".format(output_path))

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
    graph_window = WindowGenerator(input_width=config_file.forecast,
                                   label_width=config_file.forecast,
                                   shift=config_file.forecast,
                                   label_columns=config_file.labels)

    # Make a graph for each label
    for label in graph_window.label_columns:
        # Prediction data is the info of the last week
        prediction_data = dataset[
            -config_file.forecast*2:-config_file.forecast,
            label]

        # Prediction label is the info of this week
        prediction_label = dataset[
            -config_file.forecast:,
            label]

        graph_window.plot(label,
                          output_path,
                          (prediction_data,
                           prediction_label),
                          model)
