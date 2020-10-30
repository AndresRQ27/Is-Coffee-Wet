import os

import matplotlib.pyplot as plt
import numpy as np

# List of colors to use in the graph. There are 20 different colors
COLOR_LIST = ["red", "green", "blue", "cyan", "magenta", "gold",
              "black", "dodgerblue", "slategray", "hotpink", "darkmagenta",
              "turquoise", "orange", "skyblue", "darkviolet", "pink",
              "navy", "chartreuse", "darkkhaki", "gray", "coral"]

# List of markers to use in the graph. There are 20 different colors
MARKER_LIST = [".", "v", "^", "<", ">", "1", "2", "3", "s", "x",
               "p", "P", "*", "h", "+", "X", "D", "4", "H", "8"]

# Size of the graphs to make
FIG_SIZE = (30, 10)


def benchmark_graph_all(dataset, path, name, max_epochs=100):
    """
        Function that graphs the reported metrics from the history according
        to the epoch of the training in which they were taken.

        Parameters
        ----------
        dataset: pandas.DataFrame
            Dataset of the history from the tests
        path: string
            Path to the parent folder to save the graphs
        name: string
            name of the cross-validation graphed
        max_epochs: int, optional
            Limit of the number of epochs to show for a test. By default,
            only shows the 100 epochs.
    """
    path = path + "/" + name
    try:
        os.mkdir(path)
    except FileExistsError:
        print("{} already exists".format(path))

    # Retrieves the unique test names in the dataset
    # Assign an id to a test name. Use for colors and markers
    name_dict = {name: i for i, name in
                 enumerate(dataset["Name"].drop_duplicates())}

    # Obtains the list of columns to graph
    # Removes the last columns ("Name")
    graph_columns = dataset.columns.tolist()[:-1]

    # Plots a graph for each metric
    for column in graph_columns:
        plt.figure(figsize=FIG_SIZE)

        # Plots all the data for each test
        for name in name_dict:
            name_id = name_dict[name]
            data = dataset.loc[dataset["Name"] == name, column]

            epochs = max_epochs if len(data) > max_epochs else len(data)

            plt.plot(range(epochs), data[-epochs:], label=name,
                     color=COLOR_LIST[name_id],
                     marker=MARKER_LIST[name_id])

        plt.xlabel("Epochs")
        plt.ylabel(column)
        plt.legend()
        # Saves the image
        plt.savefig("{}/{}.png".format(path, column))
        plt.close()


def benchmark_graph_summary(dataset, path, name):
    """
        Function that graphs a summary of the most important values inside
        the history. These are min, max, mean and last reported value.

        Parameters
        ----------
        dataset: pandas.DataFrame
            Dataset of the history from the tests
        path: string
            Path to the parent folder to save the graphs
        name: string
            name of the cross-validation graphed
    """
    path = path + "/" + name
    try:
        os.mkdir(path)
    except FileExistsError:
        print("{} already exists".format(path))

    # Retrieves the unique test names in the dataset
    # Assign an id to a test name. Use for colors and markers
    name_dict = {name: i for i, name in
                 enumerate(dataset["Name"].drop_duplicates())}

    # Obtains the list of columns to graph
    # Removes the last columns ("Name")
    graph_columns = dataset.columns.tolist()[:-1]

    # set width of bar
    bar_width = 0.2

    # Plots a graph for each metric
    for column in graph_columns:
        plt.figure(figsize=FIG_SIZE)
        min_value = []
        max_value = []
        avg_value = []
        last_value = []

        # Obtains the values to plot from each test
        for name in name_dict:
            data = dataset.loc[dataset["Name"] == name, column]
            min_value.append(data.min())
            max_value.append(data.max())
            avg_value.append(data.mean())
            last_value.append(data.iloc[-1])

        # Set position of bar on X axis
        min_pos = np.arange(len(min_value))
        max_pos = [x + bar_width for x in min_pos]
        avg_pos = [x + bar_width for x in max_pos]
        last_pos = [x + bar_width for x in avg_pos]

        # Make the plot
        plt.bar(min_pos, min_value,
                color=COLOR_LIST[0], width=bar_width,
                edgecolor="white", label="min")
        plt.bar(max_pos, max_value,
                color=COLOR_LIST[1], width=bar_width,
                edgecolor="white", label="max")
        plt.bar(avg_pos, avg_value,
                color=COLOR_LIST[2], width=bar_width,
                edgecolor="white", label="average")
        plt.bar(last_pos, last_value,
                color=COLOR_LIST[3], width=bar_width,
                edgecolor="white", label="last")

        for i in range(len(min_value)):
            plt.text(min_pos[i]-0.09, 0, round(min_value[i], 3), fontsize=9, color="black")
            plt.text(max_pos[i]-0.09, 0, round(max_value[i], 3), fontsize=9, color="black")
            plt.text(avg_pos[i]-0.09, 0, round(avg_value[i], 3), fontsize=9, color="black")
            plt.text(last_pos[i]-0.09, 0, round(last_value[i], 3), fontsize=9, color="black")

        # Add xticks on the middle of the group bars
        plt.xlabel(column, fontweight="bold")
        plt.xticks([r + bar_width for r in range(len(min_value))],
                   name_dict.keys())

        plt.ylabel("Value")
        plt.legend()
        # Saves the image
        plt.savefig("{}/{}_bars.png".format(path, column))
        plt.close()
