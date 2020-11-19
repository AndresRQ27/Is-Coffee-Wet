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


def graph_epochs(dataset, path_resources, name, max_epochs=100):
    """
        Function that graphs the reported metrics from the history according
        to the epoch of the training in which they were taken.

        Parameters
        ----------
        dataset: pandas.DataFrame
            Dataset of the history from the tests
        path_resources: string
            Path to the parent folder to save the graphs
        name: string
            name of the cross-validation graphed
        max_epochs: int, optional
            Limit of the number of epochs to show for a test. By default,
            only shows the 100 epochs.
    """
    path_resources = path_resources + "/" + name
    try:
        os.makedirs(path_resources)
    except FileExistsError:
        print("\n{} already exists".format(path_resources))

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
        plt.savefig("{}/{}.png".format(path_resources, column))
        plt.close()


def graph_performance(datasets, graph_columns, path_resources, name):
    """
        Function that graphs a summary of the most important values inside
        the history. These are min, max, mean and last reported value.

        Parameters
        ----------
        datasets: dict[pandas.DataFrame]
            Datasets of the history from the tests. Many datasets can be
            added to the list, but all must have the same columns to graph
        graph_columns: list[string]
            names of the columns from the dataset to graph
        path_resources: string
            Path to the parent folder to save the graphs
        name: string
            name of the cross-validation graphed
    """
    path_resources = path_resources + "/" + name
    try:
        os.makedirs(path_resources)
    except FileExistsError:
        print("\n{} already exists".format(path_resources))

    # set width of bar
    bar_width = 1 / (len(datasets) + 1)

    # Plots a graph for each metric
    for column in graph_columns:
        plt.figure(figsize=FIG_SIZE)

        # Counter for the datasets to graph. They will be the bars in the graph
        bars = len(datasets) - 1

        # Graph the data of each dataset with the same-name column
        for key, value in datasets.items():
            # Numbers to plot in the graph
            # Don't graph empty values
            data = value[column].dropna(axis=0)

            # Generates the new positions that the bars will take in the graph
            positions = np.arange(len(data)) + bar_width * bars

            # Make the plot
            plt.bar(positions, data.to_list(),
                    color=COLOR_LIST[bars], width=bar_width,
                    edgecolor="white", label=key)

            # Don't graph empty values
            for i in range(len(data)):
                # Graphs the values as text
                plt.text(x=positions[i] - bar_width / 2,
                         y=0,
                         s=round(data[i], 3),
                         fontsize=7,
                         color="white")

            # Adds 1 to the counter of datasets graphed
            bars -= 1

        else:
            try:
                names = value["Name"].dropna(axis=0).drop_duplicates().to_list()

            except:
                # There is no name column, use the positions
                names = positions

        # Add xticks on the middle of the group bars
        plt.xlabel(column, fontweight="bold")
        plt.xticks([r + bar_width for r in positions],
                   names)

        plt.ylabel("Value")
        plt.legend()
        # Saves the image
        plt.savefig("{}/{}_bars.png".format(path_resources, column))
        plt.close()


def graph_predictions(datasets, graph_columns, path_resources, name):
    """
        Function that graphs the reported metrics from the history according
        to the epoch of the training in which they were taken.

        Parameters
        ----------
        datasets: dict[pandas.DataFrame]
            Datasets of the predictions from the tests. Many datasets can be
            added to the list, but all must have the same columns to graph
        graph_columns: list[string]
            names of the columns from the dataset to graph
        path_resources: string
            Path to the parent folder to save the graphs
        name: string
            name of the cross-validation graphed
    """
    path_resources = path_resources + "/" + name
    try:
        os.makedirs(path_resources)
    except FileExistsError:
        print("\n{} already exists".format(path_resources))

    # Plots a graph for each metric
    for column in graph_columns:
        plt.figure(figsize=FIG_SIZE)

        # Counter for the datasets to graph. They will be the colors to use
        color = len(datasets) - 1

        # Plots all the data for each test
        for key, value in datasets.items():
            # Numbers to plot in the graph
            # Don't graph empty values
            data = value[column].dropna(axis=0)

            plt.plot(range(len(data)), data, label=key,
                     color=COLOR_LIST[color],
                     marker=MARKER_LIST[color])

            color -= 1

        plt.xlabel("Predictions")
        plt.ylabel(column)
        plt.legend()
        # Saves the image
        plt.savefig("{}/{}.png".format(path_resources, column))
        plt.close()
