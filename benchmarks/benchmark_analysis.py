import matplotlib.pyplot as plt
import numpy as np

# List of colors to use in the graph. There are 15 different colors
COLOR_LIST = ["red", "green", "blue", "cyan", "magenta", "yellow",
              "black", "dodgerblue", "slategray", "hotpink", "darkmagenta",
              "turquoise", "orange", "skyblue", "darkviolet"]

# List of markers to use in the graph. There are 15 different colors
MARKER_LIST = [".", "v", "^", "<", ">", "1", "2", "3", "s",
               "p", "P", "*", "h", "+", "X"]

# Size of the graphs to make
FIG_SIZE = (30, 10)


def benchmark_graph_all(dataset):
    """
        TODO: description
        Parameters
        ----------
        dataset: pandas.DataFrame
    """

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
            plt.plot(range(len(data)), data, label=name,
                     color=COLOR_LIST[name_id],
                     marker=MARKER_LIST[name_id])

        plt.xlabel("Epochs")
        plt.ylabel(column)
        plt.legend()

    plt.show()


def benchmark_graph_minmax(dataset):
    """
        TODO: description
        Parameters
        ----------
        dataset: pandas.DataFrame
    """

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

        # Add xticks on the middle of the group bars
        plt.xlabel(column, fontweight="bold")
        plt.xticks([r + bar_width for r in range(len(min_value))],
                   name_dict.keys())

        plt.ylabel("Value")
        plt.legend()
    plt.show()
