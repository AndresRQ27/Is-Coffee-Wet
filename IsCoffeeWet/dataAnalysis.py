from matplotlib.pyplot import show, figure


def graphData(dataset, columnList):
    """
    Function that graph the data in a column vs the time when it was
    taken. Help visualizing big gaps where the data was interpolated
    due to missing values.

    Parameters
    ----------
    - columnList: list of strings.
        List of the names of the columns to graph.
    """
    figure_index = 0

    # Graph every column in columnList
    for columnName in columnList:
        figure(figure_index)
        figure_index += 1

        dataset[columnName].plot().set_ylabel(columnName)

    show()
