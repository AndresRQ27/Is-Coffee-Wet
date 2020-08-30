from matplotlib.pyplot import show, figure

def graphData(dataset, columnList, stop):
    """
    Function that graph the data in a column vs the time when it was
    taken. Help visualizing big gaps where the data was interpolated
    due to missing values.

    Parameters
    ----------
    - columnList: list of strings.
        List of the names of the columns to graph.
    - stop: bool.
        If true, the execution stops until all the figures are closed.
        Else, continue execution without stopping without waiting for
        the graphs to close. This is used because when the program 
        finishes, all the figures are close regardless they have the
        user focus or not.
    """
    figure_index = 0

    # Graph every column in columnList
    for columnName in columnList:
        figure(figure_index)
        figure_index += 1

        dataset[columnName].plot().set_ylabel(columnName)

    show(block=stop)