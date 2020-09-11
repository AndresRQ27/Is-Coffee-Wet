from tensorflow.signal import rfft
import numpy as np
import matplotlib.pyplot as plt

# Global parameter for the number of figures
figure_index = 0


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
    global figure_index

    for columnName in columnList:
        # Creates a new figure
        plt.figure(figure_index)
        figure_index += 1

        dataset[columnName].plot().set_ylabel(columnName)

    plt.show()


def freqDomain(dataset, columnList):
    # TODO: documentation
    global figure_index

    for column in columnList:
        # Creates a new figure
        plt.figure(figure_index)
        figure_index += 1

        fft = rfft(dataset[column])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(dataset[column])
        hours_per_year = 24*365.2524
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        plt.step(f_per_year, np.abs(fft))
        plt.xscale('log')
        plt.ylim(0, 1000000)
        plt.xlim([0.1, max(plt.xlim())])

        # Show some normals datapoints
        plt.xticks([1, 365.2524, 365.2524*24],
                   labels=['1/Year', '1/day', '1/hour'])
        _ = plt.xlabel('Frequency (log scale)')

    plt.show()
