import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow._api.v2.signal


def graph_normalize(dataset, columns):
    """
    Function that graphs all the data to check its normalized values, in
    search of any inconsistency.

    Parameters
    ----------
    dataset: pandas.DataFrame
        Dataset with the normalized data.
    columns: list[string]
        Names of the columns in the dataset to set the as labels in the
        graph.

    """
    df_std = dataset.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(columns, rotation=90)
    plt.show()


def graph_data(dataset, config_file):
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
    """

    for name in config_file.columns:
        # Creates a new figure
        plt.figure()
        dataset[name].plot().set_ylabel(name)

    plt.show()


def freq_domain(dataset, config_file):
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

    See Also
    --------
    tensorflow._api.v2.signal.rfft: Real-valued Fast Fourier Transformation.
    """
    for name in config_file.columns:
        # Creates a new figure
        plt.figure()

        fft = tensorflow._api.v2.signal.rfft(dataset[name])
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

    plt.show()
