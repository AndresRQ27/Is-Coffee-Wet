import pandas as pd

from IsCoffeeWet.preprocess import data_derived as dd
from IsCoffeeWet.preprocess import data_parser as dp
from IsCoffeeWet.preprocess import data_graph as dg
from IsCoffeeWet.preprocess import normalize as norm


def preprocess(config_file):
    """
    Function that executes all the steps involved in the preprocess of the
    data. See `Notes` for the steps involved.

    Parameters
    ----------
    config_file: config_file.ConfigFile
        Object with the needed information to pre-process the dataset

    Returns
    -------
    tuple [pandas.DataFrame, pandas.Series, pandas.Series]
        Returns the pre-processed dataset, the mean and standard deviation
        of each column new_dataset. The last to parameters are used for
        de-normalizing.

    Notes
    -----
    1. Merge datetime: merges the date column with the time column. All the
       values must have the same format.
    2. Convert numeric: fills the missing values by interpolation, removes
       the empty values/columns and assign a type for each column.
    3. Sample data: groups the data into constant time intervals. The
       grouping of the data is m the mean function by default, but can be
       different if specified in the config file.
    4. Generate derived data: generates extra data added to the dataset,
       like leaf wet accum or a time cyclical encoding.
    5. Convert numeric: fills missing data that appeared during the
       grouping and assign again the data types to the column
    6. Normalize the dataset: Standardize the data so it's ready to be fed
       to the neural network
    """
    print(">>> Preprocessing dataset...")

    dataset = pd.read_csv(config_file.ds_path, engine="c")

    # *** Merge datetime
    if config_file.datetime:
        dataset = dp.merge_datetime(dataset=dataset,
                                    config_file=config_file)

    # *** Convert numeric
    dataset = dp.convert_numeric(dataset=dataset,
                                 config_file=config_file)

    # *** Sample dataset
    # By this point, index should be datetime

    # The sampled dataset is save in a new DataFrame
    # because some information is lost in the process
    # that the derived data could use for better precision
    new_dataset = pd.DataFrame()

    for column in config_file.columns:
        series = dp.sample_series(dataset[column],
                                  config_file=config_file)
        new_dataset = pd.concat([new_dataset, series], axis=1)

    # *** Generate derived data
    if "Leaf Wet 1" in config_file.columns:
        series = dd.create_leaf_wet_accum(dataset=dataset,
                                          config_file=config_file)
        new_dataset = pd.concat([new_dataset, series], axis=1)

    if config_file.encode:
        cyclical_ds = dd.create_cyclical_encoder(
            dataset_index=new_dataset.index,
            config_file=config_file)
        new_dataset = pd.concat([new_dataset, cyclical_ds], axis=1)

    # *** 2nd Convert numeric
    new_dataset = dp.convert_numeric(dataset=new_dataset,
                                     config_file=config_file)

    # Prints the description of the dataset without normalization
    print(new_dataset.describe().transpose())
    print(new_dataset.info(verbose=True))

    # *** Normalize the dataset
    new_dataset, ds_mean, ds_std = norm.standardize(dataset=new_dataset)

    # Updates the number of data available in the dataset
    config_file.num_data = len(new_dataset)

    return new_dataset, ds_mean, ds_std


def graphs(dataset, model, config_file, output_path):
    """
    Calls the functions to create all the graphs in the project

    Parameters
    ----------
    dataset : `pandas.DataFrame`
        Dataset with the pre-processed data
    model : `tf.keras.Model`
        Model of the neural network to graph and generate predictions
    config_file: config_file.ConfigFile
        Object with the needed information to graph the data
    output_path : `str`
        Path to save the graphs
    """
    print(">>> Printing graphs...")

    # Prints the preprocessed data
    dg.graph_data(dataset=dataset,
                  config_file=config_file,
                  output_path=output_path)

    # Prints the models architecture
    dg.graph_model(model=model,
                   model_name=config_file.model_name,
                   output_path=output_path)

    # Prints the labels vs predictions for the current week
    dg.graph_labels(dataset=dataset,
                    config_file=config_file,
                    output_path=output_path,
                    model=model)
