from types import new_class
import pandas as pd

from IsCoffeeWet.preprocess import data_derived as dd
from IsCoffeeWet.preprocess import data_parser as dp
from IsCoffeeWet.preprocess import data_graph as dg


def preprocess(config_file):
    """AI is creating summary for preprocess

    Parameters
    ----------
    config_file : [type]
        [description]

    Returns
    -------
    pandas.DataFrame
        [description]
    """
    dataset = pd.read_csv(config_file.ds_path)

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

    return new_dataset


def graphs(dataset, config_file, path):
    """AI is creating summary for graphs

    Parameters
    ----------
    dataset : [type]
        [description]
    config_file : [type]
        [description]
    path : [type]
        [description]
    """
    dg.graph_data(dataset=dataset,
                  config_file=config_file,
                  path=path)
    dg.freq_domain(dataset=dataset,
                   config_file=config_file,
                   path=path)
