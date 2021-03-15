import numpy as np
import pandas as pd

from IsCoffeeWet.neural_network.utils import split_dataset, compile_and_fit
from IsCoffeeWet.neural_network.window_generator import WindowGenerator
from IsCoffeeWet.tools.test import predict


def train(dataset, model, config_file, debug=False):
    """
    Train the neural network

    Parameters
    ----------
    dataset: pandas.DataFrame
        Full dataset, then split into the training set
    model : `tf.keras.Model`
        Model to train
    config_file: config_file.ConfigFile
        Object with the needed information to train the network
    debug : `bool`
        Flag to log and save the network training performance

    Returns
    -------
    tuple [`pandas.DataFrame`, `pandas.DataFrame`]
        DataFrames with the training history of the network and a
        prediction made immediately after finishing training, if the debug
        is activated
    """
    print(">>> Training model...")

    # *** Split dataset
    (datetime_index, train_ds,
     val_ds, test_ds) = split_dataset(dataset=dataset,
                                      config_file=config_file)

    # *** Create window
    train_window = WindowGenerator(input_width=config_file.forecast,
                                   label_width=config_file.forecast,
                                   shift=config_file.forecast,
                                   train_ds=train_ds,
                                   val_ds=val_ds,
                                   test_ds=test_ds,
                                   label_columns=config_file.labels,
                                   batch_size=config_file.batch_size)

    # *** Compile & fit
    train_history = compile_and_fit(model=model,
                                    window=train_window,
                                    nn_path=config_file.nn_path,
                                    model_name=config_file.model_name,
                                    patience=config_file.patience,
                                    learning_rate=config_file.lr,
                                    max_epochs=config_file.max_epochs)

    debug_prediction = None
    if debug:
        debug_prediction = predict(dataset=train_ds,
                                   model=model,
                                   config_file=config_file)

    return pd.DataFrame(train_history.history), debug_prediction


def update(mini_dataset, model, config_file, debug):
    """
    Updates (trains) the neural network with the values in the mini_dataset

    Parameters
    ----------
    mini_dataset : `pandas.DataFrame`
        Table containing a 2-week span, the first is for training and the
        second for validation
    model : `tf.keras.Model`
        Model to update
    config_file: config_file.ConfigFile
        Object with the needed information to update the network
    debug : `bool`
        Flag to generate a prediction after the update. Useful for
        consecutive updates (e.g. `updateAll` function)

    Returns
    -------
    tuple [`pandas.DataFrame`, `pandas.DataFrame`]
        DataFrames with the update history of the network and a
        prediction made immediately after finishing updating, if the debug
        is activated
    """
    print(">>> Updating model with the last data...")

    # Resets index to add datetime as a normal column
    mini_dataset = mini_dataset.reset_index().drop("index", axis=1)

    update_window = WindowGenerator(input_width=config_file.forecast,
                                    label_width=config_file.forecast,
                                    shift=config_file.forecast,
                                    train_ds=mini_dataset[
                                        :(config_file.forecast*2)],
                                    val_ds=mini_dataset[
                                        (config_file.forecast*2):],
                                    test_ds=None,
                                    label_columns=config_file.labels,
                                    batch_size=1)

    # *** Compile & fit
    update_history = compile_and_fit(model=model,
                                     window=update_window,
                                     nn_path=config_file.nn_path,
                                     model_name=config_file.model_name,
                                     # Aggressive end early to avoid over-fitting
                                     patience=2,
                                     # Use smaller lr during updates
                                     learning_rate=config_file.lr/10,
                                     max_epochs=config_file.max_epochs)

    debug_prediction = None

    if debug:
        debug_prediction = predict(dataset=mini_dataset[
            :(config_file.forecast)],
            model=model,
            config_file=config_file)

    return pd.DataFrame(update_history.history), debug_prediction


def updateAll(dataset, model, config_file, debug):
    """
    Updates (trains) the neural network with the values in the span of the
    last year in the dataset

    Parameters
    ----------
    dataset: pandas.DataFrame
        Full dataset, then split into the update set
    model : `tf.keras.Model`
        Model to update
    config_file: config_file.ConfigFile
        Object with the needed information to update the network
    debug : `bool`
        Flag to generate a prediction after the update. Useful for
        consecutive updates (e.g. `updateAll` function)

    Returns
    -------
    tuple [`pandas.DataFrame`, `pandas.DataFrame`]
        DataFrames with the update history of the network and a
        prediction made immediately after finishing updating, if the debug
        is activated
    """
    print(">>> Updating model with the last year information...")

    # *** Split dataset
    (datetime_index, train_ds,
     val_ds, test_ds) = split_dataset(dataset=dataset,
                                      config_file=config_file)

    # Merge the validation and test datasets
    val_ds = val_ds.append(test_ds)

    # Calculate the number of iterations to update the network
    batches = int(np.floor(len(val_ds)/config_file.forecast))

    # Empty dataframe to track the training metrics
    history = pd.DataFrame()

    # Initialize dataframe to save the debug predictions
    debug_prediction = pd.DataFrame(
        columns=config_file.labels) if debug else None

    for i in range(0, batches-3):
        print("Batch {}/{}".format(i, batches))
        batch_history, batch_pred = update(
            mini_dataset=val_ds[
                i*config_file.forecast:(i+4)*config_file.forecast],
            model=model,
            config_file=config_file,
            debug=debug
        )

        history = history.append(batch_history)

        if debug:
            debug_prediction = debug_prediction.append(batch_pred)

    return history, debug_prediction
