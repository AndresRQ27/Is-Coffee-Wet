import numpy as np
import pandas as pd

from IsCoffeeWet.neural_network.utils import split_dataset, compile_and_fit
from IsCoffeeWet.neural_network.window_generator import WindowGenerator
from IsCoffeeWet.tools.test import predict


def train(dataset, model, config_file, debug):
    """
    # TODO: documentation
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
        debug_prediction = pd.DataFrame(debug_prediction,
                                        columns=config_file.labels)

    return pd.DataFrame(train_history.history), debug_prediction


def update(mini_dataset, model, config_file, debug):
    """
    # TODO: documentation
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

        debug_prediction = pd.DataFrame(debug_prediction,
                                        columns=config_file.labels)

    return pd.DataFrame(update_history.history), debug_prediction


def updateAll(dataset, model, config_file, debug):
    """
    # TODO: documentation
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
    debug_prediction = pd.DataFrame() if debug else None

    for i in range(4, batches):
        print("Batch {}/{}".format(i, batches))
        history, batch_pred = history.append(update(mini_dataset=val_ds[
            (i-4)*config_file.forecast:i*config_file.forecast],
            model=model,
            config_file=config_file),
            debug=debug)

        debug_prediction.append(batch_pred)

    return history, debug_prediction
