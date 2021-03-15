import os
import pandas as pd
from pandas._config import config

from IsCoffeeWet.preprocess.data_parser import convert_numeric
from IsCoffeeWet.preprocess.normalize import de_standardize
from IsCoffeeWet.neural_network.utils import analyze_metrics


def predict(dataset, model, config_file):
    """
    Predict the values for the next week

    Parameters
    ----------
    dataset : `pandas.DataFrame`
        Dataset with the pre-processed data
    model : `tf.keras.Model`
        Trained model to generate the predictions for the next week
    config_file: config_file.ConfigFile
        Object with the needed information to generate the prediction

    Returns
    -------
    `pandas.DataFrame`
        Next week prediction
    """
    print(">>> Predicting next values...")

    # Obtained the last day of to predict the next forecast
    data = dataset[-config_file.forecast:].reset_index().drop("index",
                                                              axis=1)

    # Reshape the data into a numpy array of
    # (batch, num_predictions, columns)
    # Batch will always be 1 because it's one prediction
    data = data.to_numpy().reshape((1, config_file.forecast,
                                    len(config_file.columns)))

    # Generate a prediction and reshape it to the correct output shape
    prediction = model(data).numpy().reshape((config_file.forecast,
                                              len(config_file.labels)))

    # Convert a two-dimension numpy array into a DataFrame
    prediction = pd.DataFrame(prediction,
                              columns=config_file.labels)

    # De-standardize the prediction
    prediction = de_standardize(dataset=prediction,
                                mean=config_file.mean[config_file.labels],
                                std=config_file.std[config_file.labels])

    return prediction


def save_predictions(prediction, last_date, config_file):
    """
    Process the generated predictions by the neural network to be saved
    into a csv file. Adds the datetime to each prediction and converts the
    predicted values into the correct data type.

    Parameters
    ----------
    prediction : pandas.DataFrame
        Table with the predicted values
    last_date : `str`
        Last date registered in the dataset
    config_file: config_file.ConfigFile
        Object with the needed information to convert the predictions
    """
    print(">>> Saving predictions...")

    # Adds one extra period to drop the first date
    # as that date is part of the original data
    index = pd.date_range(last_date,
                          periods=config_file.forecast+1,
                          freq=config_file.freq)

    # Sets the datetime index of the dataframe
    prediction.set_index(index[1:], inplace=True)

    # Run predictions through the convert_numeric function
    # Smoothens roughs values (like float leaf wet)
    prediction = convert_numeric(dataset=prediction,
                                 config_file=config_file)

    pred_path = os.path.join(config_file.output_path, "predictions.csv")

    prediction.to_csv(pred_path)

    print("Predictions saved to {}".format(pred_path))


def benchmark(predictions, labels, config_file):
    """
    Runs a benchmark by calculating the error of the prediction and
    graphing it vs the real values

    Parameters
    ----------
    predictions : `pandas.DataFrame`
        Table with the predictions
    labels : `pandas.DataFrame`
        Dataset with the ground truth values
    config_file: config_file.ConfigFile
        Object with the needed information to run the benchmarks
    """
    print(">>> Processing metrics of the last prediction period...")

    # De-standardized dataset section to use as labels
    labels = de_standardize(dataset=labels,
                            mean=config_file.mean,
                            std=config_file.std)

    # Resets the prediction index to avoid problems
    predictions.reset_index(inplace=True, drop=True)

    # Resets index to add datetime as a normal column
    labels.reset_index(inplace=True)

    # Pops the datetime from the dataset. Not use in the NN explicitly
    datetime_index = labels.pop("index")

    metrics_hour = analyze_metrics(y_true=labels[config_file.labels],
                                   y_pred=predictions,
                                   index=datetime_index,
                                   frequency="1H")

    metrics_day = analyze_metrics(y_true=labels[config_file.labels],
                                  y_pred=predictions,
                                  index=datetime_index,
                                  frequency="1D")

    metrics_week = analyze_metrics(y_true=labels[config_file.labels],
                                   y_pred=predictions,
                                   index=datetime_index,
                                   frequency="7D")

    # Save the metrics of the MAE in hours
    hour_path = os.path.join(config_file.output_path,
                             "benchmark_hour.csv")
    metrics_hour.to_csv(hour_path)

    # Save the metrics of the MAE in days
    day_path = os.path.join(config_file.output_path,
                            "benchmark_day.csv")
    metrics_day.to_csv(day_path)

    # Save the metrics of the MAE in a week
    week_path = os.path.join(config_file.output_path,
                             "benchmark_week.csv")
    metrics_week.to_csv(week_path)
