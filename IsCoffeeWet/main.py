import copy
import os

from IsCoffeeWet.tools.config_file import ConfigFile
from IsCoffeeWet.tools.parser import parse_args
from IsCoffeeWet.tools.preprocess import preprocess, graphs
from IsCoffeeWet.tools.train import train, update, updateAll
from IsCoffeeWet.tools.test import benchmark, predict, save_predictions
from IsCoffeeWet.neural_network.utils import load_model, save_model
from IsCoffeeWet.neural_network.model_generator import build_model


def main():
    """
    Main function to spawn the train and test process.
    """
    print("#"*42)
    print("#"*3+" Welcome to the IsCoffeeWet project "+"#"*3)
    print("#"*42)

    args = parse_args()

    # Main path where all the files are saved (modules, resources,
    # neural networks)
    if args.alt_path:
        # Use a custom path
        parent_path = args.alt_path
    else:
        # Use the parent path of this project
        parent_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

    config_file = ConfigFile(path_config=args.config_file,
                             parent_path=parent_path)

    # *** Preprocess the dataset
    assert config_file.ds_path != "", "[Error]: Empty dataset. Execution finished"
    (dataset,
     config_file.mean,
     config_file.std) = preprocess(config_file=config_file)  # It's standardize

    # *** Loads the model
    model = load_model(submodel=config_file.submodel,
                       name=config_file.model_name,
                       path=config_file.nn_path)

    # If model is None, create a model
    if model is None:
        model = build_model(input_size=(config_file.forecast,
                                        len(config_file.columns)),
                            output_size=(config_file.forecast,
                                         len(config_file.labels)),
                            submodel=config_file.submodel)

    # Check if the model is empty. If model is None, throw error
    assert model is not None, "[Error]: Couldn't create model. Check the submodel type"

    # Print a summary of the model to use
    model.summary()

    if args.train_flag:
        # *** Train model
        (train_history,
         debug_predictions1) = train(dataset=copy.deepcopy(dataset),
                                     model=model,
                                     config_file=config_file,
                                     debug=args.debug_flag)

        # *** Update with data of the last year
        (update_history,
         debug_predictions2) = updateAll(dataset=copy.deepcopy(dataset),
                                         model=model,
                                         config_file=config_file,
                                         debug=args.debug_flag)

        history = train_history.append(update_history)

        # *** Merge the debug predictions
        if args.debug_flag:
            debug_predictions = debug_predictions1.append(
                debug_predictions2)

    elif args.updateAll_flag:
        # *** Update with data of the last year
        (history,
         debug_predictions) = updateAll(dataset=copy.deepcopy(dataset),
                                        model=model,
                                        config_file=config_file,
                                        debug=args.debug_flag)

    elif args.update_flag:
        # *** Update with the last data
        (history,
         debug_predictions) = update(mini_dataset=dataset[-config_file.forecast*2:],
                                     model=model,
                                     config_file=config_file,
                                     debug=args.debug_flag)

    # *** Save model
    save_model(model=model,
               path=config_file.nn_path,
               name=config_file.model_name)

    if args.predict_flag:
        # *** Generate a prediction for the next period
        prediction = predict(dataset=dataset,
                             model=model,
                             config_file=config_file)

        # *** Save the prediction
        save_predictions(prediction=prediction,
                         last_date=dataset.index[-1:][0],
                         config_file=config_file)

    if args.benchmark_flag:
        # Predict using the values of the last week
        benchmark_pred = predict(dataset=dataset[
            -config_file.forecast*2:-config_file.forecast],
            model=model,
            config_file=config_file)

        # Calculates the error of this week predictions
        # Labels are de-standardize inside the function
        benchmark(predictions=benchmark_pred,
                  labels=dataset[-config_file.forecast:])

    # Create graphs when all data is available
    if args.graph_flag:
        graphs(dataset=dataset,
               model=model,
               config_file=config_file,
               output_path=config_file.output_path)

    # Save debug variables not normally shown to the user
    if args.debug_flag:
        # If a training/update has been made to the model
        if args.train_flag or args.updateAll_flag or args.update_flag:
            metrics_path = os.path.join(config_file.output_path,
                                        "model_metrics.csv")
            history.to_csv(metrics_path)

            predict_path = os.path.join(config_file.output_path,
                                        "debug_predictions.csv")
            debug_predictions.to_csv(predict_path)


if __name__ == "__main__":
    # By default, it should be `/workpaces` but could be any other path
    main()

"""

# Show the model summary (layers, i/o data, etc.)
# 

# Graphs all the labels in the model

"""
