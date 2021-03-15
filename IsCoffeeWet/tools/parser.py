import argparse
import sys

def parse_args():
    """
    Parse the following arguments to execute IsCoffeeWet project.

    Params
    ------
    config_file : `str`
        Path to the config file
    train_flag : `bool`
        Executes neural network training
    update_flag : `bool`
        Updates the neural network with the last week results
    predict_flag : `bool`
        Generates a prediction using the neural network
    graph_flag : `bool`
        Graphs preprocessed data, model architecture and labels
    benchmark_flag : `bool`
        Benchmarks the neural network error fo the last week prediction
    debug_flag : `bool`
        Generates debug information for more insight analysis of the network. 
    updateAll_flag : `bool`

    alt_path : `str`
        Alternate route where all the files are saved

    Returns
    -------
    `argparse.ArgumentParser`
        Object with the arguments passed by console
    """
    parser = argparse.ArgumentParser(
        description=("Provide training and testing pipeline for " +
        "the IsCoffeeWet project.")
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        help="Path to the config file",
        default="resources/tests/configs/test.json",
        type=str,
    )
    parser.add_argument(
        "-t", "--train",
        dest="train_flag",
        help="Executes neural network training",
        action="store_true"
    )
    parser.add_argument(
        "-u", "--update",
        dest="update_flag",
        help="Updates the neural network with the last week results",
        action="store_true"
    )
    parser.add_argument(
        "-p", "--predict",
        dest="predict_flag",
        help="Generates a prediction using the neural network",
        action="store_true"
    )
    parser.add_argument(
        "-g", "--graphs",
        dest="graph_flag",
        help="Graphs preprocessed data, model architecture and labels",
        action="store_true"
    )
    parser.add_argument(
        "-b", "--benchmark",
        dest="benchmark_flag",
        help="Benchmarks the neural network error fo the last week prediction",
        action="store_true"
    )
    parser.add_argument(
        "-d", "--debug",
        dest="debug_flag",
        help=("Generates debug information for more insight analysis of the network. " +
        "Prints training and prediction metrics"),
        action="store_true"
    )
    parser.add_argument(
        "--update-all",
        dest="updateAll_flag",
        help=("After the training, updates one year worth of information. " +
        "Used only when training was completed but the year update wasn't."),
        action="store_true"
    )
    parser.add_argument(
        "--alterante-path",
        dest="alt_path",
        help=("Specifies an alternate route where all the files are saved" +
        "(modules, resources, neural networks)"),
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        exit()
    return parser.parse_args()