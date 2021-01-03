import argparse
import sys

def parse_args():
    """
    AI is creating summary for parse_args

    Parameters
    ----------
    [name]: [type]
        [description]
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
        help="",
        action="store_true"
    )
    parser.add_argument(
        "-u", "--update",
        dest="update_flag",
        help="",
        action="store_true"
    )
    parser.add_argument(
        "-p", "--predict",
        dest="predict_flag",
        help="",
        action="store_true"
    )
    parser.add_argument(
        "-g", "--graphs",
        dest="graph_flag",
        help="",
        action="store_true"
    )
    parser.add_argument(
        "-b", "--benchmark",
        dest="benchmark_flag",
        help="",
        action="store_true"
    )
    parser.add_argument(
        "--alterante-path",
        dest="alt_path",
        help="",
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()