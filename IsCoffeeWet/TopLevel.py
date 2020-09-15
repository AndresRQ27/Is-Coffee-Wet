from pandas import read_csv
import config_file as cf
import data_graph as da
import data_parser as dp

NEW_DATASET = False
DATA_PATH = "resources\estXCompleta_1H_encodedDays_encodedHours.csv"
CONFIG_PATH = "resources\estXCompleta_hours.json"
GRAPH_DATA = True
DAY_WINDOW = 14*24

print("******************************************")
print("*** Welcome to the IsCoffeeWet project ***")
print("******************************************")

# TODO: change the dataset read to automatic load the last dataset
# TODO: automatic search the json in a folder
# Dataset configuration extracted from the JSON
ds_config = cf.ConfigFile(CONFIG_PATH)

# Initialize the dataset
if NEW_DATASET:
    # Loads the dataset
    dataset = read_csv(DATA_PATH, engine="c")

    # Remove trailing and leading spaces from column names
    dataset.columns = dataset.columns.str.strip()

    # Parse the dataset
    # convert_numeric twice to fill empty values after sampling
    dataset = (dataset.pipe(dp.merge_datetime, dateName=ds_config.date, timeName=ds_config.time)
                      .pipe(dp.convert_numeric, column_type=ds_config.cType, nullList=ds_config.null)
                      .pipe(dp.sample_dataset, column_function=ds_config.cFunction, frequency=ds_config.freq)
                      .pipe(dp.convert_numeric, column_type=ds_config.cType, nullList=ds_config.null)
                      .pipe(dp.cyclical_encoder, encodeList=ds_config.encode)
               )

    # Value used for filename of new dataset
    encode = ""
    for name in ds_config.encode:
        encode = encode + "_" + name[0]

    new_path = DATA_PATH.replace(".csv", "")
    new_path = new_path + "_" + ds_config.freq + encode + ".csv"

    # Saves the parsed dataset
    dataset.to_csv(new_path)
    print("A copy of your dataset has been save into: " + new_path)

else:
    # Sets the index using Datetime column
    dataset = read_csv(DATA_PATH, engine="c",
                       index_col="Datetime", parse_dates=True)
    # Infers the frequency
    dataset = dataset.asfreq(ds_config.freq)

#print("Do a graphical analysis of the dataset?")
#print("(If yes, NN training must be done in another execution)")
#graph_data = input("- (yes/no): ")

# Execution ends with the graph as it requires a lot of memory
# to have the graphs with the NN training
if GRAPH_DATA == "yes":
    dataset.describe().transpose()
    da.graph_data(dataset, ds_config.graph_columns)
    da.freq_domain(dataset, ds_config.columns[:2])

# Split of the data
column_indices = {name: i for i, name in enumerate(dataset.columns)}

# Number of rows in the dataset
num_data, num_features = dataset.shape

