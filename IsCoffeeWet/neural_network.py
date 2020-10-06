def normalize(dataset):
    # TODO: documentation

    # Computes the mean and standard deviation
    ds_mean = dataset.mean()
    ds_std = dataset.std()

    dataset = (dataset - ds_mean) / ds_std

    return dataset


def split_dataset(dataset, config_file):
    # TODO documentation

    # Resets index to add datetime as a normal column
    dataset = dataset.reset_index()

    # Pops the datetime from the dataset. Not use in the NN explicitly
    datetime_index = dataset.pop("Datetime")

    # Accumulates the ratio to use in slices.
    # Validation set is taken from the training set.
    train_ratio = config_file.training - config_file.validation  # e.g. from 0 to 0.5
    validation_ratio = config_file.training  # e.g. from 0.5 to 0.7

    # Divides the dataset
    train_ds = dataset[0:int(config_file.num_data * train_ratio)]
    val_ds = dataset[int(config_file.num_data * train_ratio):
                     int(config_file.num_data * validation_ratio)]
    test_ds = dataset[int(config_file.num_data * validation_ratio):]

    return datetime_index, train_ds, val_ds, test_ds


def load_model(path):
    # TODO: implement function
    # TODO: documentation
    # TODO: test
    return


def save_model(model, path):
    """
    docstring
    """
    # TODO documentation
    # TODO: tests

    model.save(path)
    print("Your model has been save to '{}'".format(path))
    return


"""
# Use tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
                            log_dir=log_dir,
                            histogram_freq=1,
                            embeddings_freq=0,
                            update_freq="epoch"
                        )
"""
