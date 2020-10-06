import tensorflow as tf


def check_ifint(value, size):
    if isinstance(value, int):
        value = [value] * size

    elif len(value) < size:
        # Completes the list by duplicating the las value
        difference = abs(size - len(value))
        value.extend([value[-1] * difference])

    elif size < len(value):
        # Trims the list
        value = value[:size]

    return value


def convolutional_model(filter_size, kernel_size, pool_size,
                        input_size, output_size, dropout=0.2,
                        graph_path=None):
    """
    docstring
    """
    # TODO

    filter_size = check_ifint(filter_size, 3)
    kernel_size = check_ifint(kernel_size, 3)
    pool_size = check_ifint(pool_size, 2)

    inputs = tf.keras.layers.Input(shape=input_size)

    x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                               kernel_size=kernel_size.pop(0),
                               activation="relu")(inputs)

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.MaxPool1D(pool_size=pool_size.pop(0))(x)

    x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                               kernel_size=kernel_size.pop(0),
                               activation="relu")(x)

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.MaxPool1D(pool_size=pool_size.pop(0))(x)

    x = tf.keras.layers.Conv1D(filters=filter_size.pop(0),
                               kernel_size=kernel_size.pop(0),
                               activation="relu")(x)

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    # Shape => [batch, 1,  label_width*features]
    dense = tf.keras.layers.Dense(units=output_size[0] * output_size[1],
                                  activation="linear")(x)

    # Shape => [batch, label_width, features]
    outputs = tf.keras.layers.Reshape([output_size[0], output_size[1]])(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv_model")

    if graph_path:
        # ! Printing may require extra libraries
        tf.keras.utils.plot_model(model, graph_path, show_shapes=True)

    return model
