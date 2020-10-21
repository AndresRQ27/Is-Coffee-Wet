import tensorflow as tf
import tensorflow.keras.layers as layers

from IsCoffeeWet import activation
from IsCoffeeWet import temporal_convolutional as tcn


def check_ifint(value, size):
    """
    Function that checks the value size.
    - If it's an int, makes it a list of the desire size.
    - If it's a list smaller than size, replicates the last value to complete it
    - If it's a list bigger than the size, trims it to the desire size

    Parameters
    ----------

    value: int or list[int]
            Value to analyze the shape.
    size: int
          Size of the value we want.

    Returns
    -------
    list[int]
        List with its size fixed.
    """
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
                        input_size, output_size, dropout=0.1,
                        graph_path=None):
    """
    Function that generates a convolutional model with the shape declared
    by the inputs.

    Parameters
    ----------

    filter_size: int or list[int]
        Number of filters to use in the convolutional layers. Can be
        thought as a neuron in a dense layer.
    kernel_size: int or list[int]
        Number of data the filter will see. A number of dimensions
        equals to the kernel size will be subtracted as there's no
        padding in the convolutional layers.
    pool_size: int or list[int]
        Size of the max pooling layer. Helps reduce the dimensionality
        of big inputs instead of making a big chain of un-padded
        convolutional layers to achieve the desire dimension.
    input_size: tuple[int]
        Shape of the input for the network. It has the number of
        data to receive and the number of features to use.
    output_size: tuple[int]
        Shape of the output for the network. It has the number of
        data to generates and the number of features the predict.
    dropout: float, optional
        Float between 0 and 1. Fraction of the input units to drop. By
        default, it's 0.2
    graph_path: string, optional
        String with the path to save the graph of the model created. By
        default, its `None` so no graph will be generated.

    Returns
    -------

    tensorflow.keras.Model
        Created model of the network.
    """

    # Check if the inputs are numbers. Convert them to lists
    filter_size = check_ifint(filter_size, 3)
    kernel_size = check_ifint(kernel_size, 3)
    pool_size = check_ifint(pool_size, 2)

    # Shape => [batch, input_width, features]
    inputs = layers.Input(shape=input_size)

    x = layers.Conv1D(filters=filter_size[0],
                      kernel_size=kernel_size[0],
                      activation="relu")(inputs)

    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.MaxPool1D(pool_size=pool_size[0])(x)

    x = layers.Conv1D(filters=filter_size[1],
                      kernel_size=kernel_size[1],
                      activation="relu")(x)

    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.MaxPool1D(pool_size=pool_size[1])(x)

    x = layers.Conv1D(filters=filter_size[2],
                      kernel_size=kernel_size[2],
                      activation="relu")(x)

    # Shape => [batch, 1,  label_width*label_columns]
    dense = layers.Dense(units=output_size[0] * output_size[1],
                         activation="linear")(x)

    # Shape => [batch, label_width, label_columns]
    outputs = layers.Reshape([output_size[0], output_size[1]])(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="conv_model")

    if graph_path:
        # ! Printing may require extra libraries
        tf.keras.utils.plot_model(model, graph_path, show_shapes=True)

    return model


def temp_conv_model(filter_size, kernel_size, dilations, input_size,
                    output_size, dropout=0.2, graph_path=None):
    # TODO: documentation
    # Check if the inputs are numbers. Convert them to lists
    filter_size = check_ifint(filter_size, dilations)
    kernel_size = check_ifint(kernel_size, dilations)

    # Shape => [batch, input_width, features]
    inputs = layers.Input(shape=input_size)

    # Tune the number of filters in the first convolutional layer
    x = tcn.ResidualBlock(filters=filter_size[0],
                          kernel_size=kernel_size[0],
                          dropout=dropout)(inputs)

    # Generates new residual layers based on the dilatation
    for factor in range(1, dilations):
        # Creates a residual layer
        dilation = 2 ** factor
        x = tcn.ResidualBlock(filters=filter_size[factor],
                              kernel_size=kernel_size[factor],
                              dilation=dilation,
                              dropout=dropout)(x)

    # Shape => [batch, label_width, label_columns]
    output = layers.Dense(units=output_size[1])(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="tcn_model")

    if graph_path:
        # ! Printing may require extra libraries
        tf.keras.utils.plot_model(model, graph_path, show_shapes=True)

    return model
