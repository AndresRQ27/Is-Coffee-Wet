import tensorflow as tf
import tensorflow.keras.layers as layers

from IsCoffeeWet.neural_network.models.temporal_convolutional import ResidualBlock
from IsCoffeeWet.neural_network.models.filternet_module import FilternetModule


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


def convolutional_model(filters, kernel_size, pool_size,
                        input_size, output_size, dropout=0.1):
    """
    Function that generates a convolutional model with the shape declared
    by the inputs.

    Parameters
    ----------

    filters: int or list[int]
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
    filters = check_ifint(filters, 3)
    kernel_size = check_ifint(kernel_size, 3)
    pool_size = check_ifint(pool_size, 2)

    # Shape => [batch, input_width, features]
    inputs = layers.Input(shape=input_size)

    x = layers.Conv1D(filters=filters[0],
                      kernel_size=kernel_size[0],
                      activation="relu")(inputs)

    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.MaxPool1D(pool_size=pool_size[0])(x)

    x = layers.Conv1D(filters=filters[1],
                      kernel_size=kernel_size[1],
                      activation="relu")(x)

    if dropout:
        x = layers.Dropout(dropout)(x)

    x = layers.MaxPool1D(pool_size=pool_size[1])(x)

    x = layers.Conv1D(filters=filters[2],
                      kernel_size=kernel_size[2],
                      activation="relu")(x)

    # Shape => [batch, 1,  label_width*label_columns]
    dense = layers.Dense(units=output_size[0] * output_size[1],
                         activation="linear")(x)

    # Shape => [batch, label_width, label_columns]
    outputs = layers.Reshape([output_size[0], output_size[1]])(dense)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="conv_model")

    return model


def temp_conv_model(filters, kernel_size, dilations, input_size,
                    output_size, activation="relu", dropout=0.2):
    """
    Function that generates a temporal convolutional model with the shape
    declared by the inputs.

    Parameters
    ----------

    filters: int or list[int]
        Number of filters to use in the convolutional layers. Can be
        thought as a neuron in a dense layer.
    kernel_size: int or list[int]
        Number of data the filter will see. A number of dimensions
        equals to the kernel size will be subtracted as there's no
        padding in the convolutional layers.
    dilations: int
        Number of dilations to do. This is reflected in the number of layers
        created. Dilation is power to square in each layer (e.g. 1, 2, 4,
        8, ...)
    input_size: tuple[int]
        Shape of the input for the network. It has the number of
        data to receive and the number of features to use.
    output_size: tuple[int]
        Shape of the output for the network. It has the number of
        data to generates and the number of features the predict.
    activation: string or activation.function, optional
        Activation function to use in the residual layers. Can be a string
        if using the name of a generic function provided by tensorflow or a
        custom function from the activation file
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
    filters = check_ifint(filters, dilations)
    kernel_size = check_ifint(kernel_size, dilations)

    # Shape => [batch, input_width, features]
    inputs = layers.Input(shape=input_size)

    # Tune the number of filters in the first convolutional layer
    x = ResidualBlock(filters=filters[0],
                          kernel_size=kernel_size[0],
                          dropout=dropout,
                          activation=activation)(inputs)

    # Generates new residual layers based on the dilatation
    for factor in range(1, dilations):
        # Creates a residual layer
        dilation = 2 ** factor
        x = ResidualBlock(filters=filters[factor],
                              kernel_size=kernel_size[factor],
                              dilation=dilation,
                              dropout=dropout,
                              activation=activation)(x)

    # Shape => [batch, label_width, label_columns]
    output = layers.Dense(units=output_size[1], activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="tcn_model")

    return model


def deep_conv_lstm(filters, kernel_size, pool_size,
                   input_size, output_size, dropout=0.1):
    """
    Function that generates a convolutional-recurrent (lstm) model with the
    shape declared by the inputs.

    Parameters
    ----------
    filters: int or list[int]
        Number of output channels (number of filters for a cnn or the
        dimensionality of the hidden state for an lstm)
    kernel_size: int or list[int]
        Kernel length (only for CNNs).
    pool_size: int or list[int]
        Size of the average pooling layers used by the filternet modules.
        Reduce the length of the series by a factor of length/pool_size.
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
    filters = check_ifint(filters, 5)
    kernel_size = check_ifint(kernel_size, 4)
    pool_size = check_ifint(pool_size, 2)

    # Shape => [batch, input_width, features]
    inputs = layers.Input(shape=input_size)

    conv1 = FilternetModule(w_out=filters[0], flm_type="cnn",
                                s=1, k=kernel_size[0],
                                p_drop=dropout)(inputs)

    conv2 = FilternetModule(w_out=filters[1], flm_type="cnn",
                                s=pool_size[0], k=kernel_size[1],
                                p_drop=dropout)(conv1)

    conv3_1 = FilternetModule(w_out=filters[2], flm_type="cnn",
                                  s=pool_size[1], k=kernel_size[2],
                                  p_drop=dropout)(conv2)

    conv3_2 = FilternetModule(w_out=filters[3], flm_type="cnn",
                                  s=pool_size[1], k=kernel_size[3],
                                  p_drop=dropout)(conv2)

    concat = layers.Concatenate(axis=1)([conv2, conv3_1, conv3_2])

    lstm = FilternetModule(w_out=filters[4], flm_type="lstm",
                               p_drop=dropout)(concat)

    # We're only interested in the channels from the output
    outputs = FilternetModule(w_out=output_size[1], flm_type="cnn",
                                  s=1, k=1, p_drop=dropout, b_bn=False)(lstm)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="deep_conv_lstm_model")

    return model
