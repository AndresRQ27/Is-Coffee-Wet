# based on the model from S. Bai, J. Z. Kolter, and V. Koltun,
# "An empirical evaluation of generic convolutional and recurrent
# networks for sequence modeling"

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons.layers as layers_addon


class ResidualBlock(tf.keras.Model):
    """
    Residual block used in the creation of a temporal convolutional
    network. Uses weight normalization, two 1D-CNN with relu as the default
    activation function and an optional extra 1D-CNN layer to mantain the
    channel size in the output.
    """

    def __init__(self, filters, kernel_size, stride=1, dilation=1,
                 padding="causal", dropout=0.2, activation="relu",
                 use_residual=True):
        """
        Initialization method of a residual block used in the
        construction of a temporal convolutional network.

        Parameters
        ----------
        filters: int
            Number of filters to use in each convolutional layer. With
            out_channel as "valid", this will also be the amount of output
            channels.
        kernel_size: int
            Size of the kernel to use.
        stride: int, optional
            Stride of the convolutional layers (by default is 1). If
            stride>1, dilatation must be 1 or else there will be an error. 
        dilation: int, optional
            Dilatation of the convolutional layers (by default is 1). If
            dilation>1, stride must be 1 or else there will be an error.
        padding: "same", "causal" or "valid"
            Padding to use in the convolutional layers (by default is
            "causal"). If using dilation, it's recommended to mantain the
            default padding, as this is the best for dilations operations.
        dropout: float
            Dropout probability in fractional mode (from 0 to 1, by default is
            0.1). It can be 0 to cancel the nullify the dropout layer. Used to
            regularize the network and improve training dinamics.
        activation: string or function
            Activation function used in the convolutional layers (default
            value is "relu"). Can be passed custom functions, like
            gated-activation, to change it.
        use_residual: bool
            Whether or not to use a residual 1D-CNN for feature transfer from
            input to the output of the residual block. The transfer is done
            by adding a 1D-CNN of kernel_size 1 with the output of the
            residual block.
        """
        super(ResidualBlock, self).__init__()

        # ***Save the values used to re-construct it when loading***
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        self.activation = activation
        self.use_residual = use_residual

        # Creates the first stack of layer: conv1D-weight_norm-activation-dropout
        self.conv1 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation1 = layers.Activation(activation)
        self.dropout1 = layers.Dropout(dropout)

        # Creates the second stack of layer: conv1D-weight_norm-activation-dropout
        self.conv2 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation2 = layers.Activation(activation)
        self.dropout2 = layers.Dropout(dropout)

        if self.use_residual:
            # Residual layer of conv 1x1 for feature mapping
            self.residual = layers.Conv1D(filters=filters,
                                          kernel_size=1)
            # Layer to add the residual layer with the rest
            self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        # Calls the first stack of layers
        x = self.conv1(inputs)
        x = self.activation1(x)
        x = self.dropout1(x, training=training)

        # Calls the second stack of layers
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.dropout2(x, training=training)

        if self.use_residual:
            # Adds the residual layer
            y = self.residual(inputs)
            x = self.add([x, y])

        return x

    def get_config(self):
        """
        Method that gets the values used to create the model. Used for
        loading and saving the model.

        Returns
        -------

        dictionary:
            Dictionary with the configuration of the model
        """
        config = {"filters": self.filters,
                  "kernel_size": self.kernel_size,
                  "stride": self.stride,
                  "dilation": self.dilation,
                  "padding": self.padding,
                  "dropout": self.dropout,
                  "activation": self.activation,
                  "use_residual": self.use_residual}

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
