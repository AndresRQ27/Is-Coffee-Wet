# Based on the module described in https://doi.org/10.3390/s20092498
# by Robert D. Chambers and Nathanael C. Yoder

import tensorflow as tf
import tensorflow.keras.layers as layers


class FilternetModule(tf.keras.Model):
    """
    Filternet module used in the construction of a deep convolutional network.
    Can be used to create a convolutional layer or a recurrent layer (LSTM),
    in which both cases can use a dropout layer, average pooling layer and/or
    batch normalization layer. Different typers of layers can be added by
    adding more processing options in the "flm_type" parameter.
    """

    def __init__(self, w_out, flm_type="cnn", s=1, k=5, p_drop=0.1, b_bn=True):
        """
        Initialization method for the filternet module. By default, the
        module always creates a convolutional filternet module.

        Parameters
        ----------
        w_out: int
            Number of output channels (number of filters for a cnn or the
            dimensionality of the hidden state for an lstm)
        flm_type: string, optional
            Type of the primary trainable sub-layer. Use "cnn" for 1-D CNN or
            "lstm" for a bi-directional LSTM.
        s: int, optional
            Stride ratio (by default 1). If s>1, then a 1-D average-pooling with
            stride s and pooling kernel of length s reduces the output length by
            a factor of s.
        k: int, optional
            Kernel length (only for CNNs, by default is 5).
        p_drop: float, optional
            Dropout probability in fractional mode (from 0 to 1, by default is
            0.1). It can be 0 to cancel the nullify the dropout layer. Used to
            regularize the network and improve training dinamics.
        b_bn: bool, optional
            1D batch normalization layer. Used to regularize the network and
            improve training dinamics.

        Notes
        -----
        Layers based on CNN use ReLu as activation for default. LSTM use tanh
        as activation and sigmoid as recurrent activation; it's recommended
        that the LSTM activations aren't change to use cuDNN implementation
        by tensorflow.
        """
        super(FilternetModule, self).__init__()

        # ***Save the values used to re-construct it when loading***
        self.w_out = w_out
        self.flm_type = flm_type
        self.s = s
        self.k = k
        self.p_drop = p_drop
        self.b_bn = b_bn

        # Creates the dropout layer
        self.dropout = layers.Dropout(rate=p_drop)

        if flm_type == "lstm":
            # Creates a LSTM layer
            self.flm_type = layers.LSTM(units=w_out, return_sequences=True)
        else:
            # Creates a 1D-CNN layer
            self.flm_type = layers.Conv1D(
                filters=w_out, kernel_size=k, padding="same")

        # Creates an average pooling layer
        self.pool = layers.AveragePooling1D(pool_size=s, strides=s)

        if b_bn:
            # Creates a batch normalization layer
            self.batch_norm = layers.BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        # Use the dropout layer with the inputs,
        x = self.dropout(inputs, training=training)

        # LSTM/CNN layer
        x = self.flm_type(x)

        # Use the pooling layer
        x = self.pool(x)

        if self.b_bn:
            # Use the batch normalization layer
            x = self.batch_norm(x, training=training)

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
        config = {"w_out": self.w_out,
                  "flm_type": self.flm_type,
                  "s": self.s,
                  "k": self.k,
                  "p_drop": self.p_drop,
                  "b_bn": self.b_bn}

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
