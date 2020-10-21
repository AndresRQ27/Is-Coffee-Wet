# based on the model from S. Bai, J. Z. Kolter, and V. Koltun,
# “An empirical evaluation of generic convolutional and recurrent
# networks for sequence modeling,”

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons.layers as layers_addon


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride=1, dilation=1,
                 padding="causal", dropout=0.2, activation="relu"):
        super(ResidualBlock, self).__init__()

        self.conv1 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation1 = layers.Activation(activation)
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation2 = layers.Activation(activation)
        self.dropout2 = layers.Dropout(dropout)

        # Residual layer of conv 1x1 for feature mapping
        self.residual = layers.Conv1D(filters=filters,
                                      kernel_size=1)

        # Layer to add the residual layer with the rest
        self.add = layers.Add()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.activation1(x)
        if training:
            x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.activation2(x)
        if training:
            x = self.dropout2(x)

        # Adds the residual layer
        y = self.residual(inputs)
        x = self.add([x, y])

        return x
