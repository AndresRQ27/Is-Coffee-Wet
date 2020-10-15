import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons.layers as layers_addon

from IsCoffeeWet import activation


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=4, stride=1, dilation=1,
                 padding="causal", dropout=0.2):
        super(ResidualBlock, self).__init__()

        self.conv1 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation1 = layers.Activation(activation.gated_activation)
        self.dropout1 = layers.Dropout(dropout)

        self.conv2 = layers_addon.WeightNormalization(layers.Conv1D(filters=filters,
                                                                    kernel_size=kernel_size,
                                                                    strides=stride,
                                                                    padding=padding,
                                                                    dilation_rate=dilation))
        self.activation2 = layers.Activation(activation.gated_activation)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.activation1(x)
        if training:
            x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.activation2(x)
        if training:
            x = self.dropout2(x)
        return x
