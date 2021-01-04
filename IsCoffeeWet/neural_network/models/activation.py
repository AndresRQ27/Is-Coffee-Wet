import tensorflow.keras.layers as layers


# Taken from https://github.com/xadrianzetx/temporal-conv-net-keras/blob/master/tcnet/activations.py

def gated_activation(x):
    """
    Activation function used in the layers of a neural network.

    Parameters
    ----------
    x:

    Returns
    -------

    """
    # Used in PixelCNN and WaveNet
    tanh = layers.Activation('tanh')(x)
    sigmoid = layers.Activation('sigmoid')(x)
    return layers.multiply([tanh, sigmoid])
