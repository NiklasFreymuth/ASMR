import numpy as np

from src.modules.helper_modules import SaveBatchnorm1d
from util.types import *
from torch import nn


def get_activation_and_regularization_layers(in_features: int,
                                             regularization_config: ConfigDict) -> nn.ModuleList:
    """
    Creates a number of activation and regularization layers to append to any feedforward network layer.
    These are usually a single activation and potentially dropout as well as some regularization like
    batch_norm or layer_norm.
    Args:
        in_features: Number of input dimension for the layer(s). Relevant for latent normalizations like batch_norm
        regularization_config: Configuration describing which kinds of activations and regularizations to apply

    Returns: A small moduleList containing the activation and regularization layer(s)
    """
    new_modules = nn.ModuleList()
    dropout = regularization_config.get("dropout")
    latent_normalization = regularization_config.get("latent_normalization")

    # add latent normalization method
    if latent_normalization in ["batch", "batch_norm"]:
        new_modules.append(SaveBatchnorm1d(num_features=in_features))
    elif latent_normalization in ["layer", "layer_norm"]:
        new_modules.append(nn.LayerNorm(normalized_shape=in_features))

    # add dropout
    if dropout:
        new_modules.append(nn.Dropout(p=dropout))

    return new_modules


def get_layer_size_layout(max_neurons: int, num_layers: int, network_shape: str = "block") -> np.ndarray:
    """
    Creates a list of neurons per layer to create a feedforward network with
    Args:
        max_neurons: Maximum number of neurons for a single layer
        num_layers: Number of layers of the network.
        network_shape: Shape of the network. May be
            "=" or "block" for a network where every layer has the same size (=max_neurons)
                E.g., {num_layers=3, max_neurons=32} will result in [32, 32, 32]
            ">" or "contracting" for a network that gets exponentially smaller.
                E.g., {num_layers=3, max_neurons=32} will result in [32, 16, 8]
            "<" or "expanding" for a network that gets exponentially bigger.
                E.g., {num_layers=3, max_neurons=32} will result in [8, 16, 32]
            "><" or "hourglass" for a network with a bottleneck in the middle.
                E.g., {num_layers=3, max_neurons=32} will result in [32, 16, 32]
            "<>" or "rhombus" for a network that expands towards the middle and then contracts again
                E.g., {num_layers=3, max_neurons=32} will result in [16, 32, 16]
            Overall, the network_shape heavily influences the total number of neurons as well as weights between them.
            We can thus order parameters by network_shape as block>=hourglass>=rhombus>=contracting/expanding, where
            the first two equalities only hold for networks with less than 3 layers, and the last equality holds for
            exactly 1 layer (in which all shapes are equal).
            In case of an even number of layers, "rhombus" and "hourglass" will repeat the "middle" layer once. I.e.,
            {num_layers=4, max_neurons=32, network_shape="rhombus"} will result in an array [16, 32, 32, 16]

    Returns: A 1d numpy array of length num_layers. Each entry specifies the number of neurons to use in that layer.
        The maximum number of neurons will always be equal to max_neurons. Depending on the shape, the other
        layers will have max_neurons/2^i neurons, where i depends on the number of "shrinking" layers between
        the current layer and one of maximum size. In other words, the smaller layers shrink exponentially.
    """
    assert isinstance(max_neurons, int), \
        "Need to have an integer number of maximum neurons. Got '{}' of type '{}'".format(max_neurons,
                                                                                          type(max_neurons))
    if num_layers == 0:
        return np.array([])  # empty list, i.e., no network
    if network_shape in ["=", "==", "block"]:
        return np.repeat(max_neurons, num_layers)
    elif network_shape in [">", "contracting"]:
        return np.array([int(np.maximum(1, max_neurons // (2 ** distance)))
                         for distance in range(num_layers)])
    elif network_shape in ["<", "expanding"]:
        return np.array([int(np.maximum(1, max_neurons // (2 ** current_layer)))
                         for current_layer in reversed(range(num_layers))])
    elif network_shape in ["><", "hourglass"]:
        return np.array([int(np.maximum(1, max_neurons // (2 ** distance)))
                         for distance in list(range((num_layers + 1) // 2)) + list(reversed(range(num_layers // 2)))])
    elif network_shape in ["<>", "rhombus"]:
        # we want a way to produce a list like [n, n-1, ..., 1, 0, 1, ..., n] that works for both num_layers = 2n and
        # num_layers=2n+1.
        return np.array([int(np.maximum(1, max_neurons // (2 ** distance)))
                         for distance in
                         list(reversed(range((num_layers + 1) // 2))) +
                         list(range((num_layers + 1) // 2))[num_layers % 2:]])
    else:
        raise ValueError("Unknown network_shape '{}'. Eligible shapes are = ('block'), > ('contracting'), "
                         "< ('expanding'), >< ('hourglass'), <> ('rhombus')".format(network_shape))
