from torch import nn as nn
import torch
from torch.nn import utils
from src.modules.helper_modules import View
from src.modules.module_util import get_activation_and_regularization_layers, get_layer_size_layout
import numpy as np
from util.types import *


def build_mlp(in_features: Shape,
              feedforward_config: ConfigDict,
              latent_dimension: Optional[int] = None,
              out_features: Optional[int] = None) -> Tuple[nn.ModuleList, int]:
    """
    Builds the discriminator (sub)network. This part of the network accepts some latent space as the input and
    outputs a classification
    Args:
        in_features: Number of input features
        feedforward_config: Dictionary containing the specification for the feedforward network.
          Includes num_layers, max_neurons and a shape of the network. May also include a subdict "regularization"
          for dropout, batchnorm etc.
          also includes "include_last_activation": Whether or not to include an activation after the last (non-output)
            MLP layer.
            Can e.g., be set to false for intermediate building blocks that do not want an activation function before
             processing the data further.
        latent_dimension: Optional latent size of the mlp layers.
          Overwrites "max_neurons" and "network_shape" if provided.
        out_features: Output dimension of the feedforward network
    Returns: A nn.ModuleList representing the discriminator module, and the size of the final activation of this module
    """
    forward_layers = nn.ModuleList()
    if isinstance(in_features, (int, np.int32, np.int64)):  # can use in_features directly
        pass
    elif isinstance(in_features, tuple):
        if len(in_features) == 1:  # only one feature dimension
            in_features = in_features[0]
        else:  # more than one feature dimension. Need to flatten first
            in_features: int = int(np.prod(in_features))
            forward_layers.append(
                View(default_shape=(in_features,), custom_repr="Flattening feedforward input to 1d."))
    else:
        raise ValueError("Unknown type for 'in_features' parameter in Feedforward.py: '{}'".format(type(in_features)))

    activation_function: str = feedforward_config.get("activation_function").lower()
    regularization_config = feedforward_config.get("regularization", {})
    spectral_norm: bool = regularization_config.get("spectral_norm", False)
    include_last_activation = feedforward_config.get("include_last_activation", True)

    previous_shape = in_features

    if latent_dimension is None:
        network_layout = get_layer_size_layout(max_neurons=feedforward_config.get("max_neurons"),
                                               num_layers=feedforward_config.get("num_layers"),
                                               network_shape=feedforward_config.get("network_shape"))
    else:
        network_layout = np.repeat(latent_dimension, feedforward_config.get("num_layers"))

    # build actual network
    for current_layer_num, current_layer_size in enumerate(network_layout):
        # add main linear layer
        if spectral_norm:
            forward_layers.append(utils.spectral_norm(nn.Linear(in_features=previous_shape,
                                                                out_features=current_layer_size)))
        else:
            forward_layers.append(nn.Linear(in_features=previous_shape,
                                            out_features=current_layer_size))

        if include_last_activation or current_layer_num < len(network_layout)-1:
            # add activation function and regularization layers such as batchnorm and dropout
            if activation_function == "relu":
                forward_layers.append(nn.ReLU())
            elif activation_function == "leakyrelu":
                forward_layers.append(nn.LeakyReLU())
            elif activation_function == "elu":
                forward_layers.append(nn.ELU())
            elif activation_function in ["swish", "silu"]:
                forward_layers.append(nn.SiLU())
            elif activation_function == "mish":
                forward_layers.append(nn.Mish())
            elif activation_function == "gelu":
                forward_layers.append(nn.GELU())
            elif activation_function == "tanh":
                forward_layers.append(nn.Tanh())
            else:
                raise ValueError("Unknown activation function '{}'".format(activation_function))

            additional_layers = get_activation_and_regularization_layers(in_features=current_layer_size,
                                                                         regularization_config=regularization_config)
            forward_layers.extend(additional_layers)

        previous_shape = current_layer_size

    if out_features is not None:  # linear embedding to output size
        forward_layers.append(nn.Linear(previous_shape, out_features))
    else:
        out_features = previous_shape

    return forward_layers, out_features


class MLP(nn.Module):
    """
    Feedforward module. Gets some input x and computes an output f(x).
    """

    def __init__(self, in_features: Union[tuple, int],
                 config: ConfigDict,
                 latent_dimension: Optional[int] = None,
                 out_features: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: The input shape for the feedforward network
            out_features: The output dimension for the feedforward network
            latent_dimension: Optional latent size of the mlp layers.
             Overwrites "max_neurons" and "network_shape if provided.
            config: Dict containing information about what kind of feedforward network to build as well
              as how to regularize it (via batchnorm etc.)
        """
        super().__init__()
        self.feedforward_layers, self._out_features = build_mlp(in_features=in_features,
                                                                feedforward_config=config,
                                                                latent_dimension=latent_dimension,
                                                                out_features=out_features)
        self.device = device
        self.to(device)

    @property
    def out_features(self) -> int:
        """
        Size of the last layer of this module
        Returns:

        """
        return self._out_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass for the given input graph
        Args:
            tensor: Some input graph x

        Returns: The processed graph f(x)

        """
        for feedforward_layer in self.feedforward_layers:
            tensor = feedforward_layer(tensor)
        return tensor
