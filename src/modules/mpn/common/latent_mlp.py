import numpy as np
import torch
from torch import nn as nn

from util.types import *


class SwiGlu(nn.Module):
    def __init__(self, in_features, out_features):
        super(SwiGlu, self).__init__()
        self.l1 = nn.Linear(in_features, out_features)
        self.l2 = nn.Linear(in_features, out_features)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGlu layer
        Args:
            x: The input tensor of shape (batch_size, in_features)

        Returns: The output tensor of shape (batch_size, out_features)

        """
        h1 = self.l1(x)
        return self.act(self.l2(x)) * h1


class SaveBatchnorm1d(nn.BatchNorm1d):
    """
    Custom Layer to deal with Batchnorms for 1-sample inputs
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Normalization layer
        Args:
            tensor: The input graph

        Returns: The normalized graph

        """
        batch_size = tensor.shape[0]
        if batch_size == 1 or len(tensor.shape) == 1:
            return tensor
        else:
            return super().forward(input=tensor)


def build_latent_mlp(*,
                     in_features: int,
                     config: ConfigDict,
                     latent_dimension: int) -> nn.ModuleList:
    """
    Builds a latent MLP, i.e., a network that gets as input a number of latent features, and then feeds these features
     through a number of fully connected layers and activation functions.
    Args:
        in_features: Number of input features
        config: Dictionary containing the specification for the mlp network.
          Includes
          * num_layers: how many layers+activations to build
          * activation_function: which activation function to use
        latent_dimension: Latent size of the mlp layers.
    Returns: A nn.ModuleList representing the MLP module

    """
    mlp_layers = nn.ModuleList()

    activation_function: str = config.get("activation_function").lower()
    network_layout = np.repeat(latent_dimension, config.get("num_layers"))

    previous_shape = in_features

    # build actual network
    for current_layer_num, current_layer_size in enumerate(network_layout):
        # add main linear layer
        mlp_layers.append(nn.Linear(in_features=previous_shape,
                                    out_features=current_layer_size))

        # activation function
        if activation_function == "relu":
            mlp_layers.append(nn.ReLU())
        elif activation_function == "leakyrelu":
            mlp_layers.append(nn.LeakyReLU())
        elif activation_function == "tanh":
            mlp_layers.append(nn.Tanh())
        else:
            raise ValueError("Unknown activation function '{}'".format(activation_function))

        previous_shape = current_layer_size
    return mlp_layers


class LatentMLP(nn.Module):
    """
    Feedforward module. Gets some input x and computes an output f(x).
    """

    def __init__(self, *, in_features: int,
                 latent_dimension: int,
                 config: ConfigDict):
        """
        Initializes a new encoder module that consists of multiple different layers that together encode some
        input into a latent space
        Args:
            in_features: The input shape for the feedforward network
            latent_dimension: Latent size of the mlp layers.
            config: Dict containing information about how to build and regularize the MLP (via batchnorm etc.)
        """
        super().__init__()
        self.mlp_layers = build_latent_mlp(in_features=in_features,
                                           config=config,
                                           latent_dimension=latent_dimension)
        self.in_features = in_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass for the given input tensor
        Args:
            tensor: Some input tensor x

        Returns: The processed tensor f(x)

        """
        for mlp_layer in self.mlp_layers:
            tensor = mlp_layer(tensor)

        return tensor
