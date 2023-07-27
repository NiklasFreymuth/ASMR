import abc

from torch import nn as nn

from src.modules.mpn.message_passing_modules import MessagePassingEdgeModule, \
    MessagePassingNodeModule, MessagePassingGlobalModule
from util.types import *


class MessagePassingBlock(nn.Module, abc.ABC):
    """
         Defines a single MessagePassingLayer that takes an observation graph and updates its node and edge
         features using different modules (Edge, Node, Global).
         It first updates the edge-features. The node-features are updated next using the new edge-features. Finally,
         it updates the global features using the new edge- & node-features. The updates are done through MLPs.
    """

    def __init__(self,
                 stack_config: ConfigDict,
                 latent_dimension: int):
        """
        Initializes the MessagePassingBlock, which realizes a single iteration of message passing.
        This message passing layer consists of three modules: Edge, Node, Global, each of which updates the respective
        part of the graph.
        Args:
            stack_config:
                Configuration of the stack of GNN blocks. Should contain keys
                "num_blocks" (int),
                "use_residual_connections" (bool),
                "mlp" (ConfigDict). "mlp" is a dictionary for the general configuration of the MLP.
                    which should contain keys
                    "num_layers" (int),
                    "activation_function" (str: "relu", "leakyrelu", "tanh", "silu"), and
            latent_dimension:
                Dimension of the latent space.
        """
        super().__init__()
        self._latent_dimension = latent_dimension

        edge_in_features = 3 * latent_dimension  # edge features, and the two participating nodes
        node_in_features = latent_dimension * 2  # node features and the aggregated incoming edge features

        edge_in_features += latent_dimension
        node_in_features += latent_dimension
        self.global_module = MessagePassingGlobalModule(
            in_features=latent_dimension * 3,  # global features, reduced edge features and reduced node features
            latent_dimension=latent_dimension,
            stack_config=stack_config
        )

        # edge module
        self.edge_module = MessagePassingEdgeModule(in_features=edge_in_features,
                                                    latent_dimension=latent_dimension,
                                                    stack_config=stack_config)

        # node module
        self.node_module = MessagePassingNodeModule(in_features=node_in_features,
                                                    latent_dimension=latent_dimension,
                                                    stack_config=stack_config)

        self.reset_parameters()

    def reset_parameters(self):
        """
        This resets all the parameters for all modules
        """
        for item in [self.node_module, self.edge_module, self.global_module]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, graph: Data):
        """
        Computes the forward pass for this block/meta layer inplace

        Args:
            graph: Data object of pytorch geometric. Represents a (batch of) of graph(s)

        Returns:
            None
        """
        self.edge_module(graph)
        self.node_module(graph)
        self.global_module(graph)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n " \
               f"edge_module={self.edge_module},\n" \
               f"node_module={self.node_module},\n " \
               f"global_module={self.global_module}\n"
