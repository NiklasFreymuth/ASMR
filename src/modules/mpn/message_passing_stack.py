import abc

import torch.nn as nn

from src.modules.mpn.common.hmpn_util import noop
from src.modules.mpn.message_passing_block import MessagePassingBlock
from util.types import *


class MessagePassingStack(nn.Module, abc.ABC):
    """
    Message Passing module that acts on both node and edge features.
    Internally stacks multiple instances of MessagePassingBlocks.
    """

    def __init__(self,
                 stack_config: ConfigDict,
                 latent_dimension: int):
        """
        Args:
            stack_config: Dictionary specifying the way that the message passing network base should look like.
                num_blocks: how many blocks this stack should have
                use_residual_connections: if the blocks should use residual connections. If True,
              the original inputs will be added to the outputs.
            latent_dimension: the latent dimension of all vectors used in this stack
        """
        super().__init__()
        self._num_blocks: int = stack_config.get("num_blocks")
        self._use_residual_connections: bool = stack_config.get("use_residual_connections")
        self._latent_dimension: int = latent_dimension

        self._old_graph: Optional[Batch] = None

        if self._use_residual_connections:
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_add_residual = self._add_residual
        else:
            self.maybe_store_old_graph = noop
            self.maybe_add_residual = noop
        self._message_passing_blocks: nn.ModuleList = None

        if stack_config.get("use_layer_norm"):
            self._maybe_layer_norm = self._layer_norm
        else:
            self._maybe_layer_norm = lambda *args, **kwargs: None

        self._message_passing_blocks = nn.ModuleList([MessagePassingBlock(stack_config=stack_config,
                                                                          latent_dimension=latent_dimension)
                                                      for _ in range(self._num_blocks)])

        if stack_config.get("use_layer_norm"):
            self._node_layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=latent_dimension)
                                                    for _ in range(self._num_blocks)])
            self._edge_layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=latent_dimension)
                                                    for _ in range(self._num_blocks)])
            self._global_layer_norms = nn.ModuleList([nn.LayerNorm(normalized_shape=latent_dimension)
                                                      for _ in range(self._num_blocks)])

        else:
            self._node_layer_norms = None
            self._edge_layer_norms = None

    def _store_old_graph(self, graph: Batch):
        self._old_graph = {"x": graph.x,
                           "edge_attr": graph.edge_attr,
                           "u": graph.u}

    def _add_residual(self, graph: Batch):
        graph.__setattr__("x", graph.x + self._old_graph["x"])
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])
        graph.__setattr__("u", graph.u + self._old_graph["u"])

    def _layer_norm(self, graph: Batch, layer_position: int) -> None:
        graph.__setattr__("x", self._node_layer_norms[layer_position](graph.x))
        graph.__setattr__("edge_attr", self._edge_layer_norms[layer_position](graph.edge_attr))
        graph.__setattr__("u", self._global_layer_norms[layer_position](graph.u))

    @property
    def num_blocks(self) -> int:
        """
        How many blocks this stack is composed of.
        """
        return self._num_blocks

    @property
    def use_residual_connections(self) -> bool:
        """
        Whether this stack makes use of residual connections or not
        Returns:

        """
        return self._use_residual_connections

    @property
    def latent_dimension(self) -> int:
        """
        Dimensionality of the features that are handled in this stack
        Returns:

        """
        return self._latent_dimension

    def forward(self, graph: Batch) -> None:
        """
        Computes the forward pass for this message passing stack.
        Updates node, edge and global features (new_node_features, new_edge_features, new_global_features)
        for each type as a tuple

        Args: graph or batch of graphs of type torch_geometric.data.Batch

        Returns: None, in-place operation
        """
        for layer_position, message_passing_block in enumerate(self._message_passing_blocks):
            self.maybe_store_old_graph(graph=graph)
            message_passing_block(graph=graph)
            self.maybe_add_residual(graph=graph)
            self._maybe_layer_norm(graph=graph, layer_position=layer_position)

    def __repr__(self):
        if self._message_passing_blocks:
            return f"{self.__class__.__name__}(\n" \
                   f" use_residual_connections={self.use_residual_connections},\n" \
                   f" num_message_passing_blocks={self.num_blocks},\n" \
                   f" first_block={self._message_passing_blocks[0]}\n"
        else:
            return f"{self.__class__.__name__}(\n" \
                   f" use_residual_connections={self.use_residual_connections},\n" \
                   f" num_message_passing_blocks={self.num_blocks}\n"
