import abc
from typing import Dict, Optional, Any

from torch import nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data

from modules.hmpn.common.hmpn_util import noop


class AbstractStep(nn.Module, abc.ABC):
    """
     Defines a single Message Passing Step that takes an observation graph and updates its node and edge
     features using different modules described in implementations of this abstract class.
     It first updates the edge-features. The node-features are updated next using the new edge-features.
    """

    def __init__(self,
                 stack_config: Dict[str, Any],
                 latent_dimension: int):
        """
        Args:
            latent_dimension: Dimensionality of the latent space
            stack_config: Dictionary specifying the way that the gnn base should look like.
                num_steps: how many steps this stack should have
                residual_connections: Which kind of residual connections to use
        """
        super().__init__()
        self._latent_dimension = latent_dimension

        residual_connections: Optional[str] = stack_config.get("residual_connections")
        residual_connections = residual_connections.lower() if residual_connections is not None else None
        layer_norm: Optional[str] = stack_config.get("layer_norm")
        layer_norm = layer_norm.lower() if layer_norm is not None else None
        self.use_layer_norm = layer_norm in ["outer", "inner"]

        self.edge_module: Optional[nn.Module] = None
        self.node_module: Optional[nn.Module] = None

        self._old_graph: Dict[str, Any] = {}

        self._initialize_maybes()

        if residual_connections == "outer":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_outer_residual = self._add_graph_residuals
        elif residual_connections == "inner":
            self.maybe_store_old_graph = self._store_old_graph
            self.maybe_inner_node_residual = self._add_node_residual
            self.maybe_inner_edge_residual = self._add_edge_residual

        if layer_norm == "outer":
            self.maybe_outer_layer_norm = self._graph_layer_norm
        elif layer_norm == "inner":
            self.maybe_inner_node_layer_norm = self._node_layer_norm
            self.maybe_inner_edge_layer_norm = self._edge_layer_norm

    def _initialize_maybes(self):
        self.maybe_store_old_graph = noop

        self.maybe_outer_residual = noop
        self.maybe_inner_node_residual = noop
        self.maybe_inner_edge_residual = noop

        self.maybe_outer_layer_norm = noop
        self.maybe_inner_node_layer_norm = noop
        self.maybe_inner_edge_layer_norm = noop

    def _store_old_graph(self, graph: Batch):
        self._store_nodes(graph)
        self._store_edges(graph)

    def _add_graph_residuals(self, graph: Batch):
        self._add_node_residual(graph)
        self._add_edge_residual(graph)

    def _graph_layer_norm(self, graph: Batch) -> None:
        self._node_layer_norm(graph)
        self._edge_layer_norm(graph)

    def _store_nodes(self, graph: Batch):
        raise NotImplementedError("'_store_nodes' not implemented for AbstractStack")

    def _store_edges(self, graph: Batch):
        raise NotImplementedError("'_store_edges' not implemented for AbstractStack")

    def _add_node_residual(self, graph: Batch):
        raise NotImplementedError("'_add_inner_node_residual' not implemented for AbstractStack")

    def _add_edge_residual(self, graph: Batch):
        raise NotImplementedError("'_add_inner_edge_residual' not implemented for AbstractStack")

    def _node_layer_norm(self, graph: Batch) -> None:
        raise NotImplementedError("'_node_layer_norm' not implemented for AbstractStack")

    def _edge_layer_norm(self, graph: Batch) -> None:
        raise NotImplementedError("'_edge_layer_norm' not implemented for AbstractStack")

    def reset_parameters(self):
        """
        This resets all the parameters for all modules
        """
        for item in [self.node_module, self.edge_module]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, graph: Data):
        """
        Computes the forward pass for this step/meta layer inplace

        Args:
            graph: Data object of pytorch geometric. Represents a (batch of) of homogeneous graph(s)

        Returns:
            None
        """
        self.maybe_store_old_graph(graph=graph)

        self.edge_module(graph)
        self.maybe_inner_edge_residual(graph=graph)
        self.maybe_inner_edge_layer_norm(graph=graph)

        self.node_module(graph)
        self.maybe_inner_node_residual(graph=graph)
        self.maybe_inner_node_layer_norm(graph=graph)

        self.maybe_outer_residual(graph=graph)
        self.maybe_outer_layer_norm(graph=graph)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n " \
               f"edge_module={self.edge_module},\n" \
               f"node_module={self.node_module},\n "
