from typing import Dict, Any

from torch import nn
from torch_geometric.data.batch import Batch

from modules.hmpn.abstract.abstract_step import AbstractStep
from modules.hmpn.homogeneous.homogeneous_modules import HomogeneousEdgeModule, \
    HomogeneousNodeModule


class HomogeneousStep(AbstractStep):
    """
         Defines a single MessagePassingLayer that takes a homogeneous observation graph and updates its node and edge
         features using different modules (Edge, Node, Global).
         It first updates the edge-features. The node-features are updated next using the new edge-features
    """

    def __init__(self,
                 stack_config: Dict[str, Any],
                 latent_dimension: int,
                 scatter_reducers,
                 flip_edges_for_nodes: bool = False):
        """
        Initializes the HomogeneousStep, which realizes a single iteration of message passing for a homogeneous graph.
        This message passing layer consists of three modules: Edge, Node, Global, each of which updates the respective
        part of the graph.
        Initializes the HomogeneousStep.
        Args:
            stack_config:
                Configuration of the stack of GNN steps. Should contain keys
                "num_steps" (int),
                "residual_connections" (str: "none", "inner", "outer"),
                "mlp" (Dict[str, Any]). "mlp" is a dictionary for the general configuration of the MLP.
                    which should contain keys
                    "num_layers" (int),
                    "add_output_layer" (bool),
                    "activation_function" (str: "relu", "leakyrelu", "tanh", "silu"), and
                    "regularization" (Dict[str, Any]),
                        which should contain keys
                        "spectral_norm" (bool),
                        "dropout" (float),
                        "latent_normalization" (str: "batch_norm", "layer_norm" or None)
            latent_dimension:
                Dimension of the latent space.
            scatter_reducers:
                reduce operators from torch_scatter. Can be e.g. [scatter_mean]
        """

        super().__init__(stack_config=stack_config,
                         latent_dimension=latent_dimension)

        n_scatter_ops = len(scatter_reducers)

        edge_in_features = 3 * latent_dimension  # edge features, and the two participating nodes
        node_in_features = latent_dimension * (1 + n_scatter_ops)  # node and aggregated incoming edge features

        # edge module
        self.edge_module = HomogeneousEdgeModule(in_features=edge_in_features,
                                                 latent_dimension=latent_dimension,
                                                 stack_config=stack_config,
                                                 scatter_reducers=scatter_reducers)

        # node module
        self.node_module = HomogeneousNodeModule(in_features=node_in_features,
                                                 latent_dimension=latent_dimension,
                                                 stack_config=stack_config,
                                                 scatter_reducers=scatter_reducers,
                                                 flip_edges_for_nodes=flip_edges_for_nodes)

        self.reset_parameters()

        if self.use_layer_norm:
            self._node_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
            self._edge_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
        else:
            self._node_layer_norms = None
            self._edge_layer_norms = None

    def _store_nodes(self, graph: Batch):
        self._old_graph["x"] = graph.x

    def _store_edges(self, graph: Batch):
        self._old_graph["edge_attr"] = graph.edge_attr

    def _add_node_residual(self, graph: Batch):
        graph.__setattr__("x", graph.x + self._old_graph["x"])

    def _add_edge_residual(self, graph: Batch):
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])

    def _node_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("x", self._node_layer_norms(graph.x))

    def _edge_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("edge_attr", self._edge_layer_norms(graph.edge_attr))
