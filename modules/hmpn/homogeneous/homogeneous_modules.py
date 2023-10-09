from typing import Dict, Any, List, Callable

import torch
from torch_geometric.data.batch import Batch

from modules.hmpn.abstract.abstract_modules import AbstractMetaModule
from modules.hmpn.common.latent_mlp import LatentMLP


class HomogeneousMetaModule(AbstractMetaModule):
    """
    Base class for the homogeneous modules used in the GNN.
    They are used for updating node- and edge features.
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: Dict[str, Any],
                 scatter_reducers: List[Callable]):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            stack_config: Dictionary specifying the way that the gnn base should look like
            scatter_reducers: How to aggregate over the nodes/edges. Can for example be [torch.scatter_mean]
        """
        super().__init__(scatter_reducers=scatter_reducers)
        mlp_config = stack_config.get("mlp")
        self._mlp = LatentMLP(in_features=in_features,
                              latent_dimension=latent_dimension,
                              config=mlp_config)

        self.in_features = in_features
        self.latent_dimension = latent_dimension


class HomogeneousEdgeModule(HomogeneousMetaModule):
    """
    Module for computing edge updates of a step on a homogeneous message passing GNN. Edge inputs are concatenated:
    Its own edge features, the features of the two participating nodes and optionally,
    """

    def forward(self, graph: Batch):
        """
        Compute edge updates for the edges of the Module for homogeneous graphs in-place.
        An updated representation of the edge attributes for all edge_types is written back into the graph
        Args:
            graph: Data object of pytorch geometric. Represents a batch of homogeneous graphs
        Returns: None
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)

        graph.__setattr__("edge_attr", self._mlp(aggregated_features))


class HomogeneousNodeModule(HomogeneousMetaModule):
    """
    Module for computing node updates of a step on a homogeneous message passing GNN. Node inputs are concatenated:
    Its own Node features, the reduced features of all incoming edges and optionally,
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: Dict[str, Any],
                 scatter_reducers: List[Callable],
                 flip_edges_for_nodes: bool = False):

        super(HomogeneousNodeModule, self).__init__(in_features=in_features,
                                                    latent_dimension=latent_dimension,
                                                    stack_config=stack_config,
                                                    scatter_reducers=scatter_reducers)

        # use the source indices for feat. aggregation if edges shall be flipped
        if flip_edges_for_nodes:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[0]
        else:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[1]

    def forward(self, graph: Batch):
        """
        Compute updates for each node feature vector
            graph: Batch object of pytorch_geometric.data, represents a batch of homogeneous graphs
        Returns: None. In-place operation
        """
        src_indices, dest_indices = graph.edge_index
        scatter_edge_indices = self._get_edge_indices((src_indices, dest_indices))

        aggregated_edge_features = self.multiscatter(features=graph.edge_attr, indices=scatter_edge_indices,
                                                     dim=0, dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)

        # update
        graph.__setattr__("x", self._mlp(aggregated_features))
