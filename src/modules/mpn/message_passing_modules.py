import abc

import torch
from torch_scatter import scatter_mean

from src.modules.mpn.common.latent_mlp import LatentMLP
from util.types import *


class MessagePassingMetaModule(torch.nn.Module, abc.ABC):
    """
    Base class for the modules used in the GNN.
    They are used for updating node-, edge-, and global features.
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 stack_config: ConfigDict):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            stack_config: Dictionary specifying the way that the gnn base should look like
        """
        super().__init__()
        mlp_config = stack_config.get("mlp")
        self._mlp = LatentMLP(in_features=in_features,
                              latent_dimension=latent_dimension,
                              config=mlp_config)

        self.in_features = in_features
        self.latent_dimension = latent_dimension

    def _concat_global(self, features, graph):
        raise NotImplementedError(f"Module {type(self)} needs to implement _concat_global(self,features,graph)")


class MessagePassingEdgeModule(MessagePassingMetaModule):
    """
    Module for computing edge updates of a block on a message passing GNN. Edge inputs are concatenated:
    Its own edge features, the features of the two participating nodes and global features.
    """

    def forward(self, graph: Batch):
        """
        Compute edge updates/messages.
        An updated representation of the edge attributes for all edge_types is written back into the graph
        Args:
            graph: Data object of pytorch geometric.
        Returns: None
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)
        aggregated_features = self._concat_global(aggregated_features, graph)

        graph.__setattr__("edge_attr", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        source_indices, _ = graph.edge_index
        global_features = graph.u[graph.batch[source_indices]]
        return torch.cat([aggregated_features, global_features], 1)


class MessagePassingNodeModule(MessagePassingMetaModule):
    """
    Module for computing node updates/messages. Node inputs are concatenated:
    Its own Node features, the reduced features of all incoming edges and global features.
    """

    def forward(self, graph: Batch):
        """
        Compute updates for each node feature vector
            graph: Batch object of pytorch_geometric.data
        Returns: None. In-place operation
        """
        _, dest_indices = graph.edge_index
        aggregated_edge_features = scatter_mean(graph.edge_attr,
                                                dest_indices,
                                                dim=0,
                                                dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)
        aggregated_features = self._concat_global(aggregated_features, graph)

        # update
        graph.__setattr__("x", self._mlp(aggregated_features))

    def _concat_global(self, aggregated_features, graph):
        """
        computation and concatenation of global features
        Args:
            aggregated_features: so-far aggregated features
            graph: pytorch_geometric.data.Batch object

        Returns: aggregated_features with the global features appended
        """
        global_features = graph.u[graph.batch]
        return torch.cat([aggregated_features, global_features], dim=1)


class MessagePassingGlobalModule(MessagePassingMetaModule):
    """
    Module for computing updates of global features of a block on a message passing GNN.
    Global feature network inputs are concatenated: Its own global features, the reduced features of all edges,
    and the reduced features of all nodes.
    """

    def forward(self, graph: Batch):
        """
        Compute updates for the global feature vector
            graph: Batch object of pytorch_geometric.data
        Returns: None. in-place operation.
        """

        reduced_node_features = scatter_mean(graph.x,
                                             graph.batch,
                                             dim=0,
                                             dim_size=graph.u.shape[0])
        source_indices, _ = graph.edge_index
        reduced_edge_features = scatter_mean(graph.edge_attr,
                                             graph.batch[source_indices],
                                             dim=0,
                                             dim_size=graph.u.shape[0])
        aggregated_features = torch.cat([reduced_edge_features, reduced_node_features, graph.u], dim=1)
        graph.__setattr__("u", self._mlp(aggregated_features))
