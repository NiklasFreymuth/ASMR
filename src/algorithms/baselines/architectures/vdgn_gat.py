from typing import Dict, Any, Optional, Type

from modules.hmpn import AbstractMessagePassingBase
from modules.hmpn.common.hmpn_util import unpack_homogeneous_features
from modules.hmpn.homogeneous.homogeneous_graph_assertions import HomogeneousGraphAssertions
from modules.hmpn.homogeneous.homogeneous_input_embedding import HomogeneousInputEmbedding
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv


class VDGNGATStep(nn.Module):
    def __init__(self, latent_dimension):
        super().__init__()
        heads = 2
        self.gat_conv = GATConv(
            latent_dimension,  # node latent dimension
            int(latent_dimension / heads),
            heads=heads,
            add_self_loops=False,
            edge_dim=latent_dimension,
        )
        self.node_layer_norm1 = nn.LayerNorm(latent_dimension)
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dimension, latent_dimension),
            nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dimension, latent_dimension),
            nn.ReLU(),
        )
        self.node_layer_norm2 = nn.LayerNorm(latent_dimension)
        self.edge_layer_norm = nn.LayerNorm(latent_dimension)

    def forward(self, graph: Batch):
        # in-place modification of graph
        out_x = self.gat_conv(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
        graph.x = out_x + graph.x  # residual connection
        graph.x = self.node_layer_norm1(graph.x)  # layer norm

        graph.x = self.node_mlp(graph.x) + graph.x  # node mlp + residual connection
        graph.edge_attr = self.edge_mlp(graph.edge_attr) + graph.edge_attr  # edge mlp + residual connection

        graph.x = self.node_layer_norm2(graph.x)  # layer norm
        graph.edge_attr = self.edge_layer_norm(graph.edge_attr)  # layer norm


class VDGNGAT(AbstractMessagePassingBase):

    def __init__(self, *, in_node_features: int,
                 in_edge_features: int,
                 latent_dimension: int,
                 base_config: Dict[str, Any],
                 unpack_output: bool = True,
                 node_name: str = "node",
                 device: str = None):
        """
        Args:
            in_node_features:
                Node feature input size for a homogeneous graph.
                Node features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            in_edge_features:
                Edge feature input size for a homogeneous graph.
                Edge features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            latent_dimension:
                Latent dimension of the network. All modules internally operate with latent vectors of this dimension
            unpack_output:
                If true, the output of the forward pass is unpacked into a tuple of (node_features, edge_features).
                If false, the output of the forward pass is the raw graph.
            node_name:
                Name of the node. Used to convert the output of the forward pass to a dictionary
        """
        create_graph_copy = base_config.get("create_graph_copy", True)
        assert_graph_shapes = base_config.get("assert_graph_shapes")
        stack_config = base_config.get("stack")
        embedding_config = base_config.get("embedding")
        scatter_reduce_strs = base_config.get("scatter_reduce")
        flip_edges_for_nodes = base_config.get('flip_edges_for_nodes', False)
        super().__init__(in_node_features=in_node_features,
                         in_edge_features=in_edge_features,
                         latent_dimension=latent_dimension,
                         embedding_config=embedding_config,
                         scatter_reduce_strs=scatter_reduce_strs,
                         unpack_output=unpack_output,
                         create_graph_copy=create_graph_copy,
                         assert_graph_shapes=assert_graph_shapes
                         )
        assert isinstance(stack_config, dict), f"Expected stack_config to be a dict, but got {type(stack_config)}"
        assert embedding_config is None, f"Expected embedding_config to be None, but got {type(embedding_config)}"
        assert flip_edges_for_nodes is False, f"Expected flip_edges_for_nodes to be False got {type(flip_edges_for_nodes)}"

        if isinstance(scatter_reduce_strs, list):
            assert len(scatter_reduce_strs) == 1, (f"Expected scatter_reduce_strs to be a list of length 1, "
                                                   f"got {scatter_reduce_strs}")
            scatter_reduce_strs = scatter_reduce_strs[0]
        assert scatter_reduce_strs in ["sum",
                                       "mean"], (f"Expected scatter_reduce_strs to be 'sum' or 'mean', "
                                                 f"got {scatter_reduce_strs}")
        self._node_name = node_name
        self.vdgn_gat_steps = nn.ModuleList([VDGNGATStep(latent_dimension=latent_dimension)
                                             for _ in range(stack_config["num_steps"])])
        self._outer_repeats = 2
        self.to(device)

    def _get_assertions(self) -> Type[HomogeneousGraphAssertions]:
        return HomogeneousGraphAssertions

    @staticmethod
    def _get_input_embedding_type() -> Type:
        return HomogeneousInputEmbedding

    def unpack_features(self, graph: Batch) -> Batch:
        return unpack_homogeneous_features(graph, node_name=self._node_name)

    def forward(self, graph: Batch) -> Batch:
        """
        Performs a forward pass through the Graph Neural Network for the given input
            batch of graphs. Note that this forward pass may not be deterministic wrt.
            floating point precision because the used scatter_reduce functions are not.

        Args:
            graph: Batch object of pytorch geometric.
                Represents a (batch of) graph(s)

        Returns:
            Either a modified graph or a tuple of (node_features, edge_features), depending on the
                configuration of the class at initialization.
                All node, edge and potentially global features went through an embedding and multiple rounds of
                message passing.

        """
        self.maybe_assertions(graph)
        graph = self.maybe_create_copy(graph)
        self.input_embeddings(graph)

        # "message passing stack"
        for outer_repeat in range(self._outer_repeats):  # "R" parameter in https://arxiv.org/pdf/2211.00801.pdf
            for step in self.vdgn_gat_steps:
                step(graph)
        return self.maybe_transform_output(graph)

