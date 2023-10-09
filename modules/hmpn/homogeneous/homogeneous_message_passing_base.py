from typing import Dict, Union, Optional, Any, List, Type

from torch_geometric.data.batch import Batch

from modules.hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from modules.hmpn.common.hmpn_util import unpack_homogeneous_features
from modules.hmpn.homogeneous.homogeneous_graph_assertions import HomogeneousGraphAssertions
from modules.hmpn.homogeneous.homogeneous_input_embedding import HomogeneousInputEmbedding
from modules.hmpn.homogeneous.homogeneous_stack import HomogeneousStack


class HomogeneousMessagePassingBase(AbstractMessagePassingBase):
    """
        Graph Neural Network (GNN) Base module processes the graph observations of the environment.
        It uses a stack of GNN Steps. Each step defines a single GNN pass.
    """

    def __init__(self, *, in_node_features: int,
                 in_edge_features: int,
                 latent_dimension: int,
                 scatter_reduce_strs: Union[List[str], str],
                 stack_config: Dict[str, Any],
                 embedding_config: Dict[str, Any],
                 unpack_output: bool,
                 create_graph_copy: bool = True,
                 assert_graph_shapes: bool = True,
                 flip_edges_for_nodes: bool = False,
                 node_name: str = "node"):
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
            scatter_reduce_strs:
                Names of the scatter reduce to use to aggregate messages of the same type.
                Can be multiple of "sum", "mean", "max", "min", "std"
                e.g. ["sum","max"]
            stack_config:
                Configuration of the stack of GNN steps. See the documentation of the stack for more information.
            embedding_config:
                Configuration of the embedding stack (can be empty by choosing None, resulting in linear embeddings).
            unpack_output:
                If true, the output of the forward pass is unpacked into a tuple of (node_features, edge_features).
                If false, the output of the forward pass is the raw graph.
            create_graph_copy:
                If True, a copy of the input graph is created and modified in-place.
                If False, the input graph is modified in-place.
            assert_graph_shapes:
                If True, the input graph is checked for consistency with the input shapes.
                If False, the input graph is not checked for consistency with the input shapes.
            node_name:
                Name of the node. Used to convert the output of the forward pass to a dictionary
        """
        super().__init__(in_node_features=in_node_features,
                         in_edge_features=in_edge_features,
                         latent_dimension=latent_dimension,
                         embedding_config=embedding_config,
                         scatter_reduce_strs=scatter_reduce_strs,
                         unpack_output=unpack_output,
                         create_graph_copy=create_graph_copy,
                         assert_graph_shapes=assert_graph_shapes
                         )

        self._node_name = node_name

        # create message passing stack
        self.message_passing_stack = HomogeneousStack(stack_config=stack_config,
                                                      latent_dimension=latent_dimension,
                                                      scatter_reducers=self._scatter_reducers,
                                                      flip_edges_for_nodes=flip_edges_for_nodes)

    def _get_assertions(self) -> Type[HomogeneousGraphAssertions]:
        return HomogeneousGraphAssertions

    @staticmethod
    def _get_input_embedding_type() -> Type[HomogeneousInputEmbedding]:
        return HomogeneousInputEmbedding

    def unpack_features(self, graph: Batch) -> Batch:
        return unpack_homogeneous_features(graph, node_name=self._node_name)
