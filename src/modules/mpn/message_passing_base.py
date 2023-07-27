import abc

from torch import nn

from src.modules.mpn.common.hmpn_util import get_create_copy
from src.modules.mpn.common.hmpn_util import noop
from src.modules.mpn.common.hmpn_util import unpack_features
from src.modules.mpn.graph_assertions import MessagePassingGraphAssertions
from src.modules.mpn.input_embedding import MessagePassingInputEmbedding
from src.modules.mpn.message_passing_stack import MessagePassingStack
from util.keys import AGENT
from util.types import *


class MessagePassingBase(nn.Module, abc.ABC):
    """
        Graph Neural Network (GNN) Base module processes the graph observations of the environment.
        It uses a stack of GNN Blocks. Each block defines a single GNN pass.
    """

    def __init__(self, *, in_node_features: int,
                 in_edge_features: int,
                 in_global_features: int,
                 latent_dimension: int,
                 stack_config: ConfigDict,
                 embedding_config: ConfigDict,
                 output_type: str = "features",
                 create_graph_copy: bool = True,
                 assert_graph_shapes: bool = True,
                 device: Optional = None,
                 agent_node_type: str = AGENT):
        """
        Args:
            in_node_features:
                Node feature input size.
                Node features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            in_edge_features:
                Edge feature input size.
                Edge features may have size 0, in which case an empty input graph of the appropriate shape/batch_size
                is expected and the initial embeddings are learned constants
            in_global_features:
                May have size 0, in which case the initial values are a learned constant. This expects (empty) global
                 input tensors-
            latent_dimension:
                Latent dimension of the network. All modules internally operate with latent vectors of this dimension
            stack_config:
                Configuration of the stack of GNN blocks. See the documentation of the stack for more information.
            embedding_config:
                Configuration of the embedding stack (can be empty by choosing None, resulting in linear embeddings).
            output_type:
                Either "features" or "graph". Specifies whether the output of the forward pass is a graph
                or a tuple of (node_features, edge_features, global_features)
            create_graph_copy:
                If True, a copy of the input graph is created and modified in-place.
                If False, the input graph is modified in-place.
            assert_graph_shapes:
                If True, the input graph is checked for consistency with the input shapes.
                If False, the input graph is not checked for consistency with the input shapes.
            device:
                Device to use for the module. If None, the default device is used.
            agent_node_type:
                Node type of the agent. Used to determine the node type of the agent.
        """
        super().__init__()
        self._latent_dimension = latent_dimension

        self._agent_node_type = agent_node_type

        self.maybe_assertions: MessagePassingGraphAssertions
        if assert_graph_shapes:
            self.maybe_assertions = self._get_assertions()(in_node_features=in_node_features,
                                                           in_edge_features=in_edge_features,
                                                           in_global_features=in_global_features)
        else:
            self.maybe_assertions = noop

        self.maybe_create_copy: Callable = get_create_copy(create_graph_copy=create_graph_copy)

        self.maybe_transform_output = self._get_transform_output(output_type=output_type)

        self.input_embeddings = MessagePassingInputEmbedding(in_node_features=in_node_features,
                                                             in_edge_features=in_edge_features,
                                                             in_global_features=in_global_features,
                                                             latent_dimension=latent_dimension,
                                                             embedding_config=embedding_config,
                                                             device=device)

        # create message passing stack
        self.message_passing_stack = MessagePassingStack(stack_config=stack_config,
                                                         latent_dimension=latent_dimension)

    def _get_assertions(self) -> Type[MessagePassingGraphAssertions]:
        return MessagePassingGraphAssertions

    @staticmethod
    def _get_input_embeddings() -> Type[MessagePassingInputEmbedding]:
        return MessagePassingInputEmbedding

    @staticmethod
    def _get_message_passing_stack() -> Type[MessagePassingStack]:
        return MessagePassingStack

    def transform_to_features(self, graph: Batch) -> Batch:
        return unpack_features(graph, agent_node_type=self._agent_node_type)

    def _get_transform_output(self, output_type: str):
        """
        Returns a function that transforms the output of the network to the desired output type.
        Args:
            output_type: Either "features" or "graph".

        Returns: Either a function that transforms the output of the network to the desired output type, or a function
        that returns nothing if output_type is "graph"
        """
        if output_type == "features":
            return self.transform_to_features
        elif output_type == "graph":
            return noop
        else:
            raise ValueError(f"Unknown output_type '{output_type}'")

    def forward(self, graph: Batch) -> Batch:
        """
        Performs a forward pass through the Message Passing/Graph Neural Network for the given input

        Args:
            graph: Batch object of pytorch geometric.
                Represents a (batch of) graph(s)

        Returns:
            Either a modified graph or a tuple of (node_features, edge_features, global_features), depending on the
                configuration of the class at initialization.
                All node, edge and potentially global features went through an embedding and multiple rounds of
                message passing.

        """
        self.maybe_assertions(graph)
        graph = self.maybe_create_copy(graph)
        self.input_embeddings(graph)
        self.message_passing_stack(graph)
        return self.maybe_transform_output(graph)
