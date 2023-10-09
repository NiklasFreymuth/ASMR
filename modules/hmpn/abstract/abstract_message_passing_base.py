import abc
from typing import Dict, Union, Optional, Tuple, List, Callable, Type, Any

from torch import nn
from torch_geometric.data.batch import Batch

from modules.hmpn.abstract.abstract_graph_assertions import AbstractGraphAssertions
from modules.hmpn.abstract.abstract_input_embedding import AbstractInputEmbedding
from modules.hmpn.abstract.abstract_stack import AbstractStack
from modules.hmpn.common.hmpn_util import get_scatter_reducers, get_create_copy
from modules.hmpn.common.hmpn_util import noop


class AbstractMessagePassingBase(nn.Module, abc.ABC):
    """
    The Message Passing base contains the feature embedding as well as all the message passing steps.
    Its output is a graph or a tuple of node_features, edge_features, batch_indices with
    feature dimension of latent_dimension.
    """

    def __init__(self, *,
                 in_node_features: int,
                 in_edge_features: int,
                 latent_dimension: int,
                 embedding_config: Dict[str, Any],
                 scatter_reduce_strs: List[str],
                 unpack_output: bool,
                 create_graph_copy: bool = True,
                 assert_graph_shapes: bool = True):
        """

        Args:
            latent_dimension:
                Latent dimension of the network. All modules internally operate with latent vectors of this dimension
            scatter_reduce_strs:
                Names of the scatter reduce to use to aggregate messages of the same type.
                Can be a singular entity or a list of "sum", "mean", "max", "min", "std"
            unpack_output:
                Either "features" or "graph". Specifies whether the output of the forward pass is a graph
                or a tuple of (node_features, edge_features)
            create_graph_copy:
                If True, a copy of the input graph is created and modified in-place.
                If False, the input graph is modified in-place.
            assert_graph_shapes:
                If True, the input graph is checked for consistency with the input shapes.
                If False, the input graph is not checked for consistency with the input shapes.
        """

        if isinstance(scatter_reduce_strs, str):
            scatter_reduce_strs = [scatter_reduce_strs]
        super().__init__()
        self._latent_dimension = latent_dimension
        self._scatter_reducers = get_scatter_reducers(scatter_reduce_strs)

        self.maybe_assertions: AbstractGraphAssertions
        if assert_graph_shapes:
            self.maybe_assertions = self._get_assertions()(in_node_features=in_node_features,
                                                           in_edge_features=in_edge_features)
        else:
            self.maybe_assertions = noop

        self.maybe_create_copy: Callable = get_create_copy(create_graph_copy=create_graph_copy)

        input_embedding_class = self._get_input_embedding_type()
        self.input_embeddings: AbstractInputEmbedding = input_embedding_class(in_node_features=in_node_features,
                                                                              in_edge_features=in_edge_features,
                                                                              latent_dimension=latent_dimension,
                                                                              embedding_config=embedding_config)
        self.message_passing_stack: AbstractStack = None  # initialized in subclass init()

        self.maybe_transform_output = self._get_transform_output(unpack_output=unpack_output)

    def _get_assertions(self) -> Type[AbstractGraphAssertions]:
        raise NotImplementedError("'get_assertions' not implemented for AbstractMessagePassingBase")

    @staticmethod
    def _get_input_embedding_type() -> Type:
        raise NotImplementedError("'get_input_embeddings' not implemented for AbstractMessagePassingBase")

    def _get_transform_output(self, unpack_output: bool):
        """
        Returns a function that transforms the output of the network to the desired output type.
        Args:
            unpack_output: If true, will unpack the processed batch of graphs to a 4-tuple of
                ({node_name: node features}, {edge_name: edge features}, [placeholder], {node_name: batch indices}).
                Else, will return the raw processed batch of graphs

        Returns: Either a function that transforms the output of the network to a tuple of features,
         or a function that returns nothing.
        """
        if unpack_output:
            return self.unpack_features
        else:
            return lambda x, *args, **kwargs: x

    def unpack_features(self, graph: Batch) -> Tuple[
        Union[int, Dict[str, int]], Union[int, Dict[Tuple[str, str, str], int]], Optional[int], Union[
            int, Dict[str, int]]]:
        """
        Unpacks the output of the network.
        Returns a tuple of (node_features, edge_features, batch_indices)
        Args:
            graph:

        Returns:

        """
        raise NotImplementedError("'unpack_features' not implemented for AbstractMessagePassingBase")

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
                All node and edge features went through an embedding and multiple rounds of
                message passing.

        """
        self.maybe_assertions(graph)
        graph = self.maybe_create_copy(graph)
        self.input_embeddings(graph)
        self.message_passing_stack(graph)
        return self.maybe_transform_output(graph)
