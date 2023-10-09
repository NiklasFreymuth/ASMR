from typing import Dict, Optional, Tuple

from torch_geometric.data.data import Data

from modules.hmpn.abstract.abstract_graph_assertions import AbstractGraphAssertions


class HomogeneousGraphAssertions(AbstractGraphAssertions):

    def __init__(self, *,
                 in_node_features: Dict[str, int],
                 in_edge_features: Dict[Tuple[str, str, str], int]
                 ):
        """

        Args:
            in_node_features:
                number of input features for nodes
            in_edge_features:
                number of input features for edges
        """
        super().__init__(in_node_features=in_node_features,
                         in_edge_features=in_edge_features)

    def __call__(self, tensor: Data):
        """
        Does various shape assertions to make sure that the (batch of) graph(s) is built correctly
        Args:
            tensor: (batch of) graph(s)

        Returns:

        """
        assert tensor.edge_index.shape[0] == 2, f"Edge index must have shape (2, num_edges), " \
                                                f"given '{tensor.edge_index.shape}' instead."
        assert tensor.edge_index.shape[1] == tensor.edge_attr.shape[0], f"Must provide one edge index per edge " \
                                                                        f"feature vector, given " \
                                                                        f"'{tensor.edge_index.shape}' and " \
                                                                        f"'{tensor.edge_attr.shape}' instead."

        in_edge_features = tensor.edge_attr.shape[1]
        expected_edge_features = self._assertion_dict.get("in_edge_features")
        assert in_edge_features == expected_edge_features, f"Feature dimensions of edges do not match. " \
                                                           f"Given '{in_edge_features}', " \
                                                           f"expected '{expected_edge_features}"

        in_node_features = tensor.x.shape[1]
        expected_node_features = self._assertion_dict.get("in_node_features")
        assert in_node_features == expected_node_features, f"Feature dimensions of nodes do not match. " \
                                                           f"Given '{in_node_features}', " \
                                                           f"expected '{expected_node_features}"