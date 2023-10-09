import abc
from typing import Dict, Tuple, Union


class AbstractGraphAssertions(abc.ABC):
    """
    Asserts that the input graph has the correct shape.
    """

    def __init__(self, *,
                 in_node_features: Union[int, Dict[str, int]],
                 in_edge_features: Union[int, Dict[Tuple[str, str, str], int]]
                 ):
        """

        Args:
            in_node_features: The number of input node features (per node type)
            in_edge_features: The number of input edge features (per edge type)
        """
        self._assertion_dict = {"in_node_features": in_node_features,
                                "in_edge_features": in_edge_features,
                                }
