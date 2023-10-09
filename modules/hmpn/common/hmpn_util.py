import copy
from typing import Union, Tuple, List

import torch
import torch_geometric
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData


def get_default_edge_relation(sender_node_type: str, receiver_node_type: str,
                              include_nodes: bool = True) -> Union[str, Tuple[str, str, str]]:
    """
    Wrapper function for uniform edge identifiers. Builds a string 'sender_node_type+"2"+receiver_node_type'

    Args:
        sender_node_type: Node type that sends a message along the specified edge
        receiver_node_type: Node type that receives a message along the specified edge
        include_nodes: If True, return a 3-tuple of strings
        (sender_node_type, 'sender_node_type+"2"+receiver_node_type', receiver_node_type).
        If False, return a string 'sender_node_type+"2"+receiver_node_type'

    Returns: If include_nodes,a 3-tuple of strings
        (sender_node_type, 'sender_node_type+"2"+receiver_node_type', receiver_node_type).

        Else a string 'sender_node_type+"2"+receiver_node_type'

    """
    edge_relation = sender_node_type + "2" + receiver_node_type
    if include_nodes:
        return sender_node_type, edge_relation, receiver_node_type
    else:
        return edge_relation


def tuple_to_string(input_tuple: Tuple) -> str:
    """
    Converts a tuple to a string.
    Args:
        input_tuple: The tuple to convert

    Returns: The string representation of the tuple

    """
    return "".join(input_tuple)


def noop(*args, **kwargs):
    """
    No-op function.
    Args:
        *args: Arguments to be passed to the function
        **kwargs: Keyword arguments to be passed to the function

    Returns: None

    """
    return None


def get_scatter_reducers(names: Union[List[str], str]) -> List[callable]:
    """
    Translates a list of strings to the appropriate functions from torch_scatter.
    Args:
        names: (List of) the names of the scatter operations: "std", "mean", "max", "min", "sum"

    Returns: (List of) the appropriate functions from torch_scatter with signature (src, index, dim, dim_size)

    """
    if type(names) == str:  # fallback case for single reducer
        names = [names]
    names: List[str]

    return [get_scatter_reduce(name) for name in names]


def get_scatter_reduce(name: str) -> callable:
    """
    Translates a string to the appropriate function from torch_scatter.
    Args:
        name: the name of the scatter operation: "std", "mean", "max", "min", "sum"

    Returns: the appropriate function from torch_scatter with signature (src, index, dim)

    """
    if name == "mean":
        from torch_scatter import scatter_mean
        scatter_reduce = scatter_mean
    elif name == "min":
        from torch_scatter import scatter_min
        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif name == "max":
        from torch_scatter import scatter_max
        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif name == "sum":
        from torch_scatter import scatter_add
        scatter_reduce = scatter_add
    elif name == "std":
        from torch_scatter import scatter_std
        scatter_reduce = scatter_std
    else:
        raise ValueError(f"Unknown scatter reduce '{name}'")
    return scatter_reduce


def unpack_homogeneous_features(graph: Union[Batch, Data], node_name: str):
    """
    Unpacking important data from homogeneous graphs.
    Args:
        graph: The input homogeneous observation
        node_name: Name of the node type of the agent
     Returns:
        Tuple of edge_features, edge_index, node_features, None (placeholder) and batch
    """
    # edge features
    edge_features = graph.edge_attr
    edge_index = graph.edge_index.long()  # cast to long for scatter operators

    # node features
    node_features = graph.x if graph.x is not None else graph.pos

    batch = graph.batch if hasattr(graph, "batch") else None
    if batch is None:
        batch = torch.zeros(node_features.shape[0]).long()

    return ({node_name: node_features},
            {get_default_edge_relation(node_name, node_name): {"edge_index": edge_index,
                                                               "edge_attr": edge_features}},
            None,
            {node_name: batch})


def make_batch(data: Union[HeteroData, Data, List[torch.Tensor], List[Data], List[HeteroData]], **kwargs):
    """
    adds the .batch-argument with zeros
    Args:
        data:

    Returns:

    """
    if isinstance(data, torch_geometric.data.Data):
        return Batch.from_data_list([data], **kwargs)
    elif isinstance(data, list) and isinstance(data[0], torch_geometric.data.Data):
        return Batch.from_data_list(data, **kwargs)
    elif isinstance(data, list) and isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)

    return data


def get_create_copy(create_graph_copy: bool):
    """
    Returns a function that creates a copy of the graph.
    Args:
        create_graph_copy: Whether to create a copy of the graph or not
    Returns: A function that creates a copy of the graph, or an empty function if create_graph_copy is False
    """
    if create_graph_copy:
        return lambda x: copy.deepcopy(x)
    else:
        return lambda x: x
