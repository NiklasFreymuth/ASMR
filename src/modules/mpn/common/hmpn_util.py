import torch
import torch_geometric

from util import keys as Keys
from util.types import *


def noop(*args, **kwargs):
    """
    No-op function.
    Args:
        *args: Arguments to be passed to the function
        **kwargs: Keyword arguments to be passed to the function

    Returns: None

    """
    return None


def unpack_features(graph: Data, agent_node_type: str = Keys.AGENT):
    """
    Unpacking important data from graphs.
    Args:
        graph (): The input observation
        agent_node_type: The name of the type of graph node that acts as the agent
     Returns:
        Tuple of edge_features, edge_index, node_features, global_features and batch
    """
    # edge features
    edge_features = graph.edge_attr
    edge_index = graph.edge_index.long()  # cast to long for scatter operators

    # node features
    node_features = graph.x if graph.x is not None else graph.pos

    # global features
    global_features = get_global_features(graph=graph) if hasattr(graph, "u") else None
    batch = graph.batch if hasattr(graph, "batch") else None
    if batch is None:
        batch = torch.zeros(node_features.shape[0]).long()

    return ({agent_node_type: node_features},
            {"edges": {"edge_index": edge_index,
                       "edge_attr": edge_features}},
            global_features,
            {agent_node_type: batch})


def get_global_features(graph: Batch) -> torch.Tensor:
    """
    Unpacks the global features of Batch
    Args:
        graph: The graph to unpack global features from
    Returns:
        The global features
    """
    global_features = graph.u
    num_graphs = graph.u.shape[0]  # [-1] + 1
    global_features = global_features.reshape((-1, int(len(global_features) / num_graphs)))
    global_features = global_features.float()
    return global_features


def make_batch(data: Union[Data, List[torch.Tensor], List[Data],], **kwargs):
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
