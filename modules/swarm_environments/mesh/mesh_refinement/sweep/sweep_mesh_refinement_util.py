import numpy as np
import torch
from modules.swarm_environments.util.torch_util import detach


def get_resource_budget(num_agents: int, num_max_agents: int):
    return np.full((num_agents,), num_agents / num_max_agents)


def get_average_with_same_shape(input_array: np.ndarray):
    return np.full((input_array.shape[0],), np.mean(input_array))


def get_neighbors_area(edge_indices: torch.Tensor, areas: np.ndarray, aggregation: str):
    """
    For each graph node, get the mean/min/max of its neighbor face areas.
    Args:
        edge_indices: edge index in the mesh
        areas: areas of mesh faces
        aggregation: either mean, min or max
    """
    areas = torch.tensor(areas)  # convert areas to tensor
    edge_indices = edge_indices[:, edge_indices[0] != edge_indices[1]]
    edge_features = areas[edge_indices[0]]
    if aggregation == "mean":
        from torch_scatter import scatter_mean
        scatter_reduce = scatter_mean
    elif aggregation == "min":
        from torch_scatter import scatter_min
        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif aggregation == "max":
        from torch_scatter import scatter_max
        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'")
    neighbor_areas = scatter_reduce(edge_features, edge_indices[1], dim=0)
    return detach(neighbor_areas)


def get_edge_attributes(edge_dict: dict, feature_position: int, aggregation: str):
    """
    For each graph node, get the mean/min/max of its incoming edge features.
    Args:
        edge_dict: edge dictionary containing the edge index and the edge feature attributes
        feature_position: index of edge feature (there can be multiple edge features)
        aggregation: mean, min or max
    """
    edge_indices = edge_dict.get("edge_index")
    edge_features = edge_dict.get("edge_attr")[:, feature_position].double()  # get the i'th edge feature from edge dict

    self_edges = edge_indices[0] == edge_indices[1]
    edge_indices = edge_indices[:, ~self_edges]
    edge_features = edge_features[~self_edges]
    if aggregation == "mean":
        from torch_scatter import scatter_mean
        scatter_reduce = scatter_mean
    elif aggregation == "min":
        from torch_scatter import scatter_min
        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif aggregation == "max":
        from torch_scatter import scatter_max
        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'")
    edge_attributes = scatter_reduce(edge_features, edge_indices[1], dim=0)
    return detach(edge_attributes)
