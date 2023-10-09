from typing import Dict, Union, Optional, Tuple, Any

from torch_geometric.data.data import Data
from torch_geometric.data.hetero_data import HeteroData

from modules.hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase

def get_hmpn_from_graph(*, example_graph: Union[Data, HeteroData],
                        latent_dimension: int,
                        base_config: Dict[str, Any],
                        node_name: str = "node",
                        unpack_output: bool = True,
                        device: Optional = None) -> AbstractMessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config from the provided example graph.
    Args:
        example_graph: A graph that is used to infer the input feature dimensions for nodes, edges and globals
        latent_dimension: Dimensionality of the latent space
        base_config: Dictionary specifying the way that the gnn base should
        node_name: Name of the node for homogeneous graphs. Will be used for the unpacked output namingook like.
        unpack_output: If true, will unpack the processed batch of graphs to a 4-tuple of
            ({node_name: node features}, {edge_name: edge features}, placeholder, {node_name: batch indices}).
            Else, will return the raw processed batch of graphs
        device: The device to put the base on. Either cpu or a single gpu

    Returns:

    """
    if isinstance(example_graph, Data):
        in_node_features = example_graph.x.shape[1]
        in_edge_features = example_graph.edge_attr.shape[1]
    elif isinstance(example_graph, HeteroData):
        in_node_features = {node_type: example_graph[node_type].x.shape[1]
                            for node_type in example_graph.node_types}
        in_edge_features = {edge_type: example_graph[edge_type].edge_attr.shape[1]
                            for edge_type in example_graph.edge_types}
    else:
        raise TypeError(f"Expected example_graph to be of type Data or HeteroData, but got {type(example_graph)}")
    return get_hmpn(in_node_features=in_node_features,
                    in_edge_features=in_edge_features,
                    latent_dimension=latent_dimension,
                    base_config=base_config,
                    node_name=node_name,
                    unpack_output=unpack_output,
                    device=device)


def get_hmpn(*,
             in_node_features: Union[int, Dict[str, int]],
             in_edge_features: Union[int, Dict[Tuple[str, str, str], int]],
             latent_dimension: int,
             base_config: Dict[str, Any],
             node_name: str = "node",
             unpack_output: bool = True,
             device: Optional = None) -> AbstractMessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config. The base may be either suited for message
    passing on homogeneous graphs, depending on whether the input feature dimensions for nodes and
    edges are given as dictionaries Dict[str, int], or as simple integers.

    Args:
        in_node_features: Either a single integer, or a dictionary of node types to integers
        in_edge_features: Either a single integer, or a dictionary of (source_node_type, edge_type, target_node_type)
            to integers
        latent_dimension: The dimension of the latent space
        base_config: The config for the base
        node_name: Name of the node for homogeneous graphs. Will be used for the unpacked output naming
        unpack_output: If true, will unpack the processed batch of graphs to a 4-tuple of
            ({node_name: node features}, {edge_name: edge features}, placeholder, {node_name: batch indices}).
            Else, will return the raw processed batch of graphs
        device: The device to put the base on. Either cpu or a single gpu

    Returns:
        A Message Passing Base operating on either homogeneous graphs.

    """
    assert type(in_node_features) == type(in_edge_features), f"May either provide feature dimensions as int or Dict, " \
                                                             f"but not both. " \
                                                             f"Given '{in_node_features}', '{in_edge_features}'"

    create_graph_copy = base_config.get("create_graph_copy", True)
    assert_graph_shapes = base_config.get("assert_graph_shapes")
    stack_config = base_config.get("stack")
    embedding_config = base_config.get("embedding")
    scatter_reduce_strs = base_config.get("scatter_reduce")
    flip_edges_for_nodes = base_config.get('flip_edges_for_nodes', False)
    if isinstance(scatter_reduce_strs, str):
        scatter_reduce_strs = [scatter_reduce_strs]

    params = dict(in_node_features=in_node_features,
                  in_edge_features=in_edge_features,
                  latent_dimension=latent_dimension,
                  scatter_reduce_strs=scatter_reduce_strs,
                  stack_config=stack_config,
                  unpack_output=unpack_output,
                  embedding_config=embedding_config,
                  create_graph_copy=create_graph_copy,
                  assert_graph_shapes=assert_graph_shapes,
                  flip_edges_for_nodes=flip_edges_for_nodes,
                  )

    from modules.hmpn.homogeneous.homogeneous_message_passing_base import HomogeneousMessagePassingBase
    base = HomogeneousMessagePassingBase(**params,
                                         node_name=node_name)
    base = base.to(device)
    return base
