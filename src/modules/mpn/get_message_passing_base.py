from src.modules.mpn.message_passing_base import MessagePassingBase
from util.types import *


def get_message_passing_base(*,
                             in_node_features: int,
                             in_edge_features: int,
                             in_global_features: int,
                             latent_dimension: int,
                             base_config: ConfigDict,
                             agent_node_type: str,
                             device: Optional = None) -> MessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config.

    Args:
        in_node_features: Either a single integer, or a dictionary of node types to integers
        in_edge_features: Either a single integer, or a dictionary of (source_node_type, edge_type, target_node_type)
            to integers
        in_global_features: Either None, or an integer
        latent_dimension: The dimension of the latent space
        base_config: The config for the base
        agent_node_type: The node type of the agent
        device: The device to put the base on

    Returns:
        A Message Passing Base.

    """
    assert type(in_node_features) == type(in_edge_features), f"May either provide feature dimensions as int or Dict, " \
                                                             f"but not both. " \
                                                             f"Given '{in_node_features}', '{in_edge_features}'"

    create_graph_copy = base_config.get("create_graph_copy", True)
    assert_graph_shapes = base_config.get("assert_graph_shapes", False)
    stack_config = base_config.get("stack")
    embedding_config = base_config.get("embedding")

    return MessagePassingBase(in_node_features=in_node_features,
                              in_edge_features=in_edge_features,
                              in_global_features=in_global_features,
                              latent_dimension=latent_dimension,
                              stack_config=stack_config,
                              embedding_config=embedding_config,
                              create_graph_copy=create_graph_copy,
                              assert_graph_shapes=assert_graph_shapes,
                              device=device,
                              agent_node_type=agent_node_type)
