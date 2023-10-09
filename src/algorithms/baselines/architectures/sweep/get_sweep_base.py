from typing import Optional, Union, Dict

from modules.hmpn.common.embedding import Embedding

from util.types import ConfigDict


def get_sweep_base(*,
                   in_node_features: Union[int, Dict[str, int]],
                   latent_dimension: int,
                   base_config: ConfigDict,
                   node_name: str,
                   device: Optional[str] = None) -> Embedding:
    """
    Build and return a Message Passing Base specified in the config.

    Args:
        in_node_features: Either a single integer, or a dictionary of node types to integers
        latent_dimension: The dimension of the latent space
        base_config: The config for the base
        node_name: The node type of the agent
        device: The device to put the base on
    """
    embedding_config = base_config.get("embedding")
    if isinstance(in_node_features, int):
        in_features = in_node_features
    else:
        in_features = in_node_features[node_name]
    return Embedding(in_features=in_features,
                     latent_dimension=latent_dimension,
                     embedding_config=embedding_config).to(device)
