from typing import Optional, Union, Dict

from src.modules.mpn.common.embedding import Embedding
from util.types import ConfigDict


def get_sweep_base(*,
                   in_node_features: Union[int, Dict[str, int]],
                   latent_dimension: int,
                   base_config: ConfigDict,
                   device: Optional = None) -> Embedding:
    """
    Build and return a "sweep base" specified in the config

    Args:
        in_node_features: Number of (local) input features
        latent_dimension: The dimension of the latent space
        base_config: The config for the base
        device: The device to put the base on
    """
    from src.modules.mpn.common.embedding import Embedding
    embedding_config = base_config.get("embedding")
    return Embedding(in_features=in_node_features,
                     latent_dimension=latent_dimension,
                     embedding_config=embedding_config,
                     device=device)
