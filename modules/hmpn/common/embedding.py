from typing import Dict, Optional, Any

import torch
from torch import nn

from modules.hmpn.common.empty_linear_layer import EmptyLinear
from modules.hmpn.common.latent_mlp import LatentMLP


class Embedding(nn.Module):
    """
    Linear Embedding module. Essentially a learned matrix multiplication (and a bias) to make input dimension of tokens
    compatible with the "main" architecture that uses them.
    """

    def __init__(self, in_features: int, latent_dimension: int, embedding_config: Optional[Dict[str, Any]]):
        """

        Args:
            in_features:
            latent_dimension:
            embedding_config:
        """
        super().__init__()
        if in_features == 0:
            # If there are no input features to embed, we instead learn constant initial values.
            # In this case, we also do not make use of the embedding configuration
            self.embedding = EmptyLinear(out_features=latent_dimension)
            self.is_constant_embedding = True
        else:
            if embedding_config is None:
                # Linearly embed if not embedding config is provided
                self.embedding = nn.Linear(in_features=in_features, out_features=latent_dimension)
            else:
                # Otherwise, we embed the features with an MLP
                mlp_config = embedding_config.get("mlp")
                self.embedding = LatentMLP(in_features=in_features, latent_dimension=latent_dimension,
                                           config=mlp_config)
            self.is_constant_embedding = False

        self._out_features = latent_dimension
        self.in_features = in_features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Embed the input graph using an MLP. If the input dimension for this embedding is 0, we instead use a learned
        constant placeholder value.
        Args:
            tensor:

        Returns:

        """
        return self.embedding(tensor)

    @property
    def out_features(self) -> int:
        return self._out_features

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"is_constant={self.is_constant_embedding}"
