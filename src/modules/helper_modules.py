import torch
from torch import nn
from util.types import *


class View(nn.Module):
    """
    Custom Layer to change the shape of an incoming graph.
    """

    def __init__(self, default_shape: Union[tuple, int], custom_repr: str = None):
        """
        Utility layer to reshape an input into the given shape. Auto-converts for different batch_sizes
        Args:
            default_shape: The shape (without the batch size) to convert to
            custom_repr: Custom representation string of this layer.
            Shown when printing the layer or a network containing it
        """
        import numpy as np
        super(View, self).__init__()
        if isinstance(default_shape, (int, np.int32, np.int64)):
            self._default_shape = (default_shape,)
        elif isinstance(default_shape, tuple):
            self._default_shape = default_shape
        else:
            raise ValueError("Unknown type for 'shape' parameter of View module: {}".format(default_shape))
        self._custom_repr = custom_repr

    def forward(self, tensor: torch.Tensor, shape: Optional[Iterable] = None) -> torch.Tensor:
        """
        Shape the given input graph with the provided shape, or with the default_shape if None is provided
        Args:
            tensor: Input graph of arbitrary shape
            shape: The shape to fit the input into

        Returns: Same graph, but shaped according to the shape parameter (if provided) or self._default_shape)

        """
        tensor = tensor.contiguous()  # to have everything nicely arranged in memory, which is a requisite for .view()
        if shape is None:
            return tensor.view((-1, *self._default_shape))  # -1 to deal with different batch sizes
        else:
            return tensor.view((-1, *shape))

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        if self._custom_repr is not None:
            return "{} default_shape={}".format(self._custom_repr, self._default_shape)
        else:
            return "default_shape={}".format(self._default_shape)


class Squeeze(nn.Module):
    """
    Custom Layer to squeeze an input, because Torch seemingly comes without any practical functionality
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(tensor)


class SkipConnection(nn.Module):
    """
    Custom Layer that expects two inputs and concatenates them into a common output along a specified axis/dimension
    """

    def __init__(self, dim: int = 1):
        self._dim = dim
        super().__init__()

    def forward(self, new_tensor: torch.Tensor, old_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(tensors=[new_tensor, old_tensor], dim=self._dim)


class GradientBlock(nn.Module):
    """
    Custom Layer that blocks a gradient from going through
    """
    def forward(self, tensor: Union[Tensor, List[Tensor], Dict[Key, Tensor]]) -> Union[Tensor,
                                                                                       List[Tensor],
                                                                                       Dict[Key, Tensor]]:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
        elif isinstance(tensor, list):
            return [_tensor.detach() for _tensor in tensor]
        elif isinstance(tensor, dict):
            return {key: _tensor.detach() for key, _tensor in tensor.items()}
        else:
            raise ValueError(f"Unknown graph type '{type(tensor)}' for graph '{tensor}'")


class SaveBatchnorm1d(nn.BatchNorm1d):
    """
    Custom Layer to deal with Batchnorms for 1-sample inputs
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size = tensor.shape[0]
        if batch_size == 1 or len(tensor.shape) == 1:
            return tensor
        else:
            return super().forward(input=tensor)


class LinearEmbedding(nn.Module):
    """
    Linear Embedding module. Essentially a learned matrix multiplication (and a bias) to make input dimension of tokens
    compatible with the "main" architecture that uses them.
    """

    def __init__(self, in_features: int, out_features: int):
        """

        Args:
            in_features: The total number of input features.
            out_features: The total number of output features.
        """
        super().__init__()
        if in_features == 0:
            # If there are no input features to embed, we instead learn constant initial values.
            self.is_constant_embedding = True
            in_features = 1
        else:
            self.is_constant_embedding = False
            
        self._out_features = out_features
        self.embedding_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Linearly embed the input graph. If the input dimension for this embedding is 0, we instead use a placeholder
        of a single one per batch as an input to generate valid output values.
        Args:
            tensor:

        Returns:

        """
        if self.is_constant_embedding:
            tensor = torch.ones(size=(*tensor.shape[:-1], 1))
        return self.embedding_layer(tensor)
    
    @property
    def out_features(self) -> int:
        return self._out_features
