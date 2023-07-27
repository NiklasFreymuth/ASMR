import torch
from torch import nn as nn
from torch.nn import Parameter, init


class EmptyLinear(nn.Module):
    """
    "Linear" layer that expects an input_size of 0.
    This layer is meant to be called with an empty input of shape (batch_size, 0) and will return
    batch_size repeats of its (trained) bias as an array of shape (batch_size, out_features)
    """
    def __init__(self, out_features: int, bias: bool = True, device=None, dtype=None, **kwargs) -> None:
        """
        Initializes a new EmptyLinear layer. Essentially a learned, fixed vector of size out_features.

        Args:
            out_features: The number of output features.
            bias: Whether to include a bias term. If false, the output will be a zero vector.
            device: The device to use for the layer.
            dtype: The data type to use for the layer.
            **kwargs:
        """
        super(EmptyLinear, self).__init__()
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
            self._forward_function = self._forward_with_bias
        else:
            self.register_parameter('bias', None)
            self._forward_function = self._forward_without_bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the layer. initializes the bias to a uniform distribution between -1 and 1.

        Returns: None

        """
        if self.bias is not None:
            init.uniform_(self.bias, -1, 1)

    def _forward_with_bias(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).repeat(tensor.shape[0], 1)

    def _forward_without_bias(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(size=(tensor.shape[0], self.out_features))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        Args:
            tensor: An empty graph of size (batch_size, 0)

        Returns: An output graph of shape (batch_size, self.out_features) containing #batch_size copies of the bias,
          or 0 if no bias is provided

        """
        return self._forward_function(tensor)

    def extra_repr(self) -> str:
        return 'out_features={}, bias={}'.format(self.out_features, self.bias is not None)

