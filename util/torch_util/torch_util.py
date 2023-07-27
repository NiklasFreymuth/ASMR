import torch
import torch.optim as optim

from util.types import *


def detach(tensor: Union[Tensor, Dict[Key, Tensor], List[Tensor]]) -> \
        Union[ndarray, Dict[Key, ndarray], List[ndarray], BaseData]:
    if isinstance(tensor, dict):
        return {key: detach(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [detach(value) for value in tensor]
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def normalize(tensor: Tensor, epsilon: float = 1.0e-8, dim: int = 0) -> Tensor:
    return (tensor - torch.mean(tensor, dim=dim)) / (torch.std(tensor, dim=dim) + epsilon)


def assert_same_shape(reference_tensor: Tensor, *other_tensors):
    for other_tensor in other_tensors:
        assert reference_tensor.shape == other_tensor.shape
