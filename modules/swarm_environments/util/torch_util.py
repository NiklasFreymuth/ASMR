from typing import Dict, List, Union

from numpy import ndarray
from torch import Tensor
from torch_geometric.data.data import BaseData


def detach(tensor: Union[Tensor, Dict[str, Tensor], List[Tensor]]) -> \
        Union[ndarray, Dict[str, ndarray], List[ndarray], BaseData]:
    if isinstance(tensor, dict):
        return {key: detach(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [detach(value) for value in tensor]
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()
