from typing import Dict, Any, List, Union, Iterable, Callable, Optional, Tuple, Generator, Type, Set, Type, Literal
from numpy import ndarray
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import Data, BaseData
import copy
from functools import partial

"""
Custom class that redefines various types to increase clarity.
"""
Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, ndarray]
Shape = Union[int, Iterable, ndarray]

InputBatch = Union[Dict[Key, Tensor], Tensor, Batch, Data, None]
OutputTensorDict = Dict[Key, Tensor]
