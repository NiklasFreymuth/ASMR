import collections
from functools import update_wrapper

import numpy as np

from util.types import *
from util.types import ConfigDict


def save_concatenate(arrays: Iterable[np.array], *args, **kwargs) -> Optional[np.ndarray]:
    arrays = [array for array in arrays if array is not None]
    if len(arrays) == 0:
        return None
    return np.concatenate(arrays, *args, **kwargs)


def safe_mean(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def safe_max(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the max of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.
    """
    return np.nan if len(arr) == 0 else np.max(arr)


def safe_min(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the max of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.
    """
    return np.nan if len(arr) == 0 else np.min(arr)




def get_from_nested_dict(dictionary: Dict[Any, Any], list_of_keys: List[Any],
                         raise_error: bool = False,
                         default_return: Optional[Any] = None) -> Any:
    """
    Utility function to traverse through a nested dictionary. For a given dict d and a list of keys [k1, k2, k3], this
    will return d.get(k1).get(k2).get(k3) if it exists, and default_return otherwise
    Args:
        dictionary: The dictionary to search through
        list_of_keys: List of keys in the order to traverse the dictionary by
        raise_error: Raises an error if any of the keys is missing in its respective subdict. If false, returns the
        default_return instead
        default_return: The thing to return if the dictionary does not contain the next key at some level

    Returns:

    """
    current_dict = dictionary
    for key in list_of_keys:
        if isinstance(current_dict, dict):  # still traversing dictionaries
            current_dict = current_dict.get(key, None)
        if current_dict is None:  # key of sub-dictionary not found
            if raise_error:
                raise ValueError("Dict '{}' does not contain list_of_keys '{}'".format(dictionary, list_of_keys))
            else:
                return default_return
    return current_dict  # bottom level reached


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def list_of_dicts_from_dict_of_lists(nested_dict: ConfigDict) -> List[ConfigDict]:
    """
    Takes a nested dictionary where some values may be lists and returns a new dictionary for each row-wise combination
    of elements for this list. These new dictionaries are then put in an outer list.
    E.g., a dictionary
    a:
      b: 1
      c: [2,3]
    d:
      e: [4,5]
      f: 6
    g: [7,8]

    will be turned into a list
    [a:
      b: 1
      c: 2
    d:
      e: 4
      f: 6
    g: 7,

    a:
      b: 1
      c: 3
    d:
      e: 5
      f: 6
    g: 8]
    Args:
        nested_dict: The nested dictionary to transform. May contain list elements, in which case all lists must be
          of the same length

    Returns: A new dictionary that is a merged version of the two provided ones

    """

    def list_length_in_dict(checked_dictionary: ConfigDict) -> Optional[int]:
        length = 1
        for name, value in checked_dictionary.items():
            if isinstance(value, dict):  # recurse
                sublength = list_length_in_dict(checked_dictionary=value)
            elif isinstance(value, list):  # is a list, so take this length
                sublength = len(value)
            else:
                sublength = 1
            if sublength > 1:
                if length == 1:
                    length = sublength
                else:
                    assert length == sublength, f"Need all list entries to have common length" \
                                                f" '{length}'. Got length '{sublength}' for key '{name}'"
        return length

    def take_nth_elements(dictionary, n: int):
        for name, value in dictionary.items():
            if isinstance(value, dict):  # recurse
                dictionary[name] = take_nth_elements(value, n=n)
            elif isinstance(value, list):  # is a list, so split evenly
                dictionary[name] = value[n]
        return dictionary

    list_length = list_length_in_dict(copy.deepcopy(nested_dict))
    list_of_dicts = []
    for position in range(list_length):
        configuration = take_nth_elements(dictionary=copy.deepcopy(nested_dict), n=position)
        list_of_dicts.append(configuration)
    return list_of_dicts


def flatten_dict_to_tuple_keys(d: collections.MutableMapping):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            sub_dict = flatten_dict_to_tuple_keys(v)
            flat_dict.update({(k, *sk): sv for sk, sv in sub_dict.items()})

        elif isinstance(v, collections.MutableSequence):
            flat_dict[(k,)] = v

    return flat_dict


def merge_nested_dictionaries(destination: dict, source: dict) -> dict:
    """
    does a deep merge of the given dictionaries. The destination dictionary will be overwritten by the source one in
    case of conflicting keys
    Args:
        destination:
        source:

    Returns: A new dictionary that is a merged version of the two provided ones

    """

    def _merge_dictionaries(dict1, dict2):
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    _merge_dictionaries(dict1[key], dict2[key])
                elif dict1[key] == dict2[key]:
                    pass  # same leaf value
                else:
                    dict1[key] = dict2[key]
            else:
                dict1[key] = dict2[key]
        return dict1

    return _merge_dictionaries(copy.deepcopy(destination), copy.deepcopy(source))


def prefix_keys(dictionary: Dict[str, Any], prefix: Union[str, List[str]], separator: str = "/") -> Dict[str, Any]:
    if isinstance(prefix, str):
        prefix = [prefix]
    prefix = separator.join(prefix + [""])
    return {prefix + k: v for k, v in dictionary.items()}


def add_to_dictionary(dictionary: ValueDict, new_scalars: ValueDict) -> ValueDict:
    for k, v in new_scalars.items():
        if k not in dictionary:
            dictionary[k] = []
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim == 1):
            dictionary[k] = dictionary[k] + v
        else:
            dictionary[k].append(v)
    return dictionary


def get_triangle_areas_from_indices(positions: np.array, triangle_indices: np.array) -> np.array:
    """
    Computes the area for an array of triangles using the triangle-wise formula
    Area = 0.5*| (Xb-Xa)(Yc-Ya)-(Xc-Xa)(Yb-Ya) | where a,b,c are 3 vertices
    for coordinates X and Y
    Args:
        positions: Array of shape (#points, 2) of (x,y) coordinates
        triangle_indices: Array of shape (#triangles, 3) containing point indices that span triangles

    Returns: An array of shape (#triangles,) of areas for the input triangles

    """

    area = np.abs(0.5 * ((positions[triangle_indices[:, 1], 0] - positions[triangle_indices[:, 0], 0]) *
                         (positions[triangle_indices[:, 2], 1] - positions[triangle_indices[:, 0], 1]) -
                         (positions[triangle_indices[:, 2], 0] - positions[triangle_indices[:, 0], 0]) *
                         (positions[triangle_indices[:, 1], 1] - positions[triangle_indices[:, 0], 1])))
    return area


def get_triangle_areas_from_positions(triangles: np.array) -> np.array:
    """
    Computes the area for an array of triangles using the triangle-wise formula
    Area = 0.5*| (Xb-Xa)(Yc-Ya)-(Xc-Xa)(Yb-Ya) | where a,b,c are 3 vertices
    for coordinates X and Y
    Args:
        triangles: Array of shape (..., 3, 2) of triangles in 2d space

    Returns: An array of shape (...,) of areas for the input triangles

    """
    unscaled_area = (triangles[..., 1, 0] - triangles[..., 0, 0]) * (triangles[..., 2, 1] - triangles[..., 0, 1]) - \
                    (triangles[..., 2, 0] - triangles[..., 0, 0]) * (triangles[..., 1, 1] - triangles[..., 0, 1])
    area = np.abs(0.5 * unscaled_area)
    return area


def filter_included_fields(dictionary: ConfigDict) -> List[str]:
    """
    A helper function to filter out the fields that are not included in the config.
    Args:
        dictionary: A dictionary containing the fields to filter. The values are booleans indicating whether to include
            the field.

    Returns:

    """
    return [feature_name for feature_name, include_feature in dictionary.items() if include_feature]
