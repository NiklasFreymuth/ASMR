import os
import numpy as np
import yaml
from pathlib import Path

from util.types import *


def save_dict(path: Path, file_name: str, dict_to_save: Dict,
              overwrite: bool = False, save_type: str = "npz") -> None:
    """
    Saves the given dict
    Args:
        path: Full/Absolute path to save to
        file_name: Name of the file to save the dictionary to
        dict_to_save: The dictionary to save
        overwrite: Whether to overwrite an existing file
        save_type: The type to save the dict as

    Returns:

    """
    if not file_name.endswith(f".{save_type}"):
        file_name += f".{save_type}"
    path.mkdir(parents=True, exist_ok=True)
    file_to_save = path / file_name
    if overwrite or not file_to_save.is_file():
        if save_type == "npz":
            np.savez_compressed(file_to_save, **dict_to_save)
        elif save_type == "yaml":
            yaml.dump(dict_to_save, file_name, sort_keys=True, indent=2)
        else:
            raise ValueError(f"Unknown save_type '{save_type}' for dictionary")


def load_dict(path: str) -> dict:
    """
    Loads the dictionary saved at the specified path
    Args:
        path: Path to the npz file in which the dictionary is saved. May or may not include the ".npz" at the end

    Returns: The dictionary saved at the specified path
    """
    if not path.endswith(".npz"):
        path += ".npz"
    assert os.path.isfile(path), "Path '{}' does not lead to a .npz file".format(path)
    return dict(np.load(path, allow_pickle=True))


def undo_numpy_conversions(dictionary: dict) -> dict:
    """
    Numpy does some weird conversions when you save dictionary objects. This method undoes them.
    Args:
        dictionary: The dictionary to un-convert

    Returns: The same dictionary but with un-numpied values

    """
    converted_dictionary = dictionary
    for converting_type in [float, int, dict]:
        converted_dictionary = {
            k: v.item() if isinstance(v, np.ndarray)
                           and (v.ndim == 0 or len(v) == 1)
                           and isinstance(v.item(), converting_type)
            else v for k, v in converted_dictionary.items()
        }

    none_converted_dict = {k: None if isinstance(v, np.ndarray)
                                      and (v.ndim == 0 or len(v) == 1)
                                      and v == None else v
                           for k, v in converted_dictionary.items()}
    tuple_converted_dict = {k: (v,) if isinstance(v, np.ndarray) and
                                       (v.ndim == 0 or len(v) == 1) and
                                       isinstance(v.item(), int) else v
                            for k, v in none_converted_dict.items()}
    return tuple_converted_dict
