"""
Utility file that implements methods used by the recorder and loggers
"""
import os
from util.types import *


def map_index_to_dash_format(index: int, throw_error: bool = False) -> str:
    if index == 0:
        return "solid"
    elif index == 1:
        return "dash"
    elif index == 2:
        return "dot"
    elif index == 3:
        return "longdash"
    elif index == 4:
        return "dashdot"
    elif index == 5:
        return "longdashdot"
    else:
        if throw_error:
            raise NotImplementedError(f"Currently only implements 6 dash styles. Requested index {index}")
        else:
            return map_index_to_dash_format(index=index % 6)  # do again but guarantee an index


def process_logger_name(logger_name: str) -> str:
    logger_name = logger_name.lower()
    if logger_name.endswith("logger"):
        logger_name = logger_name[:-6]
    return logger_name


def get_trial_directory_path() -> str:
    from cw2.cw_config import cw_conf_keys as cw2_keys
    trial_directory_path = cw2_keys.i_REP_LOG_PATH  # filepath for the current cw2 trial
    return trial_directory_path


def process_logger_message(entity_key: str, entity_value: Any, indent: int = 30) -> str:
    message_string_template = "{" + ":<{}".format(indent) + "}: {}"
    if isinstance(entity_value, float):
        entity_value = round(entity_value, ndigits=3)
    return message_string_template.format(entity_key.title(), entity_value)


def save_to_yaml(dictionary: Dict[Key, Any], save_name: str, recording_directory: str) -> None:
    """
    Save the current dictionary as an input_type.yaml

    Args:
        dictionary: The dictionary to save
        save_name: Name of the file to save to
        recording_directory: The directory to record (or in this case save the .yaml) to

    Returns:

    """
    import yaml
    filename = os.path.join(recording_directory, save_name + ".yaml")
    dict_to_save = {}
    with open(filename, "w") as file:
        # make scalars readable
        for key, value in dictionary.items():
            if isinstance(value, ndarray):
                dict_to_save[key] = value.tolist()
            else:
                dict_to_save[key] = value
        yaml.dump(dict_to_save, file, sort_keys=True, indent=2)


def to_title(string: str) -> str:
    return string.title().replace("_", " ")