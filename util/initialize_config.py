from util.types import *
import copy


def initialize_config(config: ConfigDict, repetition: int) -> ConfigDict:
    recording_structure = _get_recording_structure(config=config, repetition=repetition)
    iterations = config.get("iterations")
    config = config.get("params")  # move into custom params
    assert "_recording_structure" not in config, "May not use pre-defined '_recording_structure' subconfig"
    config["_recording_structure"] = recording_structure
    assert "iterations" not in config, "Iterations must be defined outside of 'params'."
    config["iterations"] = iterations
    config["random_seeds"] = _get_random_seeds(random_seeds=config.get("random_seeds"), repetition=repetition)
    config = _process_config_keys(current_config=config)
    return config


def _get_recording_structure(config: ConfigDict, repetition: int) -> Dict[Key, str]:
    rep_log_path = config.get("_rep_log_path")
    experiment_name = config.get("_experiment_name")
    job_name = config.get("name")
    return {
        "_groupname": experiment_name,
        "_runname": experiment_name + "_" + str(repetition),
        "_recording_dir": rep_log_path,
        "_job_name": job_name
    }


def _get_random_seeds(random_seeds: Dict[Key, Optional[Key]], repetition: int) -> Dict[Key, Optional[Key]]:
    numpy_seed = random_seeds.get("numpy")
    if numpy_seed == "default":
        random_seeds["numpy"] = repetition

    pytorch_seed = random_seeds.get("pytorch")
    if pytorch_seed == "default":
        random_seeds["pytorch"] = repetition
    elif pytorch_seed == "tied":
        random_seeds["pytorch"] = random_seeds.get("numpy")
    return random_seeds


def _process_config_keys(current_config: ConfigDict, full_config: Optional[ConfigDict] = None) -> ConfigDict:
    """
    Recursively parses a given dictionary by going through its keys and adapting the values into a suitable and most
    importantly standardized format
    Args:
        current_config: The current (sub-)config
        full_config: The full config for reference. Used for filtering out subconfigs that are not needed for the
        specific run

    Returns: The parsed dictionary

    """
    if full_config is None:
        full_config = copy.deepcopy(current_config)
    parsed_config = {}
    for key, value in current_config.items():
        # assure that logs happen last
        if isinstance(value, dict):
            parsed_config[key] = _process_config_keys(current_config=value, full_config=full_config)
        elif key.startswith("log_") and isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, int) and value > 0:
                parsed_value = int(2 ** value)
            elif isinstance(value, int) and value < -30:  # round very small values to 0
                parsed_value = 0
            else:
                parsed_value = 2 ** value

            parsed_config[key.replace("log_", "", 1)] = parsed_value
        elif isinstance(value, float) and value.is_integer():
            parsed_config[key] = int(value)  # parse to int when possible
        else:
            parsed_config[key] = value
    return parsed_config
