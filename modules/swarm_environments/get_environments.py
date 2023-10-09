import copy
from typing import Dict, Any, List, Union, Optional, Tuple

from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.swarm_environments.environment_loader import EnvironmentLoader
from modules.swarm_environments.util.function import merge_nested_dictionaries, list_of_dicts_from_dict_of_lists


def get_environments(environment_config: Dict[Union[str, int], Any], seed: Optional[int] = None) -> Tuple[
    AbstractSwarmEnvironment, List[AbstractSwarmEnvironment], List[AbstractSwarmEnvironment]]:
    """
    Build and return a training and lists of evaluation and final environments for the task as specified in the config.

    Args:
        environment_config: The config as specified in the respective .yaml/.yml file.
          Assumes a structure of the form:
            environment_name: $environment_name
            environment_class: ...
            $environment_name:
                ...
          The environment config "$name" can contain an "evaluation" and "final"
          parameter that is used to create an evaluation environment that is different from the training env.
          The "evaluation" and "final" environments only specify the *differences* between the train and
           evaluation/final environments. If it is empty, a single evaluation and final environment will be created
           with parameters equal to the training one.
          You can specify more than one evaluation/final environment by using lists of values, where each list must have
          the same length. In this case, the evaluation/final environments will be lists of environments, where each
          environment is built using the parameters specified in the respective position of the lists.
          If you want to use a parameter that is a list in itself, you can do so outside the evaluation/final scope via
          a single list, and inside it via a list of lists. Here, the outer list will be interpreted as a list over
          environments, and the inner over a list of parameters for the environment.
        seed: Used to initialize the training environment. If None, will not set a seed. The environments use
          internal numpy random states, so you might want to set a seed here if you want to reproduce results.

    Returns:
        A tuple (training_environment, evaluation_environments, final_environments).
    """
    environment_name = environment_config["environment"]
    raw_environment_params = environment_config[environment_name]
    environment_class_name = environment_config.get("environment_class", environment_name)

    clean_base_params = get_clean_base_params(raw_environment_params, keys=["evaluation", "final"])

    # Extract evaluation and final parameters
    evaluation_environment_param_list = extract_environment_param_list(raw_environment_params.get("evaluation"),
                                                                       base_params=clean_base_params)
    final_environment_param_list = extract_environment_param_list(raw_environment_params.get("final"),
                                                                  base_params=clean_base_params)

    env_loader = EnvironmentLoader()
    training_environment = env_loader.create(environment_class_name, environment_config=clean_base_params, seed=seed)

    evaluation_environments = [env_loader.create(environment_class_name,
                                                 environment_config=environment_params,
                                                 seed=environment_params.get("seed", position + seed))
                               for position, environment_params in enumerate(evaluation_environment_param_list)]

    final_environments = [env_loader.create(environment_class_name,
                                            environment_config=environment_params,
                                            seed=123)
                          for position, environment_params in enumerate(final_environment_param_list)]

    return training_environment, evaluation_environments, final_environments

def get_clean_base_params(environment_params: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Returns the environment parameters without the provided keys."""
    return {k: v for k, v in environment_params.items() if k not in keys}


def extract_environment_param_list(subenvironment_params: Dict[str, Any],
                                   base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts specific environment parameters and returns a list of extracted parameters."""
    if subenvironment_params is not None and len(subenvironment_params) > 0:
        extracted_params = list_of_dicts_from_dict_of_lists(subenvironment_params)
        extracted_param_list = [merge_nested_dictionaries(destination=base_params, source=param) for param in
                                extracted_params]
    else:
        extracted_param_list = [base_params]

    return extracted_param_list
