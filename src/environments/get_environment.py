from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.environments.environment_loader import EnvironmentLoader
from util.function import get_from_nested_dict, merge_nested_dictionaries, list_of_dicts_from_dict_of_lists
from util.types import *


def get_environment(config: ConfigDict, seed: Optional[int] = None) \
        -> Tuple[AbstractSwarmEnvironment, List[AbstractSwarmEnvironment]]:
    """
    Build and return a training and an evaluation environment for the task as specified in the config
    Args:
        config: The config as specified in the respective .yaml file. Assumes a sub-configs ["task", environment_name]
          that specify how exactly to build the environments. This environment sub-config can contain an "evaluation"
          parameter that is used to create an evaluation environment that is different from the testing one.
          The "evaluation" environments only specifies *the differences* between the train and evaluation environments.
          If it is empty, the evaluation environment will use the same parameters as the training one.
          You can specify more than one evaluation environment by using lists of values, where each list must have
          the same length. In this case, the evaluation environments will be a list of environments, where each
          environment is built using the parameters specified in the respective position of the lists.
          If you want to use a parameter that is a list in itself, you can do so outside the evaluation scope via
          a single list, and inside it via a list of lists. Here, the outer list will be interpreted as a list over
          environments, and the inner over a list of parameters for the environment
        seed: Used to initialize the training environment

    Returns: A tuple (environment, evaluation_environments), where the evaluation environments may be specified
    differently from the original environment.

    """
    environment_name = get_from_nested_dict(config, list_of_keys=["task", "environment"], raise_error=True)
    environment_class_name = get_from_nested_dict(config, list_of_keys=["task", "environment_class"],
                                                  default_return=environment_name)
    environment_params = get_from_nested_dict(config, list_of_keys=["task", environment_name], raise_error=True)

    if "evaluation" in environment_params.keys():  # extract evaluation params
        evaluation_environment_params = environment_params.get("evaluation")
        clean_environment_params = copy.deepcopy(environment_params)
        del clean_environment_params["evaluation"]
        evaluation_environment_param_list = list_of_dicts_from_dict_of_lists(evaluation_environment_params)
        evaluation_environment_param_list = [merge_nested_dictionaries(destination=clean_environment_params,
                                                                       source=evaluation_params)
                                             for evaluation_params in evaluation_environment_param_list]
    else:
        clean_environment_params = environment_params
        evaluation_environment_param_list = [clean_environment_params]

    # get environment class
    environment_class = EnvironmentLoader().load(environment_class_name)

    environment: AbstractSwarmEnvironment = environment_class(environment_config=clean_environment_params, seed=seed)

    evaluation_environments = [
        environment_class(environment_config=environment_params, seed=environment_params.get("seed", position + seed))
        for position, environment_params in enumerate(evaluation_environment_param_list)]

    return environment, evaluation_environments
