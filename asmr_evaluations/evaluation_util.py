import os
from typing import Optional

import yaml
from modules.swarm_environments import MeshRefinement

from util.keys import NUM_AGENTS
from util.torch_util.torch_util import detach
from util.types import ConfigDict


def single_rollout(algorithm):
    """
    Do a single rollout and return the remaining error and number of agents.
    Returns:

    """
    environment: MeshRefinement = algorithm.environment

    # reset environment and prepare loop over rollout
    observation = environment.reset()
    done = False
    additional_information = {}
    while not done:
        actions, values = algorithm.policy_step(observation=observation)
        actions = detach(actions)
        observation, reward, done, additional_information = environment.step(action=actions)
    num_agents = additional_information.get(NUM_AGENTS)
    squared_error = additional_information.get("squared_error")
    mean_error = additional_information.get("mean_error")
    top0_1_error = additional_information.get("top0.1_error")
    return num_agents, squared_error, mean_error, top0_1_error


def get_config_path(root, config_name: str):
    path = os.path.join(root, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' not found."
                                f"Make sure to start this script from the root folder 'DAVIS' of the project")
    return path


def get_config(experiment_path, experiment_folder, repetition_folder: str,
               repetition_idx: int, iteration: Optional[int], num_pdes: int) -> ConfigDict:
    repetition_path = os.path.join(experiment_path, repetition_folder)
    current_config_path = os.path.join(repetition_path, "config.yaml")
    with open(current_config_path) as file:
        current_config = yaml.safe_load(file)
    current_config["environment"]["mesh_refinement"]["fem"]["num_pdes"] = num_pdes
    current_config["algorithm"]["checkpoint"] = {
        "experiment_name": experiment_folder,
        "iteration": iteration,
        "repetition": repetition_idx,
    }
    return current_config


def get_element_penalty_from_exp_name(exp_name):
    # asmr, vdgn cases
    if "penalty" in exp_name:
        current_ep = float(exp_name[exp_name.rfind("penalty=") + 8:])
    elif "mes.ele.v" in exp_name:
        current_ep = exp_name[exp_name.rfind(".mes.ele.v") + 10:]
        current_ep = float(current_ep.split("_")[0])

    elif "mes.nt" in exp_name:  # single agent
        current_ep = float(exp_name[exp_name.rfind(".mes.nt") + 7:])
    elif "mes.me" in exp_name:  # sweep
        current_ep = float(exp_name[exp_name.rfind(".mes.me") + 7:])
        print("@@@", current_ep)
    else:
        raise ValueError(f"Unknown experiment name '{exp_name}'")
    return current_ep
