from typing import Callable, Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.swarm_environments import MeshRefinement
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.swarm_environments.util.keys import NUM_AGENTS, ELEMENT_PENALTY_LAMBDA
from modules.swarm_environments.util.torch_util import detach


def _single_rollout(environment: MeshRefinement,
                    policy_step_function: Callable) -> List[Dict[str, Any]]:
    """

    Args:
        environment:
        policy_step_function:

    Returns:

    """

    # reset environment and prepare loop over rollout
    _ = environment.reset()
    observation = environment.last_observation  # reset observation since the element penalty was changed
    done = False
    full_additional_information = []
    while not done:
        actions, values = policy_step_function(observation=observation)
        actions = detach(actions)
        observation, reward, done, additional_information = environment.step(action=actions)
        full_additional_information.append(additional_information)
    return full_additional_information


def _get_results_from_additional_information(last_additional_information):
    num_agents = last_additional_information.get(NUM_AGENTS)
    squared_error = last_additional_information.get("squared_error")
    log_squared_error = np.log(squared_error + 1e-16)
    current_results = [num_agents,
                       squared_error,
                       log_squared_error]
    return current_results


def _create_dataframe(experiment_results: np.array):
    num_penalties, num_pdes, _ = experiment_results.shape

    # Reshape the results to store all individual results in the dataframe
    data = experiment_results.reshape(num_penalties * num_pdes, -1)

    # Create a dataframe with multi-index for num_penalties and num_pdes
    index = pd.MultiIndex.from_product([range(num_penalties), range(num_pdes)], names=['penalty', 'pde'])
    columns = ["num_agents",
               "squared_error",
               "log_squared_error"]
    df = pd.DataFrame(data=data, index=index, columns=columns)

    # Cast specific columns to the desired dtype
    df["num_agents"] = df["num_agents"].astype(int)
    return df


def evaluate_mesh_refinement(policy_step_function: Callable,
                             environment: AbstractSwarmEnvironment,
                             num_pdes: int) -> pd.DataFrame:
    """

    Args:
        policy_step_function: A function that takes an observation and returns an action and a value.
        environment: The environment to evaluate.
        num_pdes: The number of pdes to evaluate.

    Returns: A dataframe with the results of the evaluation. Uses multi-indexing for the columns to store the
        num_penalties and num_pdes.

    """
    from modules.swarm_environments import SweepMeshRefinement
    from modules.swarm_environments.mesh.mesh_refinement import EvaluationFEMProblemCircularQueue
    fem_config = environment.fem_problem_queue.fem_config
    fem_config["num_pdes"] = num_pdes
    eval_queue = EvaluationFEMProblemCircularQueue(fem_config=fem_config,
                                                   random_state=np.random.RandomState(seed=123)
                                                   )
    environment.fem_problem_queue = eval_queue
    if isinstance(environment, SweepMeshRefinement):
        environment.train(False)

    # create empty array to store results
    experiment_results = np.empty((1, num_pdes, 3)) * np.nan

    with tqdm(total=num_pdes, desc="Evaluating Mesh Refinement") as pbar:
        for current_pde in range(num_pdes):
            # the inner loop will cycle through the pdes, so we do not need to take special care during the reset()

            full_additional_information = _single_rollout(policy_step_function=policy_step_function,
                                                          environment=environment)
            last_additional_information = full_additional_information[-1]
            current_results = _get_results_from_additional_information(last_additional_information)
            experiment_results[0, current_pde, :] = current_results

            pbar.update(1)

    dataframe = _create_dataframe(experiment_results=experiment_results)
    return dataframe
