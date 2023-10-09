import os
import sys

import numpy as np
import torch
from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging

from modules.swarm_environments import MeshRefinement
from modules.swarm_environments import get_environments
from modules.swarm_environments.mesh.mesh_refinement import EvaluationFEMProblemCircularQueue

# path hacking for scripts from top level
current_directory = os.getcwd()
sys.path.append(current_directory)

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from util.function import get_from_nested_dict
from util.initialize_config import initialize_config
from util.keys import NUM_AGENTS
from util.types import *


def evaluate_config(config, num_envs: int):
    """
    Evaluate the oracle heuristic for a given config.
    Args:
        config: Config dict
        num_envs:

    Returns:

    """
    num_steps: int = get_from_nested_dict(dictionary=config,
                                          list_of_keys=["environment", "mesh_refinement", "num_timesteps"],
                                          raise_error=True)
    num_steps = num_steps - 1  # do not do final step as that always equals the reference mesh

    fem_config = get_from_nested_dict(config,
                                      list_of_keys=["environment", "mesh_refinement", "fem"],
                                      raise_error=True)
    fem_buffer = EvaluationFEMProblemCircularQueue(fem_config=fem_config,
                                                   random_state=np.random.RandomState(seed=123)
                                                   )

    data_array = np.empty((num_steps, num_envs, 4))

    evaluate_uniform(config=config,
                     data_array=data_array,
                     num_envs=num_envs,
                     fem_buffer=fem_buffer,
                     num_steps=num_steps)

    task_name: str = fem_config.get("pde_type")

    # save data!
    name = f"uniform"
    os.makedirs(f"evaluation_results/iclr2023/{task_name}", exist_ok=True)
    np.savez_compressed(
        f"evaluation_results/iclr2023/{task_name}/{task_name}_{name}.npz",
        **{f"idx={4100}_method={name}": data_array})
    print("Saved data! 🎉")


def evaluate_uniform(config, data_array, num_envs, fem_buffer, num_steps: int, seed: int = 123):
    print(f"Evaluating uniform meshes")
    environment, _, _ = get_environments(environment_config=config.get("environment"), seed=seed)
    environment: MeshRefinement
    environment.fem_problem_queue = fem_buffer
    for env_id in range(num_envs):
        print(f"  Evaluating environment #{env_id}")
        data_array[:, env_id, :] = uniform_rollout(environment, num_steps)


def uniform_rollout(environment, num_steps):
    environment.reset()
    metrics = np.empty((num_steps, 4)) * np.nan
    for step in range(num_steps):
        actions = np.ones(environment.num_agents, dtype=np.int32)
        observation, reward, done, additional_information = environment.step(action=actions)

        # store data in-place
        metrics[step] = np.array([additional_information.get(NUM_AGENTS),
                                  additional_information.get("squared_error"),
                                  additional_information.get("mean_error"),
                                  additional_information.get("top0.1_error")])
    return metrics


class IterativeExperiment(experiment.AbstractIterativeExperiment):
    def __init__(self):
        super(IterativeExperiment, self).__init__()
        self._algorithm: AbstractIterativeAlgorithm = None
        self._config: ConfigDict = None

    def initialize(self, config: ConfigDict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self._config = initialize_config(config=copy.deepcopy(config), repetition=rep)

        # initialize random seeds
        numpy_seed = self._config.get("random_seeds").get("numpy")
        pytorch_seed = self._config.get("random_seeds").get("pytorch")
        if numpy_seed is not None:
            np.random.seed(seed=numpy_seed)
        if pytorch_seed is not None:
            torch.manual_seed(seed=pytorch_seed)

        # initialize environment

    def iterate(self, _: ConfigDict, rep: int, n: int) -> ValueDict:
        num_envs = 100

        evaluate_config(config=self._config,
                        num_envs=num_envs)
        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':
    cluster_work.ClusterWork(IterativeExperiment).run()
