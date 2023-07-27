import os

import numpy as np
import torch
from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.environments.get_environment import get_environment
from src.environments.mesh.mesh_refinement.mesh_refinement import MeshRefinement
from src.environments.mesh.mesh_refinement.problems.fem_buffer import FEMBuffer
from util.function import get_from_nested_dict
from util.initialize_config import initialize_config
from util.keys import NUM_AGENTS
from util.types import *


class EvaluationFEMBuffer(FEMBuffer):
    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        super().__init__(fem_config=fem_config, random_state=random_state)
        self._index_sampler = None
        num_pdes = fem_config.get("num_pdes")  # number of pdes to store. None, 0 or -1 means infinite
        self._current_index = 0
        self._max_index = num_pdes

    def next(self) -> None:
        """
        Draws the next finite element problem. This method is called at the beginning of each episode and draws a
        (potentially new) finite element problem from the buffer.
        Returns:

        """
        pde_idx = self._current_index % self._max_index
        self._current_index += 1

        self._next_from_idx(pde_idx=pde_idx)

    @property
    def num_pdes(self):
        return self._max_index


def evaluate_config(config, num_envs: int):
    """
    Evaluate the oracle heuristic for a given config.
    Args:
        config: Config dict
        num_envs:

    Returns:

    """
    task_name: str = get_from_nested_dict(dictionary=config,
                                          list_of_keys=["task", "mesh_refinement", "fem", "pde_type"],
                                          raise_error=True)
    num_steps: int = get_from_nested_dict(dictionary=config,
                                          list_of_keys=["task", "mesh_refinement", "num_timesteps"],
                                          raise_error=True)
    num_steps = num_steps - 1  # do not do final step as that always equals the reference mesh

    fem_config = get_from_nested_dict(config,
                                      list_of_keys=["task", "mesh_refinement", "fem"],
                                      raise_error=True)
    fem_buffer = EvaluationFEMBuffer(fem_config=fem_config,
                                     random_state=np.random.RandomState(seed=123)
                                     )

    data_array = np.empty((num_steps, num_envs, 4))

    evaluate_uniform(config=config,
                     data_array=data_array,
                     num_envs=num_envs,
                     fem_buffer=fem_buffer,
                     num_steps=num_steps)

    # save data!
    name = "uniform.i50"
    filename = "uniform"
    os.makedirs(f"evaluation_results/neurips/{task_name}", exist_ok=True)
    np.savez_compressed(
        f"evaluation_results/neurips/{task_name}/{task_name}_{filename}.npz",
        **{name: data_array})
    print("Saved data! 🎉")


def evaluate_uniform(config, data_array, num_envs, fem_buffer, num_steps: int, seed: int = 123):
    print(f"Evaluating uniform meshes")
    environment, _ = get_environment(config=config, seed=seed)
    environment: MeshRefinement
    environment.fem_problem = fem_buffer
    for env_id in range(num_envs):
        print(f"  Evaluating environment #{env_id}")

        # loop over rollout
        additional_information = None

        environment.reset()
        for step in range(num_steps):
            actions = np.ones(environment.num_elements, dtype=np.int32)
            observation, reward, done, additional_information = environment.step(action=actions)

            # store data in-place
            data_array[step, env_id, 0] = additional_information.get(NUM_AGENTS)
            data_array[step, env_id, 2] = additional_information.get("top0.1_error")
            data_array[step, env_id, 4] = additional_information.get("top5_error")
            data_array[step, env_id, 6] = additional_information.get("mean_error")

            print(f"Step:  {step}")
            print(f"    Number of agents: {data_array[step, env_id, 0]}")
            print(f"    Top 0.1 error: {data_array[step, env_id, 1]}")
            print(f"    Top 5 error: {data_array[step, env_id, 2]}")
            print(f"    Mean error: {data_array[step, env_id, 3]}")


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
