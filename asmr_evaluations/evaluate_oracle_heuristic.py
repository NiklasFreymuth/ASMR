import os

import numpy as np
import torch
import tqdm
from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging
from matplotlib import pyplot as plt
from skfem.visuals.matplotlib import draw, plot

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.environments.get_environment import get_environment
from src.environments.mesh.mesh_refinement.mesh_refinement import MeshRefinement
from src.environments.mesh.mesh_refinement.problems.fem_buffer import FEMBuffer
from src.environments.mesh.mesh_refinement.remeshing_heuristic import RemeshingHeuristic
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


def evaluate_config(all_thetas, config, num_envs: int, visualize: bool):
    """
    Evaluate the oracle heuristic for a given config.
    Args:
        all_thetas: List of thetas to evaluate
        config: Config dict
        num_envs:

    Returns:

    """
    task_name: str = get_from_nested_dict(dictionary=config,
                                          list_of_keys=["task", "mesh_refinement", "fem", "pde_type"],
                                          raise_error=True)
    num_evals = len(all_thetas)

    fem_config = get_from_nested_dict(config,
                                      list_of_keys=["task", "mesh_refinement", "fem"],
                                      raise_error=True)
    fem_buffer = EvaluationFEMBuffer(fem_config=fem_config,
                                     random_state=np.random.RandomState(seed=123)
                                     )

    data_array = np.empty((num_evals, num_envs, 4))
    for theta_id, theta in tqdm.tqdm(enumerate(all_thetas)):
        evaluate_theta(config=config,
                       data_array=data_array,
                       num_envs=num_envs,
                       theta=theta,
                       theta_id=theta_id,
                       visualize=visualize,
                       fem_buffer=fem_buffer)
    if not visualize:
        # save data!
        if fem_config.get("error_metric") == "maximum":
            name = "maximum_oracle.i17"
        else:
            name = "oracle.i16"
        os.makedirs(f"evaluation_results/neurips/{task_name}", exist_ok=True)
        np.savez_compressed(f"evaluation_results/neurips/{task_name}/{name}.npz",
                            **{name: data_array})
        print("Saved data! 🎉")


def evaluate_theta(config, data_array, num_envs, theta, theta_id, fem_buffer, visualize: bool, seed: int = 123):
    print(f"Evaluating theta: {theta}")
    heuristic = RemeshingHeuristic(theta=theta, area_scaling=False)
    environment, _ = get_environment(config=config, seed=seed)
    environment: MeshRefinement
    environment.fem_problem = fem_buffer
    for env_id in range(num_envs):
        print(f"  Evaluating environment #{env_id}")

        # loop over rollout
        additional_information = None

        environment.reset()
        done = False
        while not done:
            error_per_element = environment.error_per_element
            error_per_element = np.dot(error_per_element, environment.solution_dimension_weights)

            actions = heuristic(error_per_element=error_per_element, element_areas=environment.element_areas)

            observation, reward, done, additional_information = environment.step(action=actions)

        if visualize:
            # create subfolder structure
            fem_config = get_from_nested_dict(dictionary=config,
                                              list_of_keys=["task", "mesh_refinement", "fem"],
                                              raise_error=True)
            if fem_config.get("error_metric") == "maximum":
                name = "maximum_oracle"
            else:
                name = "oracle"
            task_name: str = fem_config.get("pde_type")
            output_path = f"evaluation_results/vis/neurips/{task_name}/{name}/theta_{theta}"
            os.makedirs(output_path, exist_ok=True)
            _visualize_and_save_solution(environment=environment, idx=env_id, output_path=output_path)

        # store data in-place
        data_array[theta_id, env_id, 0] = additional_information.get(NUM_AGENTS)
        data_array[theta_id, env_id, 1] = additional_information.get("top0.1_error")
        data_array[theta_id, env_id, 2] = additional_information.get("top5_error")
        data_array[theta_id, env_id, 3] = additional_information.get("mean_error")

        print(f"    Number of agents: {data_array[theta_id, env_id, 0]}")
        print(f"    Top 0.1 error: {data_array[theta_id, env_id, 2]}")
        print(f"    Top 5 error: {data_array[theta_id, env_id, 4]}")
        print(f"    Mean error: {data_array[theta_id, env_id, 6]}")


def _visualize_and_save_solution(environment, idx, output_path):
    # visualize solution
    print(f"    Visualizing solution #{idx}")
    mesh = environment.mesh
    weighted_solution = np.dot(environment.solution, environment.solution_dimension_weights)
    _visualize_solution(mesh=mesh, solution=weighted_solution)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(output_path, f"pde_{idx:.2f}.pdf"), bbox_inches='tight',
                pad_inches=0,
                transparent=True)
    # plt.show()
    plt.clf()
    plt.close()


def _visualize_solution(mesh, solution):
    ax = draw(mesh)
    plot(mesh, solution, ax=ax, shading='gouraud', colorbar=False)
    ax.patch.figure.axes[1].remove()
    plt.tight_layout()


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
        visualize: bool = True
        if visualize:
            num_envs = 10
            num_evals = 10
        else:
            num_envs = 100
            num_evals = 100

        all_thetas = np.linspace(0, 1, num_evals + 1)[1:][::-1]
        evaluate_config(all_thetas=all_thetas,
                        config=self._config,
                        num_envs=num_envs, visualize=visualize)
        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':
    cluster_work.ClusterWork(IterativeExperiment).run()
