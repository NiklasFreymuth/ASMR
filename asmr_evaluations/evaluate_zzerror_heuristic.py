import os
import sys

# path hacking for scripts from top level
current_directory = os.getcwd()
sys.path.append(current_directory)

import numpy as np
import torch
import tqdm
from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging
from matplotlib import pyplot as plt
from skfem.visuals.matplotlib import draw, plot
from modules.swarm_environments import MeshRefinement
from modules.swarm_environments import get_environments

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from util.function import get_from_nested_dict
from util.initialize_config import initialize_config
from util.keys import NUM_AGENTS
from util.types import *

from modules.swarm_environments.mesh.mesh_refinement import EvaluationFEMProblemCircularQueue


def evaluate_config(all_thetas, config, num_envs: int,
                    uniform_refs: int,
                    visualize: bool):
    """
    Evaluate the oracle heuristic for a given config.
    Args:
        all_thetas: List of thetas to evaluate
        config: Config dict
        num_envs:

    Returns:

    """
    num_evals = len(all_thetas)

    fem_config = get_from_nested_dict(config,
                                      list_of_keys=["environment", "mesh_refinement", "fem"],
                                      raise_error=True)
    fem_buffer = EvaluationFEMProblemCircularQueue(fem_config=fem_config,
                                                   random_state=np.random.RandomState(seed=123)
                                                   )

    data_array = np.empty((num_evals, num_envs, 4))
    for theta_id, theta in tqdm.tqdm(enumerate(all_thetas)):
        evaluate_theta(config=config,
                       data_array=data_array,
                       num_envs=num_envs,
                       theta=theta,
                       theta_id=theta_id,
                       uniform_refs=uniform_refs,
                       visualize=visualize,
                       fem_buffer=fem_buffer)
    return data_array


def evaluate_theta(config, data_array, num_envs, theta, theta_id, fem_buffer,
                   uniform_refs: int,
                   visualize: bool, seed: int = 123):
    print(f"Evaluating theta: {theta}")
    from modules.swarm_environments.mesh.mesh_refinement.remeshing_heuristics import ZienkiewiczZhuErrorHeuristic
    heuristic = ZienkiewiczZhuErrorHeuristic(theta=theta, refinement_strategy="absolute")
    environment, _, _ = get_environments(environment_config=config.get("environment"), seed=seed)
    environment: MeshRefinement
    environment.fem_problem_queue = fem_buffer
    for env_id in range(num_envs):
        print(f"  Evaluating environment #{env_id}")

        # loop over rollout
        additional_information = None

        environment.reset()
        done = False
        i = 0
        while not done:
            if i < uniform_refs:
                actions = np.ones(environment.num_agents)
                # do initial uniform refinement
            else:
                actions = heuristic(mesh=environment.mesh,
                                    solution=environment.scalar_solution)
            i += 1

            observation, reward, done, additional_information = environment.step(action=actions)

        if visualize:
            # create subfolder structure
            task_name: str = get_from_nested_dict(dictionary=config,
                                                  list_of_keys=["environment", "mesh_refinement", "fem", "pde_type"],
                                                  raise_error=True)
            output_path = f"evaluation_results/vis/iclr2023/{task_name}_zzerror/{theta:.2e}"
            os.makedirs(output_path, exist_ok=True)
            _visualize_and_save_solution(environment=environment, idx=env_id, output_path=output_path)

        # store data in-place
        data_array[theta_id, env_id, 0] = additional_information.get(NUM_AGENTS)
        data_array[theta_id, env_id, 1] = additional_information.get("squared_error")
        data_array[theta_id, env_id, 1] = additional_information.get("mean_error")
        data_array[theta_id, env_id, 1] = additional_information.get("top0.1_error")



        print(f"    Number of agents: {data_array[theta_id, env_id, 0]}")
        print(f"    Squared error: {data_array[theta_id, env_id, 1]}")


def _visualize_and_save_solution(environment, idx: int, output_path):
    # visualize solution
    print(f"    Visualizing solution #{idx}")
    mesh = environment.mesh
    weighted_solution = environment.scalar_solution
    _visualize_solution(mesh=mesh, solution=weighted_solution)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(output_path, f"pde_{idx}.pdf"), bbox_inches='tight',
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

    def iterate(self, _: ConfigDict, rep: int, n: int) -> ValueDict:
        visualize: bool = False
        if visualize:
            num_envs = 10
            num_evals = 10
            all_thetas = np.logspace(np.log10(0.001), np.log10(1.0), num_evals, endpoint=False)
            num_uniform_refs = [2]
        else:
            num_envs = 100
            num_evals = 100
            all_thetas = np.logspace(np.log10(0.001), np.log10(1.0), num_evals)
            num_uniform_refs = [0, 1, 2]
        self._config["environment"]["mesh_refinement"]["fem"]["num_pdes"] = num_envs

        eval_config = {}
        for uniform_refs in num_uniform_refs:
            # sample more thetas in the beginning, and "oversample" near 0.0 by sampling from a logarithmic
            # distribution
            data_array = evaluate_config(all_thetas=all_thetas,
                                         config=self._config,
                                         num_envs=num_envs,
                                         uniform_refs=uniform_refs,
                                         visualize=visualize)

            idx = 3000 + uniform_refs
            if self._config["environment"]["mesh_refinement"]["num_timesteps"] == 4:
                idx = idx + 3
                eval_config[f"idx={idx}_method=zz_absolute{uniform_refs}_small"] = data_array
            else:
                eval_config[f"idx={idx}_method=zz_absolute{uniform_refs}"] = data_array
        if not visualize:
            # save data!
            task_name: str = get_from_nested_dict(dictionary=self._config,
                                                  list_of_keys=["environment", "mesh_refinement", "fem", "pde_type"],
                                                  raise_error=True)
            os.makedirs(f"evaluation_results/iclr2023/{task_name}", exist_ok=True)
            if self._config["environment"]["mesh_refinement"]["num_timesteps"] == 4:
                filename = f"{task_name}_zz_absolute_small"
            else:
                filename = f"{task_name}_zz_absolute"
            np.savez_compressed(f"evaluation_results/iclr2023/{task_name}/filename.npz",
                                **eval_config)
            print("Saved data! 🎉")
        return {}

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':
    cluster_work.ClusterWork(IterativeExperiment).run()
