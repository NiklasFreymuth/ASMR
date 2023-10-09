# Script to evaluate policies on a set of environments. In our setup, we have
# Tasks -> Configs -> Experiments -> Repetitions as our hierarchy
# Tasks is e.g., Poisson
# Configs is e.g., "poisson_asmr". All configs lead with a task name, followed by the algorithm/ablation
# Usually, each config has 1-4 "main" experiments, but each experiment is split into 10-15 element penalties,
#  so we expect 10-60 Experiments per config.
# Repetitions are the random seeds


import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from skfem.visuals.matplotlib import draw, plot
from tqdm import tqdm

# path hacking for scripts from top level
current_directory = os.getcwd()
sys.path.append(current_directory)

from asmr_evaluations.evaluation_util import single_rollout, get_config_path, get_config, \
    get_element_penalty_from_exp_name
from src.algorithms.baselines.sweep_dqn import SweepDQN
from src.algorithms.baselines.sweep_ppo import SweepPPO
from src.algorithms.get_algorithm import get_algorithm
from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm
from modules.swarm_environments import MeshRefinement
from modules.swarm_environments.mesh.mesh_refinement import EvaluationFEMProblemCircularQueue
from util.function import get_from_nested_dict
from util.types import *


class Evaluator:

    def __init__(self, num_pdes: int, seed: int = 123):
        self.num_pdes = num_pdes
        self.seed = seed  # set distinct seed.
        # Since this is only evaluation, the seed does not really matter except for the FEMProblemCircularQueue.
        # It should be fixed though, so that the same problems are used for all configs and algorithms.

        self._custom_eval_buffer = None

    def custom_eval_buffer(self, fem_config: ConfigDict):
        if self._custom_eval_buffer is None or self._reset_buffer:
            self._custom_eval_buffer = EvaluationFEMProblemCircularQueue(fem_config=fem_config,
                                                                         random_state=np.random.RandomState(
                                                                             seed=self.seed)
                                                                         )
        return self._custom_eval_buffer

    def evaluate(self, config_name: str, num_repetitions: int = 10, timeit: bool = False,
                 reset_buffer: bool = False, iteration: Optional[int] = None):
        self._reset_buffer = reset_buffer
        config_path = get_config_path(root="reports", config_name=config_name)

        data = {}
        final_checkpoint_exists_data = {}

        pde_type = None

        for experiment_position, experiment_folder in enumerate(os.listdir(config_path)):

            experiment_path = os.path.join(config_path, experiment_folder, "log")  # log folder contains logs

            print(f"Starting experiment {experiment_position + 1} of {len(os.listdir(config_path))}")
            print(f"Experiment: {experiment_folder}")

            experiment_results = np.empty((num_repetitions, self.num_pdes, 4)) * np.nan
            final_checkpoint_exists = np.zeros(num_repetitions, dtype=bool)
            # each experiment saves the following results:
            # shape: (num_repetitions, num_pdes, metrics),
            # where metrics = (num_agents, *[topk errors], mean_error, squared_error)

            for repetition_folder in sorted(os.listdir(experiment_path))[:num_repetitions]:
                start = time.time()
                repetition_folder: str
                repetition_idx = int(repetition_folder.split("_")[-1])
                try:
                    _final_checkpoint_exists, algorithm = self._init_algorithm(experiment_folder, experiment_path,
                                                                               repetition_folder,
                                                                               repetition_idx, iteration=iteration)
                    if pde_type is None:
                        pde_type = algorithm.config.get("environment").get("mesh_refinement").get("fem").get("pde_type")
                except IndexError:
                    print(f"Could not load algorithm for {experiment_folder}, {repetition_folder}. Skipping...")
                    continue

                # loop over self.num_pdes rollouts and take the average
                print(f"  Calculating evaluation metrics for repetition: {repetition_folder}")

                results = np.array([single_rollout(algorithm=algorithm)
                                    for _ in tqdm(range(self.num_pdes))])
                # shape: (num_pdes, metrics), where metrics = (error, num_agents)
                experiment_results[repetition_idx] = results
                final_checkpoint_exists[repetition_idx] = _final_checkpoint_exists

                avg_num_agents = np.mean(results[:, 0])
                squared_error = np.mean(results[:, 1])
                print(f"Squared error: {squared_error:.4f}, "
                      f"average number of agents: {avg_num_agents}")
                end = time.time()
                if timeit:
                    print(f"  Time for this repetition: {end - start}")

            data[experiment_folder] = experiment_results
            final_checkpoint_exists_data[experiment_folder] = final_checkpoint_exists

        # save the results
        # data is a dict with experiment names as keys and the results as values
        # where each result is a numpy array of shape (num_repetitions, num_pdes, metrics)

        os.makedirs(f"evaluation_results/iclr2023/{pde_type}", exist_ok=True)
        os.makedirs(f"evaluation_results/iclr2023/final_checkpoint_exists/{pde_type}", exist_ok=True)
        np.savez_compressed(
            f"evaluation_results/iclr2023/{pde_type}/{config_name}.npz",
            **data)
        np.savez_compressed(
            f"evaluation_results/iclr2023/final_checkpoint_exists/{config_name}_final_checkpoint_exists.npz",
            **final_checkpoint_exists_data)
        print("Saved data! 🎉")

    def _init_algorithm(self, experiment_folder, experiment_path, repetition_folder, repetition_idx, iteration: int):
        print(f"  Building config for repetition: {repetition_folder}")
        current_config = get_config(experiment_path=experiment_path,
                                    experiment_folder=experiment_folder,
                                    repetition_folder=repetition_folder,
                                    repetition_idx=repetition_idx,
                                    iteration=iteration,
                                    num_pdes=self.num_pdes)
        print(f"  Getting algorithm for repetition: {repetition_folder}")
        algorithm = self._get_algorithm(current_config)
        if isinstance(algorithm, (SweepPPO, SweepDQN)):
            print(f"   Setting environment to evaluation mode for DRL baseline")
            algorithm.environment.train(False)
        # check if the final checkpoint exists
        _final_checkpoint_exists = False
        for checkpoint in os.listdir(os.path.join(experiment_path, repetition_folder, "checkpoints")):
            if "final" in checkpoint:
                _final_checkpoint_exists = True
                break
        if not _final_checkpoint_exists:
            print(f"  Final checkpoint does not exist for repetition: {repetition_folder}")
        return _final_checkpoint_exists, algorithm

    def visualize(self, config_name: str, num_repetitions: int = 10, reset_buffer: bool = False, iteration: int = None):
        def visualize_solution(mesh, solution):
            ax = draw(mesh)
            plot(mesh, solution, ax=ax, shading='gouraud', colorbar=False)
            ax.patch.figure.axes[1].remove()
            plt.tight_layout()

        self._reset_buffer = reset_buffer

        config_path = get_config_path(root="reports", config_name=config_name)

        # create index key before each folder to sort the element penalties correctly by size and not alphabetically
        all_exps = os.listdir(config_path)

        eps = [get_element_penalty_from_exp_name(exp) for exp in all_exps]

        indices = np.argsort(eps)
        eps = [eps[i] for i in indices]
        all_exps = [all_exps[i] for i in indices]

        for experiment_position, experiment_folder in enumerate(all_exps):
            experiment_path = os.path.join(config_path, experiment_folder, "log")  # log folder contains logs

            print(f"Visualizing experiment {experiment_position + 1} of {len(all_exps)}")
            print(f"Experiment: {experiment_folder}")
            for repetition_folder in sorted(os.listdir(experiment_path))[:num_repetitions]:
                repetition_folder: str
                repetition_idx = int(repetition_folder.split("_")[-1])

                _, algorithm = self._init_algorithm(experiment_folder, experiment_path, repetition_folder,
                                                    repetition_idx, iteration=iteration)

                # create subfolder structure
                output_path = (f"evaluation_results/vis/iclr2023/{config_name}/"
                               f"{eps[experiment_position]}/{repetition_folder}")
                os.makedirs(output_path, exist_ok=True)

                for idx in range(self.num_pdes):
                    single_rollout(algorithm=algorithm)
                    # visualize solution
                    environment: MeshRefinement = algorithm.environment
                    mesh = environment.mesh
                    weighted_solution = environment.scalar_solution
                    if "linear_elasticity" in config_name:  # deform the mesh for a nicer visualization
                        basis = algorithm.environment.fem_problem._basis
                        displacement = algorithm.environment.fem_problem.fem_problem._displacement[basis.nodal_dofs]
                        deformed_mesh = mesh.translated(displacement)
                        visualize_solution(mesh=deformed_mesh, solution=weighted_solution)
                    else:
                        visualize_solution(mesh=mesh, solution=weighted_solution)
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

    def _get_algorithm(self, current_config):
        # build algorithm from checkpoint for current exp and rep,
        # overwrite FEMProblemCircularQueue to have fixed set of problems that is only computed once
        algorithm: AbstractRLAlgorithm = get_algorithm(current_config, seed=self.seed)
        fem_config = get_from_nested_dict(current_config,
                                          list_of_keys=["environment", "mesh_refinement", "fem"],
                                          raise_error=True)
        algorithm.environment.fem_problem_queue = self.custom_eval_buffer(fem_config=fem_config)
        return algorithm


def _get_args():
    import argparse
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", help="Name of the experiment config to evaluate.")
    parser.add_argument('-n', "--num_pdes", default=100, help="Number of PDEs to evaluate on.")
    parser.add_argument('-r', "--num_reps", default=10, help="Number of repetitions to evaluate on.")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Visualize the solutions instead of evaluating it.")
    parser.add_argument("-t", "--timeit", action="store_true", help="Time the evaluation.")
    parser.add_argument("-res", "--reset_buffer", action="store_true",
                        help="Reset the evaluation buffer after each repetition.")
    parser.add_argument("-i", "--iteration", default=None, help="Iteration to evaluate on.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()
    num_pdes = int(args.num_pdes)
    num_reps = int(args.num_reps)
    config_name = args.config_name
    visualize = args.visualize
    timeit = args.timeit
    reset_buffer = args.reset_buffer
    iteration = int(args.iteration) if args.iteration is not None else None

    evaluator = Evaluator(num_pdes=num_pdes, seed=123)
    if visualize:
        evaluator.visualize(config_name=config_name, num_repetitions=num_reps,
                            reset_buffer=reset_buffer, iteration=iteration)
    else:
        evaluator.evaluate(config_name=config_name, num_repetitions=num_reps, timeit=timeit,
                           reset_buffer=reset_buffer, iteration=iteration)
