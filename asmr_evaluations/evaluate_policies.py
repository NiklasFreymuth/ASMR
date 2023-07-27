# Script to evaluate policies on a set of environments. In our setup, we have
# Tasks -> Configs -> Experiments -> Repetitions as our hierarchy
# Tasks is e.g., Poisson
# Configs is e.g., "poisson_asmr". All configs lead with a task name, followed by the algorithm/ablation
# Usually, each config has 1-4 "main" experiments, but each experiment is split into 10-15 element penalties,
#  so we expect 10-60 Experiments per config.
# Repetitions are the random seeds


import os
import time

import numpy as np
import yaml
from matplotlib import pyplot as plt
from skfem.visuals.matplotlib import draw, plot
from tqdm import tqdm

from src.algorithms.baselines.sweep_dqn import SweepDQN
from src.algorithms.baselines.sweep_ppo import SweepPPO
from src.algorithms.get_algorithm import get_algorithm
from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm
from src.environments.mesh.mesh_refinement.mesh_refinement import MeshRefinement
from src.environments.mesh.mesh_refinement.problems.fem_buffer import FEMBuffer
from util.function import get_from_nested_dict
from util.keys import REMAINING_ERROR, NUM_AGENTS
from util.torch_util.torch_util import detach
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


class Evaluator:

    def __init__(self, num_pdes: int, seed: int = 123):
        self.num_pdes = num_pdes
        self.seed = seed  # set distinct seed.
        # Since this is only evaluation, the seed does not really matter except for the FEMBuffer.
        # It should be fixed though, so that the same problems are used for all configs and algorithms.

        self._custom_eval_buffer = None

    def custom_eval_buffer(self, fem_config: ConfigDict):
        if self._custom_eval_buffer is None or self._reset_buffer:
            self._custom_eval_buffer = EvaluationFEMBuffer(fem_config=fem_config,
                                                           random_state=np.random.RandomState(seed=self.seed)
                                                           )
        return self._custom_eval_buffer

    def evaluate(self, config_name: str, num_repetitions: int = 10, timeit: bool = False,
                 reset_buffer: bool = False):
        self._reset_buffer = reset_buffer
        config_path = self._get_config_path(root="reports", config_name=config_name)

        data = {}
        final_checkpoint_exists_data = {}

        for experiment_position, experiment_folder in enumerate(os.listdir(config_path)):

            experiment_path = os.path.join(config_path, experiment_folder, "log")  # log folder contains logs

            print(f"Starting experiment {experiment_position + 1} of {len(os.listdir(config_path))}")
            print(f"Experiment: {experiment_folder}")

            experiment_results = np.empty((num_repetitions, self.num_pdes, 4)) * np.nan
            final_checkpoint_exists = np.zeros(num_repetitions, dtype=bool)
            # each experiment saves the following results:
            # shape: (num_repetitions, num_pdes, metrics), where metrics = (num_agents, [error metrics])
            # [error metrics] = [top_01_error, top_5_error, mean_error]

            for repetition_folder in sorted(os.listdir(experiment_path))[:num_repetitions]:
                start = time.time()
                repetition_folder: str
                repetition_idx = int(repetition_folder.split("_")[-1])
                try:
                    _final_checkpoint_exists, algorithm = self._init_algorithm(experiment_folder, experiment_path,
                                                                               repetition_folder,
                                                                               repetition_idx)
                except IndexError:
                    print(f"Could not load algorithm for {experiment_folder}, {repetition_folder}. Skipping...")
                    continue

                # loop over self.num_pdes rollouts and take the average
                print(f"  Calculating evaluation metrics for repetition: {repetition_folder}")

                results = np.array([self._single_rollout(algorithm=algorithm, idx=idx)
                                    for idx in tqdm(range(self.num_pdes))])
                # shape: (num_pdes, metrics), where metrics = (error, num_agents)
                experiment_results[repetition_idx] = results
                final_checkpoint_exists[repetition_idx] = _final_checkpoint_exists

                avg_num_agents = np.mean(results[:, 0])
                avg_top_01_error = np.mean(results[:, 1])
                avg_top_5_error = np.mean(results[:, 2])
                mean_error = np.mean(results[:, 3])
                print(f"  Average number of agents: {avg_num_agents}")
                print(f"  Average top 0.1%/5%/mean error:"
                      f" {avg_top_01_error:.4f}/{avg_top_5_error:.4f}/{mean_error:.4f}")
                end = time.time()
                if timeit:
                    print(f"  Time for this repetition: {end - start}")

            data[experiment_folder] = experiment_results
            final_checkpoint_exists_data[experiment_folder] = final_checkpoint_exists

        # save the results
        # data is a dict with experiment names as keys and the results as values
        # where each result is a numpy array of shape (num_repetitions, num_pdes, metrics)

        os.makedirs(f"evaluation_results/neurips/{config_name.rsplit('_', 1)[0]}", exist_ok=True)
        np.savez_compressed(
            f"evaluation_results/neurips/{config_name.rsplit('_', 1)[0]}/{config_name}.npz",
            **data)
        print("Saved data! 🎉")
        # plot the results


    def _init_algorithm(self, experiment_folder, experiment_path, repetition_folder, repetition_idx):
        print(f"  Building config for repetition: {repetition_folder}")
        current_config = self._get_config(experiment_path=experiment_path,
                                          experiment_folder=experiment_folder,
                                          repetition_folder=repetition_folder,
                                          repetition_idx=repetition_idx)
        print(f"  Getting algorithm for repetition: {repetition_folder}")
        algorithm = self._get_algorithm(current_config)
        if isinstance(algorithm, (SweepPPO, SweepDQN)):
            print(f"   Setting environment to evaluation mode for the Sweep baseline")
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

    def visualize(self, config_name: str, num_repetitions: int = 10, reset_buffer: bool = False):
        def visualize_solution(mesh, solution):
            ax = draw(mesh)
            plot(mesh, solution, ax=ax, shading='gouraud', colorbar=False)
            ax.patch.figure.axes[1].remove()
            plt.tight_layout()

        self._reset_buffer = reset_buffer

        config_path = self._get_config_path(root="reports", config_name=config_name)

        # create index key before each folder to sort the element penalties correctly by size and not alphabetically
        all_exps = os.listdir(config_path)
        if config_name.endswith("asmr") or config_name.endswith("vdgn"):
            key_word = ".ep"
        elif config_name.endswith("argmax"):
            key_word = ".nt"
        elif config_name.endswith("sweep"):
            key_word = ".me"
        else:
            raise ValueError("Cannot find method to select the element penalty keword")
        eps = [float(f[f.rfind(key_word) + len(key_word):]) for f in all_exps]
        indices = np.argsort(eps)
        all_exps = [all_exps[i] for i in indices]
        print("stop")

        for experiment_position, experiment_folder in enumerate(all_exps):
            experiment_path = os.path.join(config_path, experiment_folder, "log")  # log folder contains logs

            print(f"Visualizing experiment {experiment_position + 1} of {len(all_exps)}")
            print(f"Experiment: {experiment_folder}")
            for repetition_folder in sorted(os.listdir(experiment_path))[:num_repetitions]:
                repetition_folder: str
                repetition_idx = int(repetition_folder.split("_")[-1])

                _, algorithm = self._init_algorithm(experiment_folder, experiment_path, repetition_folder,
                                                    repetition_idx)

                # create subfolder structure
                output_path = f"evaluation_results/vis/neurips/{config_name}/{experiment_position:02d}_{experiment_folder}/{repetition_folder}"
                os.makedirs(output_path, exist_ok=True)

                for idx in range(self.num_pdes):
                    self._single_rollout(algorithm=algorithm, idx=idx)
                    # visualize solution
                    if "linear_elasticity" in config_name:
                        environment: MeshRefinement = algorithm.environment
                        basis = algorithm.environment.fem_problem._current_basis
                        mesh = basis.mesh
                        deformed_mesh = mesh.translated(
                            algorithm.environment.fem_problem.current_fem_problem._displacement[
                                basis.nodal_dofs])  # deformed mesh
                        weighted_solution = np.dot(environment.solution, environment.solution_dimension_weights)
                        visualize_solution(mesh=deformed_mesh, solution=weighted_solution)
                    else:
                        environment: MeshRefinement = algorithm.environment
                        mesh = environment.mesh
                        weighted_solution = np.dot(environment.solution, environment.solution_dimension_weights)
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
        # overwrite FEMBuffer to have fixed set of problems that is only computed once
        algorithm: AbstractRLAlgorithm = get_algorithm(current_config, seed=self.seed)
        fem_config = get_from_nested_dict(current_config,
                                          list_of_keys=["task", "mesh_refinement", "fem"],
                                          raise_error=True)
        algorithm.environment.fem_problem = self.custom_eval_buffer(fem_config=fem_config)
        return algorithm

    def _get_config(self, experiment_path, experiment_folder, repetition_folder: str,
                    repetition_idx: int) -> ConfigDict:
        repetition_path = os.path.join(experiment_path, repetition_folder)
        current_config_path = os.path.join(repetition_path, "config.yaml")
        with open(current_config_path) as file:
            current_config = yaml.safe_load(file)
        current_config["task"]["mesh_refinement"]["fem"]["num_pdes"] = self.num_pdes
        current_config["algorithm"]["checkpoint"] = {
            "experiment_name": experiment_folder,
            "iteration": None,
            "repetition": repetition_idx,
        }
        return current_config

    def _get_config_path(self, root, config_name: str):
        path = os.path.join(root, config_name)
        if not os.path.exists(path):
            raise FileNotFoundError("Make sure to start this script from the root folder 'ASMR' of the project")
        return path

    def _single_rollout(self, algorithm, idx: int):
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
        mean_error = additional_information.get("mean_error")
        num_agents = additional_information.get(NUM_AGENTS)
        top01_error = additional_information.get("top0.1_error")
        top5_error = additional_information.get("top5_error")
        return num_agents, top01_error, top5_error, mean_error


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

    evaluator = Evaluator(num_pdes=num_pdes, seed=123)
    if visualize:
        evaluator.visualize(config_name=config_name, num_repetitions=num_reps, reset_buffer=reset_buffer)
    else:
        evaluator.evaluate(config_name=config_name, num_repetitions=num_reps, timeit=timeit, reset_buffer=reset_buffer)
