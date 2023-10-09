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
from tqdm import tqdm

# path hacking for scripts from top level
current_directory = os.getcwd()
sys.path.append(current_directory)

from src.algorithms.baselines.sweep_dqn import SweepDQN
from src.algorithms.baselines.sweep_ppo import SweepPPO
from src.algorithms.get_algorithm import get_algorithm
from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm
from modules.swarm_environments import MeshRefinement
from modules.swarm_environments.mesh.mesh_refinement import EvaluationFEMProblemCircularQueue
from util.function import get_from_nested_dict
from util.torch_util.torch_util import detach
from util.types import *

from asmr_evaluations.evaluation_util import get_config_path, get_config


def single_rollout(algorithm, idx: int, verbose: bool = True):
    """
    Do a single rollout and return the remaining error and number of agents.
    Returns:

    """
    if verbose:
        print(f"    Rollout for idx {idx}")
    environment: MeshRefinement = algorithm.environment

    # reset without calculating the fine-grained reference
    start = time.time()

    environment.fem_problem = environment.fem_problem_queue.next()
    environment.fem_problem.calculate_solution()
    environment._reset_internal_state()
    observation = environment.last_observation

    done = False
    while not done:
        actions, values = algorithm.policy_step(observation=observation)
        actions = detach(actions)
        observation, _, done, _ = environment.inference_step(action=actions)
    end = time.time()
    total_time = end - start
    num_agents = environment.num_agents
    if verbose:
        print(f"    Total time: {total_time}. Num Agents: {num_agents}")
    return total_time, num_agents


def calculate_reference_solutions(algorithm, num_timesteps: int, num_pdes: int):
    print(f"Calculating reference solution times")
    environment = algorithm.environment

    reference_solutions = np.zeros((num_pdes, 2))
    # 2 for time in milliseconds and num_agents

    for n in tqdm(range(num_pdes)):
        fem_problem = environment.fem_problem_queue.next().fem_problem
        start = time.time()
        initial_mesh = fem_problem._domain._get_initial_mesh()
        integration_mesh = initial_mesh.refined(num_timesteps)
        integration_basis = fem_problem.add_boundary_conditions_and_create_basis(integration_mesh)
        solution = fem_problem.calculate_solution(integration_basis)
        end = time.time()
        total_time = end - start
        num_elements = integration_mesh.t.shape[1]
        reference_solutions[n, 0] = total_time
        reference_solutions[n, 1] = num_elements
    print(f"Finished calculating reference solution times")
    print(f"Average time: {np.mean(reference_solutions[:, 0])}. "
          f"Average num agents: {np.mean(reference_solutions[:, 1])}")

    return reference_solutions


class Evaluator:

    def __init__(self, num_pdes: int, seed: int = 123):
        self.num_pdes = num_pdes
        self.seed = seed  # set distinct seed.
        # Since this is only evaluation, the seed does not really matter except for the FEMProblemCircularQueue.
        # It should be fixed though, so that the same problems are used for all configs and algorithms.

        self._custom_eval_buffer = None

    def custom_eval_buffer(self, fem_config: ConfigDict):
        if self._custom_eval_buffer is None:
            self._custom_eval_buffer = EvaluationFEMProblemCircularQueue(fem_config=fem_config,
                                                                         random_state=np.random.RandomState(
                                                                             seed=self.seed)
                                                                         )
        return self._custom_eval_buffer

    def evaluate(self, config_name: str, num_repetitions: int = 10, num_timesteps: int = 6,
                 reference_only: bool = False, main_only: bool = False) -> ValueDict:
        config_path = get_config_path(root="reports", config_name=config_name)

        data = {}

        experiments = sorted(os.listdir(config_path))
        experiments = [experiment for experiment in experiments if "asmr" in experiment]
        # experiments = [experiment for experiment in experiments if not ".yml" in experiment]
        # testing the speed of our method

        pde_type = None

        for experiment_position, experiment_folder in enumerate(experiments):

            experiment_path = os.path.join(config_path, experiment_folder, "log")  # log folder contains logs

            print(f"Starting experiment {experiment_position + 1} of {len(experiments)}")
            print(f"Experiment: {experiment_folder}")

            experiment_results = np.empty((num_repetitions, self.num_pdes, 2))
            # shape: (num_repetitions, num_pdes, metrics), where metrics = (time, num_agents)

            for repetition_folder in sorted(os.listdir(experiment_path))[:num_repetitions]:
                repetition_folder: str
                repetition_idx = int(repetition_folder.split("_")[-1])

                _final_checkpoint_exists, algorithm = self._init_algorithm(experiment_folder, experiment_path,
                                                                           repetition_folder,
                                                                           repetition_idx, main_only=main_only)

                if pde_type is None:
                    pde_type = algorithm.config.get("environment").get("mesh_refinement").get("fem").get("pde_type")

                if repetition_idx == 0 and experiment_position == 0 and not main_only:
                    reference_solution_times = calculate_reference_solutions(algorithm=algorithm,
                                                                             num_timesteps=num_timesteps,
                                                                             num_pdes=self.num_pdes)

                    if reference_only:
                        os.makedirs(f"evaluation_results/iclr2023/{config_name}", exist_ok=True)
                        np.savez_compressed(
                            f"evaluation_results/iclr2023/{config_name}/{config_name}_uniform_times.npz",
                            **{f"{config_name}_reference_times": reference_solution_times})
                        print("Saved data! 🎉")
                        import sys
                        sys.exit()

                # loop over self.num_pdes rollouts and take the average
                print(f"  Calculating evaluation metrics for repetition: {repetition_folder}")
                results = np.array([single_rollout(algorithm=algorithm, idx=idx, verbose=False)
                                    for idx in range(self.num_pdes)])
                # shape: (num_pdes, metrics), where metrics = (error, num_agents)
                experiment_results[repetition_idx] = results

                avg_time = np.mean(results[:, 0])
                avg_num_agents = np.mean(results[:, 1])
                print(f"  Average time: {avg_time}, average number of agents: {avg_num_agents}")

            data[experiment_folder] = experiment_results

        # save the results
        # data is a dict with experiment names as keys and the results as values
        # where each result is a numpy array of shape (num_repetitions, num_pdes, metrics)

        os.makedirs(f"evaluation_results/iclr2023/runtime", exist_ok=True)
        np.savez_compressed(f"evaluation_results/iclr2023/runtime/{pde_type}_asmr.npz",
                            **data)

        if not main_only:
            np.savez_compressed(
                f"evaluation_results/iclr2023/runtime/{pde_type}_reference.npz",
                **{f"{config_name}_reference_times": reference_solution_times})
        print("Saved data! 🎉")

    def _init_algorithm(self, experiment_folder, experiment_path, repetition_folder, repetition_idx, main_only=False):
        print(f"  Building config for repetition: {repetition_folder}")
        current_config = get_config(experiment_path=experiment_path,
                                    experiment_folder=experiment_folder,
                                    iteration=None,
                                    repetition_folder=repetition_folder,
                                    repetition_idx=repetition_idx,
                                    num_pdes=self.num_pdes)
        if main_only:
            current_config["environment"]["mesh_refinement"]["fem"]["domain"]["num_integration_refinements"] = 2
            # do not need to evaluate this, so just make it fast
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
    parser.add_argument('-ref', "--reference_only", action="store_true", help="Only evaluate reference, i.e., the "
                                                                              "reference/integration mesh")
    parser.add_argument('-main', "--main_only", action="store_true", help="Only evaluate main results, i.e., policies")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()
    num_pdes = int(args.num_pdes)
    num_reps = int(args.num_reps)
    config_name = args.config_name
    reference_only = args.reference_only
    main_only = args.main_only
    assert not (reference_only and main_only), "Only one of reference_only and main_only can be true"

    evaluator = Evaluator(num_pdes=num_pdes, seed=123)
    evaluator.evaluate(config_name=config_name, num_repetitions=num_reps,
                       reference_only=reference_only, main_only=main_only)
