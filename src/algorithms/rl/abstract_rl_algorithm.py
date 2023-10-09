import abc

import numpy as np
import plotly.graph_objects as go
import torch
from stable_baselines3.common.utils import safe_mean
from modules.swarm_environments import get_environments, AbstractSwarmEnvironment

import util.keys as Keys
from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from src.algorithms.rl.evaluation.rl_algorithm_evaluator import RLAlgorithmEvaluator
from src.modules.abstract_architecture import AbstractArchitecture
from util import keys
from util.function import prefix_keys, add_to_dictionary, get_from_nested_dict
from util.save_and_load.swarm_rl_checkpoint import SwarmRLCheckpoint
from util.torch_util.torch_util import detach
from util.types import *


class AbstractRLAlgorithm(AbstractIterativeAlgorithm, abc.ABC):
    def __init__(self, config: ConfigDict, seed: Optional[int] = None):
        """
        Initializes a framework for a Reinforcement Learning algorithm. This includes a train and an evaluation
        environment, as well as utility for recording the training progress over time.
        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        """
        super().__init__(config=config)
        self._algorithm_config = config.get("algorithm")
        self._verbose: bool = self.algorithm_config.get("verbose", False)
        self._network_config: ConfigDict = self.algorithm_config.get("network")
        self._batch_size: int = self.algorithm_config.get("batch_size")
        self._discount_factor: float = self.algorithm_config.get("discount_factor")
        self._ignore_truncated_dones = self.algorithm_config.get("ignore_truncated_dones")

        # Specifies the number of evaluation episodes used to reduce noise in the metrics (default is 1)
        self._num_evaluation_episodes = self.algorithm_config.get("num_evaluation_episodes", 1)

        # whether to ignore done flags that result from a timeout of the environment rather than some failure state.
        self._environment, self._evaluation_environments, self._final_environments = get_environments(
            environment_config=config.get("environment"),
            seed=seed)

        # whether to record videos
        self._record_videos: bool = get_from_nested_dict(config, ['recording', 'record_videos'], default_return=False)

        if self.algorithm_config.get("use_gpu"):
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def fit_and_evaluate(self) -> ValueDict:
        """
        Trains the algorithm for a single iteration, evaluates its performance and subsequently organizes and provides
        metrics, losses, plots etc. of the fit and evaluation
        Returns:

        """
        train_values = self.fit_iteration()
        evaluation_values = self.evaluate()

        # collect and organize values
        full_values = train_values | evaluation_values
        scalars = {}
        network_history = {}

        # filter scalars from dict
        value_dict = {}
        for key, value in full_values.items():
            if key in [Keys.FIGURES, Keys.VIDEO_ARRAYS]:  # may either be a single figure, or a list of figures
                value_dict[key] = value
            elif isinstance(value, list):
                if len(value) > 0:  # list is not empty
                    scalars[key] = value[-1]
                    network_history[key] = value  # must in this case be a part of the network history
            else:
                scalars[key] = value

        value_dict[Keys.SCALARS] = scalars
        value_dict[Keys.NETWORK_HISTORY] = network_history
        return value_dict

    def evaluate(self) -> ValueDict:
        """
        Perform a full rollout using the mean of the current policy on each evaluation environment.
        Track the reward_info of the environment, and plot the environment at its final state.
        Args:
        Returns:

        """
        self.set_training_mode(False)

        evaluation_dict = {Keys.FIGURES: [], Keys.VIDEO_ARRAYS: []}
        for environment_position, environment in enumerate(self._evaluation_environments):
            environment_prefix = f"env{environment_position}"

            current_evaluations = self._evaluate_environment(environment=environment)

            # use prefixes to keep identities of different evaluation environments
            current_environment_dict = current_evaluations.get("environment_dict")
            current_environment_dict = prefix_keys(dictionary=current_environment_dict, prefix=environment_prefix)
            evaluation_dict = evaluation_dict | current_environment_dict
            evaluation_dict[Keys.FIGURES].append(current_evaluations.get(Keys.FIGURES))
            if self._record_videos:
                evaluation_dict[Keys.VIDEO_ARRAYS].append(current_evaluations.get(Keys.VIDEO_ARRAYS))

        return evaluation_dict

    def _evaluate_environment(self, environment: AbstractSwarmEnvironment) -> ValueDict:
        """
        Evaluate a single environment for multiple episodes, and compute & record the average of the metrics for
        each episode. Render the result of the first evaluation episode.
        Args:
            environment: AbstractSwarmEnvironment
        Returns: A dictionary containing metrics

        """

        mean_infos_of_environments, last_infos_of_environments = [], []
        render_traces, rgb_video_array, additional_render_information = [], [], {}

        first_episode = True

        # Loop over multiple evaluation episodes to reduce the noise in the recorded metrics
        for _ in range(self._num_evaluation_episodes):
            # reset environment and prepare loop over rollout
            observation = environment.reset()
            done = False
            reward_info = {}

            # Render the environment & record video only in the first episode
            if first_episode:
                render_traces, additional_render_information = environment.render(mode="human")
                if self._record_videos:
                    rgb_video_array = [environment.render(mode="rgb_array")]
                else:
                    rgb_video_array = None

            # loop over rollout
            previous_values = None

            # compute a delta for the value to measure how the critic "progresses" throughout the episode
            while not done:
                actions, values = self.policy_step(observation=observation)
                actions = detach(actions)
                values = detach(values)
                if previous_values is None:
                    previous_values = values

                observation, reward, done, additional_information = environment.step(action=actions)

                # Render the environment & record video only in the first episode
                if first_episode:
                    current_traces, current_additional_render_information = environment.render(mode="human")
                    render_traces.extend(current_traces)
                    additional_render_information |= current_additional_render_information
                    if self._record_videos:
                        rgb_video_array.append(environment.render(mode="rgb_array"))

                reward_info = add_to_dictionary(reward_info, new_scalars={key: np.sum(value)
                                                                          for key, value
                                                                          in additional_information.items()})
                reward_info = add_to_dictionary(reward_info, {"critic_values": np.mean(values)})
                reward_info = add_to_dictionary(reward_info, {"delta_critic_values": np.mean(values) -
                                                                                     np.mean(previous_values)})
                previous_values = values

            first_episode = False

            # Add one metrics dictionary per evaluation episode to a list of dictionaries (for mean & last infos)
            mean_infos_of_environments.append(
                prefix_keys(dictionary={key: safe_mean(value) for key, value in reward_info.items()}, prefix="mean"))
            last_infos_of_environments.append(
                prefix_keys(dictionary={key: float(value[-1]) for key, value in reward_info.items()}, prefix="last"))

        def get_mean_dict(dict_list):
            # Nested method to compute the average of dictionaries for each key and given a list of dictionaries.
            mean_dict = {}
            for key in dict_list[0].keys():
                mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
            return mean_dict

        # Average the metrics of all evaluation episodes
        environment_dict = get_mean_dict(mean_infos_of_environments) | get_mean_dict(last_infos_of_environments)

        # render figure from previously provided traces and additional information such as the layout
        figure = go.Figure(data=render_traces, **additional_render_information)
        evaluation_dict = {"environment_dict": environment_dict, Keys.FIGURES: figure}

        if self._record_videos:
            rgb_video_array = np.array(rgb_video_array).transpose(0, 3, 1, 2)
            # transpose to have (time x channel x height x width)
            evaluation_dict[Keys.VIDEO_ARRAYS] = rgb_video_array

        return evaluation_dict

    def policy_step(self, *, observation: InputBatch, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single deterministic step of the current policy and return the action(s) taken by it. Does not
        compute gradients
        Args:
            observation: A (batch of) graph-based observation(s).
            **kwargs:

        Returns: A tuple (actions, values) of the actions taken by the agents of the policy, and the (q)-value(s) of
          these agents, either individually or for the full graph

        """
        raise NotImplementedError("AbstractRLAlgorithm does not implement policy_step()")

    def set_training_mode(self, mode: bool):
        self.policy.train(mode)

    @property
    def algorithm_config(self) -> ConfigDict:
        return self._algorithm_config

    @property
    def environment(self) -> AbstractSwarmEnvironment:
        return self._environment

    @property
    def evaluation_environments(self) -> List[AbstractSwarmEnvironment]:
        """
        Environment(s) used to evaluate the current progress of the algorithm.
        Returns: A dictionary of evaluation environments, where the key is a unique identifier of the environment.
        May only include a single environment, in which case the key will be the empty string ("")

        """
        return self._evaluation_environments

    @property
    def policy(self) -> AbstractArchitecture:
        raise NotImplementedError("AbstractRLAlgorithm does not implement network property")

    @property
    def environment_normalizer(self) -> Optional[AbstractEnvironmentNormalizer]:
        """
        Wrapper for the environment normalizer. May be None if no normalization is used.
        Returns:

        """
        return None

    #################
    # save and load #
    #################
    def save_checkpoint(self, directory: str, iteration: Optional[int] = None,
                        is_final_save: bool = False, is_initial_save: bool = False) -> None:
        """
        Save the current state of the algorithm to the given directory. This includes the policy, the optimizer, and
        the environment normalizer (if applicable).
        Args:
            directory:
            iteration:
            is_final_save:
            is_initial_save:

        Returns:

        """
        from pathlib import Path
        checkpoint_path = Path(directory)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        policy_save_path = self.policy.save(destination_folder=checkpoint_path,
                                            file_index=Keys.FINAL if is_final_save else iteration,
                                            save_kwargs=is_initial_save)
        if self.environment_normalizer is not None:
            import util.save_and_load.save_and_load_keys as K
            # load environment normalizer if available. Load from same iteration as network
            normalizer_save_path = policy_save_path.with_name(
                policy_save_path.stem + f"_{K.NORMALIZATION_PARAMETER_SUFFIX}.pkl")
            self.environment_normalizer.save(destination_path=normalizer_save_path)

    def load_from_checkpoint(self, checkpoint_config: ConfigDict) -> SwarmRLCheckpoint:
        """
        Loads the algorithm state from the given checkpoint path/experiment configuration name.
        May be used at the start of the algorithm to resume training.
        Args:
            checkpoint_config: Dictionary containing the configuration of the checkpoint to load. Includes
                checkpoint_path: Path to a checkpoint folder of a previous execution of the same algorithm
                iteration: (iOptional[int]) The iteration to load. If not provided, will load the last available iter
                repetition: (int) The algorithm repetition/seed to load. If not provided, will load the first repetition

        Returns:

        """
        import os
        import pathlib
        import util.save_and_load.save_and_load_keys as K

        # get checkpoint path and iteration
        experiment_name = checkpoint_config.get("experiment_name")
        iteration = checkpoint_config.get("iteration")
        repetition = checkpoint_config.get("repetition")
        if repetition is None:
            repetition = 0  # default to first repetition

        # format checkpoint path
        if "__" in experiment_name:
            # grid experiments, add the main experiment as the first part of the path
            experiment_name = os.path.join(experiment_name[0:experiment_name.find("__")], experiment_name)

        if checkpoint_config.get("load_root_dir") is None:
            load_root_dir = K.REPORT_FOLDER
        else:
            load_root_dir = checkpoint_config["load_root_dir"]

        checkpoint_path = os.path.join(load_root_dir,
                                       experiment_name,
                                       "log",
                                       f"rep_{repetition:02d}",
                                       K.SAVE_DIRECTORY)
        checkpoint_path = pathlib.Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"

        # load state dict for network
        if iteration is not None:  # if iteration is given, load the corresponding file
            file_name = f"{K.TORCH_SAVE_FILE}{iteration:04d}.pt"
        else:
            file_name = f"{K.TORCH_SAVE_FILE}_{Keys.FINAL}.pt"
            if not (checkpoint_path / file_name).exists():
                # if final.pkl does not exist, load the last iteration instead
                file_name = sorted(list(checkpoint_path.glob("*.pt")))[-1].name
        state_dict_file = checkpoint_path / file_name
        architecture = AbstractArchitecture.load_from_path(state_dict_path=state_dict_file,
                                                           environment=self.environment)
        # load environment normalizer if available. Load from same iteration as network
        normalizer_path = state_dict_file.with_name(state_dict_file.stem + f"_{K.NORMALIZATION_PARAMETER_SUFFIX}.pkl")
        if normalizer_path.exists():
            normalizer = AbstractEnvironmentNormalizer.load(checkpoint_path=normalizer_path)
        else:
            normalizer = None

        return SwarmRLCheckpoint(architecture=architecture, normalizer=normalizer)

    ####################
    # additional plots #
    ####################

    def additional_plots(self, iteration: int) -> Dict[Key, go.Figure]:
        """
        May provide arbitrary functions here that are used to draw additional plots.
        Args:
            iteration: Algorithm iteration that this function was called at
        Returns: A dictionary of {plot_name: plot}, where plot_function is any function that takes
          this algorithm at a current point as an argument, and returns a plotly figure.

        """
        self.set_training_mode(False)
        additional_plots = {}
        for environment_position, environment in enumerate(self._evaluation_environments):
            # since we do not reset the evaluation environments between the evaluation and the additional plots,
            # the environments are already in the correct state
            evaluation_environment_plots = environment.additional_plots(iteration=iteration,
                                                                        policy_step_function=self.policy_step)
            additional_plots = additional_plots | prefix_keys(evaluation_environment_plots,
                                                              prefix=f"env{environment_position}",
                                                              separator="/")
            # the "/" separator is interpreted as a subdirectory by the wandb logger, leading to a cleaner separation
            # of plots
        return additional_plots

    def get_final_values(self) -> ValueDict:
        """
        Returns a dictionary of values that should be logged at the end of the training.
        Returns:

        """
        self.set_training_mode(False)
        evaluator = RLAlgorithmEvaluator(config=self.config,
                                         policy_step_function=self.policy_step,
                                         environments=self._final_environments)
        final_values = {keys.TABLES: evaluator.get_tables()}
        return final_values
