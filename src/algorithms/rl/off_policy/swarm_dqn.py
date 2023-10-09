import numpy as np
import torch

from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm
from src.algorithms.rl.architectures.swarm_dqn_policy import SwarmDQNPolicy
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from src.algorithms.rl.normalizers.dummy_swarm_environment_normalizer import DummySwarmEnvironmentNormalizer
from src.algorithms.rl.normalizers.swarm_environment_observation_normalizer import SwarmEnvironmentObservationNormalizer
from src.algorithms.rl.off_policy.buffers.swarm_dqn_buffer import SwarmDQNBuffer
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import SwarmDQNPrioritizedBuffer
from modules.hmpn.common.hmpn_util import make_batch
from util.function import prefix_keys, add_to_dictionary, safe_mean, safe_max, safe_min
from util.progress_bar import ProgressBar
from util.types import *
from util.torch_util.torch_util import detach


class SwarmDQN(AbstractRLAlgorithm):
    """
    Graph-Based Deep Q-Networks implementation compatible with our GraphEnvironments.
    """

    def __init__(self, config: ConfigDict, seed: Optional[int] = None) -> None:
        super().__init__(config=config, seed=seed)

        # DQN specific config parts
        dqn_config: ConfigDict = self.algorithm_config.get("dqn")
        self._steps_per_iteration: int = dqn_config.get("steps_per_iteration")
        self._initial_replay_buffer_samples: int = dqn_config.get("initial_replay_buffer_samples")
        self._initial_sampling_strategy: str = dqn_config.get("initial_sampling_strategy")
        self._num_gradient_steps: int = dqn_config.get("num_gradient_steps")
        self.prioritized_buffer = dqn_config.get("use_prioritized_buffer")
        self._replay_buffer = self._build_buffer(dqn_config=dqn_config)

        self._iterations = config.get("iterations")
        if self.algorithm_config.get("checkpoint", {}).get("experiment_name") is not None:
            # optionally loading the network from a checkpoint. Does not currently load the replay buffer.
            from util.save_and_load.swarm_rl_checkpoint import SwarmRLCheckpoint
            checkpoint_config = self.algorithm_config.get("checkpoint")
            checkpoint: SwarmRLCheckpoint = self.load_from_checkpoint(checkpoint_config=checkpoint_config)
            assert isinstance(checkpoint.architecture, SwarmDQNPolicy), \
                f"checkpoint must contain a GraphDQNPolicy, given type: '{type(checkpoint.architecture)}'"
            self._policy: SwarmDQNPolicy = checkpoint.architecture
            self._environment_normalizer: AbstractEnvironmentNormalizer = checkpoint.normalizer
        else:  # build the network from scratch
            self._policy: SwarmDQNPolicy = self._build_policy()
            self._environment_normalizer: AbstractEnvironmentNormalizer = self._build_normalizer(dqn_config)

        self._kickoff_environments()

    def _build_buffer(self, dqn_config: ConfigDict) -> Union[SwarmDQNBuffer, SwarmDQNPrioritizedBuffer]:
        sample_buffer_on_gpu = self.algorithm_config.get("sample_buffer_on_gpu")
        buffer_size = dqn_config.get("max_replay_buffer_size")
        buffer_device = self.device if sample_buffer_on_gpu else torch.device("cpu")
        if self.prioritized_buffer:
            buffer_config = dqn_config.get("prioritized_buffer")
            alpha = buffer_config.get("alpha")
            self._prioritized_buffer_beta_init = buffer_config.get("beta_init")
            self._prioritized_buffer_beta_final = buffer_config.get("beta_final")
            return SwarmDQNPrioritizedBuffer(buffer_size=buffer_size, device=buffer_device, alpha=alpha,
                                             scalar_reward_and_done=self.scalar_rewards_and_dones)
        else:
            return SwarmDQNBuffer(buffer_size=buffer_size,
                                  device=buffer_device,
                                  scalar_reward_and_done=self.scalar_rewards_and_dones
                                  )

    def _build_policy(self) -> SwarmDQNPolicy:
        return self.policy_class(environment=self._environment,
                                 algorithm_config=self.algorithm_config,
                                 iterations=self._iterations,
                                 use_gpu=self.algorithm_config.get("use_gpu"))

    def _build_normalizer(self, dqn_config: ConfigDict) -> AbstractEnvironmentNormalizer:
        # normalization
        normalize_observations = dqn_config["normalize_observations"]
        if normalize_observations:
            environment_normalizer = SwarmEnvironmentObservationNormalizer(
                graph_environment=self._environment,
                normalize_nodes=True,
                normalize_edges=True,
            )
        else:
            environment_normalizer = DummySwarmEnvironmentNormalizer()
        return environment_normalizer

    def _kickoff_environments(self):
        observation = self._environment.reset()  # initially reset once to "kick off" the environment
        _ = self._environment_normalizer.reset(observations=observation)  # add initial observation to normalizer

    def fit_iteration(self) -> ValueDict:
        """
        Performs self._steps_per_iteration alternating rollout and training steps.
        Sets current scalars
        Returns the scalars of the policy training and information about the environment interaction
        Returns:

        """
        self.set_training_mode(True)
        if self._replay_buffer.size < self._initial_replay_buffer_samples:
            self._collect_learning_start_samples()

        progress_bar = ProgressBar(num_iterations=self._steps_per_iteration,
                                   verbose=self._verbose,
                                   separate_scalar_display=False)

        scalars = {}
        for _ in range(self._steps_per_iteration):
            rollout_scalars = self.rollout_step()  # add one sample to the rollout buffer
            rollout_scalars = {key: np.sum(value) for key, value in rollout_scalars.items()}
            rollout_scalars = prefix_keys(rollout_scalars, prefix="rollout")

            training_step_scalars = self.training_step()  # perform self._gradient_steps gradient steps

            scalars = add_to_dictionary(dictionary=scalars, new_scalars=training_step_scalars)
            scalars = add_to_dictionary(dictionary=scalars, new_scalars=rollout_scalars)

            progress_bar(**training_step_scalars)

        # inc step for the policy in order to compute the correct exploration rate
        self.policy.inc_step_counter()

        scalars = {"mean_" + key: safe_mean(value) for key, value in scalars.items()} | \
                  {"max_" + key: safe_max(value) for key, value in scalars.items()} | \
                  {"min_" + key: safe_min(value) for key, value in scalars.items()}

        scalars["buffer_size"] = self._replay_buffer.size
        scalars["exploration_rate"] = self.policy.exploration_rate

        if self.policy.learning_rate_scheduler is not None:
            scalars["learning_rate"] = self.policy.learning_rate_scheduler.get_last_lr()[0]

        return scalars

    def rollout_step(self, action_sampling_strategy: str = "agent") -> ValueDict:
        """
        Perform a single rollout step of self._environment using the current actor. Add the result to the rollout
        buffer
        Args:
            action_sampling_strategy: How to sample steps from the environment. May either be
                "agent" for samples from the current agent, or
                "random" for random samples in the action space of the environment.

        Returns: A dictionary of values for this rollout step
        """
        transition = self._collect_rollout_transition(action_sampling_strategy)
        observation, actions, reward, next_observation, done, additional_information, previous_num_agents = transition

        done, reward = self._cast_reward_and_done(done, reward, previous_num_agents)

        # add to replay buffer
        agent_mapping = None if self.scalar_rewards_and_dones else self._environment.agent_mapping
        self._replay_buffer.put(observation=observation,
                                actions=actions,
                                reward=reward,
                                next_observation=next_observation,
                                done=done,
                                agent_mapping=agent_mapping)

        return additional_information

    def _collect_rollout_transition(self, action_sampling_strategy: str = "agent") -> \
            Tuple[InputBatch, torch.Tensor, Union[np.array, float], InputBatch, bool, dict, int]:
        """
        Collect a single transition from the environment and return it
        Args:
            action_sampling_strategy: How to sample steps from the environment. May either be
                "agent" for samples from the current agent, including this agent's exploration noise, or
                "random" for random samples in the action space of the environment.

        Returns: A tuple of the observation, action, reward, next_observation, done, additional_information and
            previous_num_agents

        """
        self.set_training_mode(False)

        if self._environment.is_terminal:
            observation = self._environment.reset()
            # update normalization statistics with current step
            self._environment_normalizer.update_observations(observation)
        else:
            observation = self._environment.last_observation  # or maybe auto-reset?

        if action_sampling_strategy == "random":
            actions = self._get_random_actions()
        elif action_sampling_strategy == "agent":
            normalized_observation = self._environment_normalizer.normalize_observations(observation)
            actions = self._policy.sample_actions(observations=make_batch(normalized_observation), deterministic=False)

        else:
            raise ValueError(f"Unknown action strategy '{action_sampling_strategy}'")

        previous_num_agents = self._environment.num_agents
        next_observation, rewards, done, additional_information = self._environment.step(action=detach(actions))
        self._environment_normalizer.update_observations(next_observation)
        # add unnormalized observations to buffer, as those are normalized w/ updated data during the training step
        return observation, actions, rewards, next_observation, done, additional_information, previous_num_agents

    def _cast_reward_and_done(self, done: Union[bool, np.array], reward: Union[float, np.array],
                              previous_num_agents: int) -> Tuple[np.array, np.array]:
        """
        Casts the reward and done to the correct shape, depending on self.scalar_rewards_and_dones
        Args:
            done:
            reward:
            previous_num_agents:

        Returns:

        """
        if self.scalar_rewards_and_dones:
            reward = reward.sum()  # aggregate potential agent-wise rewards
            done = float(done)
        else:
            # cast from global to agent-wise reward
            if isinstance(reward, float) or reward.ndim == 0:
                reward = np.repeat(reward, previous_num_agents)
            done = np.repeat(float(done), previous_num_agents)
        return done, reward

    def _get_random_actions(self):
        # environment action space must be discrete
        actions = torch.randint(low=0,
                                high=self._environment.action_dimension,
                                size=(self._environment.num_agents,)).to(self.device)
        return actions

    def _collect_learning_start_samples(self):
        progress_bar = ProgressBar(num_iterations=self._initial_replay_buffer_samples - self._replay_buffer.size,
                                   verbose=self._verbose,
                                   separate_scalar_display=False,
                                   display_name="Initial Samples")
        while self._replay_buffer.size < self._initial_replay_buffer_samples:  # get additional rollouts at the start
            self.rollout_step(action_sampling_strategy=self._initial_sampling_strategy)
            progress_bar()

    def training_step(self) -> ValueDict:
        # Switch to train mode (this affects batch norm / dropout)
        self.set_training_mode(True)

        losses = []
        agents_per_batch = []
        for _ in range(self._num_gradient_steps):
            # Sample replay buffer
            if self.prioritized_buffer:
                replay_data = self._replay_buffer.sample(self._batch_size, beta=self.prioritized_buffer_beta)
            else:
                replay_data = self._replay_buffer.sample(self._batch_size)

            # normalize observations with current statistics. This does not update the statistics!
            replay_data.observations = self._environment_normalizer.normalize_observations(replay_data.observations)
            replay_data.next_observations = self._environment_normalizer.normalize_observations(
                replay_data.next_observations)

            # Train the DQN Policy
            loss, priorities = self.policy.gradient_step(replay_data=replay_data)
            if self.prioritized_buffer:
                indices = replay_data.indices
                self._replay_buffer.update_priorities(indices, priorities)
            losses.append(loss)
            agents_per_batch.append(replay_data.actions.shape[0])

            self.policy.update_target_q_network()

        # Update learning rate according to schedule
        if self.policy.learning_rate_scheduler is not None:
            self.policy.learning_rate_scheduler.step()

        return {"loss": np.mean(losses),

                "agents_per_batch": np.mean(agents_per_batch)
                }

    def policy_step(self, *, observation: InputBatch, deterministic: bool = True, no_grad: bool = False,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single policy step, i.e. compute the q-value for the current observation per agent and return the
        action with the highest q-value as well as the q-value itself.
        Args:
            observation:
            deterministic:
            no_grad:
            **kwargs:

        Returns:

        """
        observation = self._environment_normalizer.normalize_observations(observation)
        observation = make_batch(observation)
        if no_grad:
            with torch.no_grad():
                q_values = self._policy.predict(observations=observation)
        else:
            q_values = self._policy.predict(observations=observation)

        values, actions = q_values.max(dim=1)
        return actions, values

    @property
    def environment_normalizer(self) -> Union[SwarmEnvironmentObservationNormalizer, DummySwarmEnvironmentNormalizer]:
        return self._environment_normalizer

    @property
    def policy(self) -> SwarmDQNPolicy:
        return self._policy

    @property
    def current_iteration(self) -> int:
        return self._policy.current_iteration

    @property
    def prioritized_buffer_beta(self) -> float:
        """
        Scheduler for beta rate
        Returns the current beta for the prioritized replay buffer
        Returns:

        """
        r = max((self._iterations - self.current_iteration) / self._iterations, 0)
        return self._prioritized_buffer_beta_init * r + self._prioritized_buffer_beta_final * (1 - r)

    ############################################
    # Interfaces for inheritance and baselines #
    ############################################

    @property
    def scalar_rewards_and_dones(self):
        """
        DQN uses rewards/dones per agent instead of per graph.
        Returns:

        """
        return False

    @property
    def policy_class(self):
        mixed_return_config = self.algorithm_config.get("mixed_return", {})
        use_mixed_return = mixed_return_config.get("global_weight", 0) > 0
        if use_mixed_return:
            from src.algorithms.rl.architectures.swarm_dqn_mixed_return_policy import SwarmDQNMixedRewardPolicy
            return SwarmDQNMixedRewardPolicy
        else:
            return SwarmDQNPolicy
