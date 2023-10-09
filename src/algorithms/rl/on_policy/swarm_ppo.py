import numpy as np
import torch
from modules.hmpn.common.hmpn_util import make_batch
from torch_scatter import scatter_mean

from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm
from src.algorithms.rl.architectures.swarm_ppo_actor_critic import SwarmPPOActorCritic
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from src.algorithms.rl.normalizers.dummy_swarm_environment_normalizer import DummySwarmEnvironmentNormalizer
from src.algorithms.rl.normalizers.on_policy.swarm_environment_ppo_normalizer import SwarmEnvironmentPPONormalizer
from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import AbstractMultiAgentOnPolicyBuffer
from src.algorithms.rl.on_policy.buffers.mixed_return_on_policy_buffer import MixedRewardOnPolicyBuffer
from src.algorithms.rl.on_policy.buffers.get_on_policy_buffer import get_on_policy_buffer
from src.algorithms.rl.on_policy.util.ppo_util import get_entropy_loss, get_policy_loss, get_value_loss
from util import keys
from util.function import prefix_keys, add_to_dictionary, safe_mean
from util.progress_bar import ProgressBar
from util.torch_util.torch_util import detach
from util.types import *


class SwarmPPO(AbstractRLAlgorithm):
    """
    Graph-Based Soft Actor Critic implementation compatible with our GraphEnvironments.
    """

    def __init__(self, config: ConfigDict, seed: Optional[int] = None) -> None:
        super().__init__(config=config, seed=seed)

        # PPO specific config parts
        ppo_config: ConfigDict = self.algorithm_config.get("ppo")
        self._epochs_per_iteration: int = ppo_config.get("epochs_per_iteration")
        self._num_rollout_steps: int = ppo_config.get("num_rollout_steps")
        self._clip_range: float = ppo_config.get("clip_range", 0.2)
        self._max_grad_norm: float = ppo_config.get("max_grad_norm", 0.5)
        self._entropy_coefficient: float = ppo_config.get("entropy_coefficient", 0.0)
        self._value_function_coefficient: float = ppo_config.get("value_function_coefficient", 0.5)
        self._value_function_clip_range: float = ppo_config.get("value_function_clip_range", 0.2)
        self._value_function_scope: str = ppo_config.get("value_function_scope")
        assert self._value_function_scope in ["graph", "vdn", "agent", "spatial"], \
            (f"Value function must be any of 'graph', 'vdn', 'agent', 'spatial',"
             f" given '{self._value_function_scope}' instead")

        # optionally load the architecture (actor, critic, optimizers) and potentially the normalizer from a checkpoint
        if self.algorithm_config.get("checkpoint", {}).get("experiment_name") is not None:
            from util.save_and_load.swarm_rl_checkpoint import SwarmRLCheckpoint
            checkpoint_config = self.algorithm_config.get("checkpoint")
            checkpoint: SwarmRLCheckpoint = self.load_from_checkpoint(checkpoint_config=checkpoint_config)
            assert isinstance(checkpoint.architecture, SwarmPPOActorCritic), \
                f"checkpoint must contain a GraphDQNPolicy, given type: '{type(checkpoint.architecture)}'"
            self._policy: SwarmPPOActorCritic = checkpoint.architecture
            self._environment_normalizer: AbstractEnvironmentNormalizer = checkpoint.normalizer
        else:
            self._policy: SwarmPPOActorCritic = self._build_policy(ppo_config=ppo_config)
            self._environment_normalizer: AbstractEnvironmentNormalizer = self._build_normalizer(ppo_config)

        self.rollout_buffer = self._build_buffer(ppo_config)

        self._kickoff_environments()

    def _build_normalizer(self, ppo_config):
        normalize_rewards = ppo_config.get("normalize_rewards")
        normalize_observations = ppo_config.get("normalize_observations")
        if normalize_rewards or normalize_observations:
            environment_normalizer = SwarmEnvironmentPPONormalizer(graph_environment=self._environment,
                                                                   discount_factor=self._discount_factor,
                                                                   normalize_rewards=normalize_rewards,
                                                                   normalize_nodes=normalize_observations,
                                                                   normalize_edges=normalize_observations)
        else:
            environment_normalizer = DummySwarmEnvironmentNormalizer()
        return environment_normalizer

    def _kickoff_environments(self):
        observation = self._environment.reset()  # initially reset once to "kick off" the environment
        _ = self._environment_normalizer.reset(observations=observation)  # add initial observation to normalizer

    def _build_buffer(self, ppo_config):
        sample_buffer_on_gpu = self.config["algorithm"]["sample_buffer_on_gpu"]
        gae_lambda: float = ppo_config.get("gae_lambda", 0.95)
        buffer_device = self.device if sample_buffer_on_gpu else torch.device("cpu")
        rollout_buffer: AbstractMultiAgentOnPolicyBuffer = \
            get_on_policy_buffer(buffer_size=self._num_rollout_steps,
                                 gae_lambda=gae_lambda,
                                 discount_factor=self._discount_factor,
                                 value_function_scope=self._value_function_scope,
                                 device=buffer_device,
                                 mixed_return_config=self.algorithm_config.get("mixed_return", {}),
                                 )
        return rollout_buffer

    def _build_policy(self, ppo_config: ConfigDict) -> SwarmPPOActorCritic:
        return SwarmPPOActorCritic(environment=self._environment,
                                   network_config=self._network_config,
                                   use_gpu=self.algorithm_config.get("use_gpu"),
                                   ppo_config=ppo_config)

    def fit_iteration(self) -> ValueDict:
        """
        Performs a single iteration of the PPO algorithm.
        This iteration includes
        * sampling steps/rollouts from the environment
        * training/updating the policy based on the on-policy samples collected for this rollout
        * logging interesting metrics
        Returns: A dictionary containing all logged metrics

        """
        rollout_scalars = self.collect_rollouts()
        rollout_scalars = prefix_keys(rollout_scalars, prefix="rollout")

        training_scalars = self.training_step()
        training_scalars = prefix_keys(training_scalars, prefix="train")

        scalars = rollout_scalars | training_scalars

        return scalars

    def collect_rollouts(self) -> ValueDict:
        """
        Collects rollouts for the given number of steps and adds them to the rollout buffer.
        Since PPO is an on-policy algorithm, the rollouts are collected from the current policy and into a fresh buffer.
        Returns:

        """
        assert self._num_rollout_steps > 1, "Need to perform at least one step"
        self.set_training_mode(False)
        self.rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        reward_info = {}
        final_reward_info = {}

        # environment can never be terminal at the beginning of a rollout, as we reset at the end of the
        # previous rollout and during the __init__ of this class
        observation = self._environment.last_observation
        # only normalize but don't update the normalizer here, as we already updated it at the end of the previous step
        observation = self._environment_normalizer.normalize_observations(observations=observation)

        progress_bar = ProgressBar(num_iterations=self._num_rollout_steps, verbose=self._verbose,
                                   separate_scalar_display=False, display_name="Rollouts")

        for step in range(self._num_rollout_steps):
            additional_information, done, observation = self._policy_sample_step(current_observation=observation)

            # log the unprocessed information given by the environment and update the progress bar
            reward_info = add_to_dictionary(reward_info,
                                            new_scalars={key: np.sum(value)
                                                         for key, value in additional_information.items()})
            if done:
                final_reward_info = add_to_dictionary(final_reward_info,
                                                      new_scalars={key: np.sum(value)
                                                                   for key, value in additional_information.items()})
            progress_bar()

        last_value = self._get_last_value(last_observation=observation)
        self.rollout_buffer.compute_returns_and_advantage(last_value=last_value)

        rollout_scalars = {"mean_" + key: safe_mean(value) for key, value in reward_info.items()} | \
                          {"final_mean_" + key: safe_mean(value) for key, value in final_reward_info.items()}
        return rollout_scalars

    def _policy_sample_step(self, current_observation) -> Tuple[ValueDict, bool, InputBatch]:
        """
        Performs a single step of the policy and adds the resulting transition to the rollout buffer.
        Args:
            current_observation: The current observation of the environment

        Returns:

        """
        with torch.no_grad():
            actions, value, log_probabilities = self._policy(make_batch(current_observation), deterministic=False)
        value = value.flatten()
        next_observation, reward, done, additional_information = self._environment.step(action=detach(actions))
        if self._value_function_scope == "spatial":
            # the environment returns one reward per agent
            from modules.swarm_environments import MeshRefinement
            assert isinstance(self._environment, MeshRefinement)
            rollout_buffer_information = {keys.AGENT_MAPPING: self._environment.agent_mapping}
            if isinstance(self.rollout_buffer, MixedRewardOnPolicyBuffer):
                rollout_buffer_information[keys.GLOBAL_REWARD] = reward.mean()
                # we take the mean here, as this value is used for all agents
        else:  # graph-wise reward
            rollout_buffer_information = None
            reward = np.sum(reward)  # summing up the rewards for the different agents in case there are multiple
            # this sum should happen *before* the reward normalization, since the normalization happens per reward
        next_observation, reward = self._environment_normalizer.update_and_normalize(observations=next_observation,
                                                                                     reward=reward)
        if done and self._ignore_truncated_dones and additional_information.get(keys.IS_TRUNCATED, False):
            reward = self._bootstrap_terminal_reward(next_observation, reward)
        # done flag is only set due to environment timeout, but should be ignored
        self.rollout_buffer.add(observation=current_observation,
                                actions=actions,
                                reward=reward,
                                done=float(done),
                                value=value,
                                log_probabilities=log_probabilities,
                                additional_information=rollout_buffer_information)
        if done:
            # we reset immediately after sampling. This ensures that the environment is never in a terminal state
            current_observation = self._environment.reset()
            current_observation = self._environment_normalizer.reset(observations=current_observation)
        else:
            current_observation = next_observation
        return additional_information, done, current_observation

    def _get_last_value(self, last_observation) -> torch.Tensor:
        with torch.no_grad():
            # Compute value for the last timestep
            _, last_value, _ = self._policy(make_batch(last_observation))
            last_value = last_value.squeeze(dim=-1).flatten()
        return last_value

    def _bootstrap_terminal_reward(self, next_observation: InputBatch, reward: torch.Tensor) -> torch.Tensor:
        """
        Bootstraps the reward for the last step of the rollout, if the environment is done due to a terminal state.
        time limit dones, i.e., environment terminations that are due to a time limit rather than
        policy failure are bootstrapped for the advantage estimate. Rather than "stopping" the advantage
        at this step, the reward is bootstraped to include the value function estimate of the current
        observation as an estimate of how the episode *should/could* have continued.
        Args:
            next_observation: The observation after the terminal state
            reward: The reward for the terminal state

        Returns: The bootstrapped reward

        """
        with torch.no_grad():
            _, terminal_value, _ = self._policy(observations=make_batch(next_observation), deterministic=True)
        if self._value_function_scope == "agent":
            # aggregate over evaluations per node to get one evaluation per graph.
            # Here, we only have one graph, so we can simply take the mean
            terminal_value = terminal_value.mean(dim=0)
        reward = reward + self._discount_factor * terminal_value
        return reward

    def training_step(self) -> ValueDict:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.set_training_mode(True)

        train_scalars = {}
        total_loss = None

        progress_bar = ProgressBar(num_iterations=self._epochs_per_iteration, verbose=self._verbose,
                                   separate_scalar_display=False)

        # train for n_epochs epochs
        old_policy_parameters = torch.cat([param.view(-1) for param in self._policy.parameters()])
        for epoch in range(self._epochs_per_iteration):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self._batch_size):
                total_loss, train_step_scalars = self._train_batch(rollout_data=rollout_data)

                train_scalars = add_to_dictionary(train_scalars, new_scalars=train_step_scalars)

            progress_bar(total_loss=total_loss.item())

        # add more metrics
        if self.learning_rate_scheduler is not None:  # update and log learning rate
            self.learning_rate_scheduler.step()
            train_scalars["learning_rate"] = self.learning_rate_scheduler.get_last_lr()[0]

        new_policy_parameters = torch.cat([param.view(-1) for param in self._policy.parameters()])
        differences = torch.abs(old_policy_parameters - new_policy_parameters)
        train_scalars["mean_network_weight_difference"] = np.mean(detach(differences))
        train_scalars["max_network_weight_difference"] = np.max(detach(differences))

        # calculate explained variance for the full buffer rather than individual batches. Done in self.rollout_buffer
        train_scalars["value_explained_variance"] = self.rollout_buffer.explained_variance

        if isinstance(self.rollout_buffer, MixedRewardOnPolicyBuffer):
            train_scalars["global_weight"] = self.rollout_buffer.global_weight
        return train_scalars

    def _train_batch(self, rollout_data) -> Tuple[torch.Tensor, ValueDict]:
        """
        Perform a single update step on a single batch of data.
        Args:
            rollout_data: The data to train on

        Returns: The loss and a dictionary of scalars to log

        """
        # gather data and put on GPU device
        observations = rollout_data.observations.to(self.device)
        actions = rollout_data.actions.to(self.device)
        old_log_probabilities = rollout_data.old_log_probabilities.to(self.device)
        old_values = rollout_data.old_values.to(self.device)
        advantages = rollout_data.advantages.to(self.device)
        returns = rollout_data.returns.to(self.device)
        # evaluate policy and value function
        values, log_probabilities, entropy = self._policy.evaluate_actions(observations=observations,
                                                                           actions=actions)
        values = values.squeeze(dim=-1)  # flattened list of one value, either per node or per graph
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        if self._value_function_scope in ["graph", "vdn"] and not len(ratio) == self._batch_size:
            # if the value function acts on full graphs, we need to aggregate the log probabilities accordingly
            if hasattr(observations, "x"):  # homogeneous graph
                batch = observations.batch
            else:
                agent_node_index = observations.node_types.index(self._environment.agent_node_type)
                batch = observations.node_stores[agent_node_index].batch
            ratio = scatter_mean(ratio, batch, dim=0)
            # use scatter_mean here for both graph and vdn, as we want the probability ratios of the graph to be the
            # normalized sum of the ratios of its agents
        # calculate losses
        policy_loss = get_policy_loss(advantages=advantages, ratio=ratio,
                                      clip_range=self._clip_range)
        value_loss = self._value_function_coefficient * get_value_loss(returns=returns,
                                                                       values=values, old_values=old_values,
                                                                       clip_range=self._value_function_clip_range)
        entropy_loss = self._entropy_coefficient * get_entropy_loss(entropy, log_probabilities)
        total_loss = policy_loss + value_loss + entropy_loss
        # Optimization step
        self._apply_loss(total_loss)
        # Logging
        with torch.no_grad():
            log_ratio = log_probabilities - old_log_probabilities
            approx_kl_div = detach(torch.mean((torch.exp(log_ratio) - 1) - log_ratio))
        train_step_scalars = {
            "total_loss": total_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "value_function_loss": value_loss.item(),
            "mean_value_function": values.mean().item(),
            "policy_loss": policy_loss.item(),
            "policy_kl": approx_kl_div,
            "policy_clip_fraction": torch.mean((torch.abs(ratio - 1) > self._clip_range).float()).item()}
        if self._policy.log_std is not None:
            train_step_scalars["log_policy_std"] = self._policy.log_std.mean().item()
        return total_loss, train_step_scalars

    def _apply_loss(self, total_loss):
        self._policy.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._max_grad_norm)  # Clip grad norm
        self._policy.optimizer.step()

    def policy_step(self, *, observation: InputBatch, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single deterministic step of the current policy and return the action(s) taken by it. Does not
        compute gradients
        Args:
            observation: A (batch of) graph-based observation(s). Should be unnormalized, as they will be normalized
                by this function. Must not be batched
            **kwargs:

        Returns: A tuple (actions, values) of the actions taken by the agents of the policy, and the (q)-value(s) of
          these agents, either individually or for the full graph

        """
        observation = self._environment_normalizer.normalize_observations(observations=observation)
        observation = make_batch(observation)
        with torch.no_grad():
            actions, values, _ = self._policy(observations=observation, deterministic=True, **kwargs)
        return actions, values

    def set_training_mode(self, mode: bool):
        """
        Set the training mode of the policy. This is important for e.g., batch normalization layers.
        Args:
            mode: True if the policy should be in training mode, False otherwise.

        Returns:

        """
        self._policy.train(mode)

    @property
    def policy(self) -> SwarmPPOActorCritic:
        self._policy: SwarmPPOActorCritic
        return self._policy

    @property
    def learning_rate_scheduler(self) -> Optional:
        return self._policy.learning_rate_scheduler

    @property
    def environment_normalizer(self) -> AbstractEnvironmentNormalizer:
        return self._environment_normalizer
