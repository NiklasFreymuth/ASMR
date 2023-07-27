import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean
from stable_baselines3.common.utils import polyak_update

from src.algorithms.rl.off_policy.buffers.swarm_dqn_buffer import DQNBufferSamples
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import DQNPrioritizedBufferSamples
from src.modules.abstract_architecture import AbstractArchitecture
from src.algorithms.rl.architectures.dqn_modules.discrete_graph_q_net import DiscreteGraphQNet
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.types import *


def project_q_values_to_previous_step(next_q_values: torch.Tensor,
                                      agent_mappings: torch.Tensor,
                                      aggregation_method: str = "sum") -> torch.Tensor:
    """
    Projects next_q_values to previous agents if the number of agents mismatches

    Args:
        next_q_values: A tensor of shape (num_agents_at_next_step, action_dimension) containing the next q values
        agent_mappings: A tensor of shape (num_agents_at_next_step,) containing the mappings from the current to
            the previous number of agents
        aggregation_method: The method to use for aggregation of the multiple agents in the latest step. Either "sum" or "mean".

    Returns: Projected observation. List of [#steps, #agents(step-1)], i.e., a projection to the previous step

    """
    if aggregation_method == "sum":
        aggregation_function = scatter_sum
    elif aggregation_method == "mean":
        aggregation_function = scatter_mean
    else:
        raise ValueError("Unknown aggregation method: {}".format(aggregation_method))
    projected_q_values = aggregation_function(next_q_values, agent_mappings, dim=0)
    return projected_q_values


class SwarmDQNPolicy(AbstractArchitecture):

    def __init__(self,
                 environment: AbstractSwarmEnvironment,
                 algorithm_config: ConfigDict,
                 iterations: int,
                 use_gpu: bool
                 ) -> None:
        """
        Args:
            environment: The environment to train the reinforcement algorithm on.
            algorithm_config: Configuration for the algorithm.
            iterations: Number of iterations to train the algorithm.
            use_gpu:
        """
        super().__init__(use_gpu=use_gpu, algorithm_config=algorithm_config, iterations=iterations)

        network_config = algorithm_config.get("network")
        dqn_config: ConfigDict = algorithm_config.get("dqn")
        self._environment = environment
        self.discount_factor: float = algorithm_config.get("discount_factor")
        self._step: int = 0

        # exploration
        self.exploration_method = dqn_config.get("exploration_method")
        self._exploration_rate_init = dqn_config.get("exploration_rate_init")
        self._exploration_rate_final = dqn_config.get("exploration_rate_final")
        num_exploration_decay_steps = dqn_config.get("num_exploration_decay_steps")
        if num_exploration_decay_steps is None:
            num_exploration_decay_steps = iterations // 2
        num_exploration_decay_steps = max(1, num_exploration_decay_steps)  # to avoid division by zero down below
        self._num_exploration_decay_steps = num_exploration_decay_steps

        # dqn improvements and other modifications
        self.target_update_rate = dqn_config.get("target_update_rate")

        self.double_q_learning = dqn_config.get("double_q_learning")
        self._project_to_previous_step_aggregation = dqn_config.get("project_to_previous_step_aggregation")

        self._initialize_q_network(dqn_config, environment, network_config)

        self.max_grad_norm = dqn_config.get("max_grad_norm", 0.5)
        self._agent_node_type = environment.agent_node_type
        self._action_dimension = environment.action_dimension

        self._initialize_optimizer_and_scheduler(training_config=network_config.get("training"))
        self.to(self._gpu_device)

    def _initialize_q_network(self, dqn_config, environment, network_config):
        # q network, target network
        dueling = dqn_config.get("dueling")
        self.q_network = DiscreteGraphQNet(environment=environment,
                                           network_config=network_config,
                                           dueling=dueling,
                                           device=self._gpu_device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.trainable = False

    def inc_step_counter(self):
        self._step += 1

    def update_target_q_network(self) -> None:
        """
        Softly updates the parameters of the target network
        Args:

        Returns:

        """
        polyak_update(params=self.q_network.parameters(),
                      target_params=self.target_q_network.parameters(),
                      tau=self.target_update_rate)

    def sample_actions(self, observations: InputBatch, deterministic: bool = False) -> torch.Tensor:
        """
        Samples actions from the policy.
        Overrides the base_class predict function to include epsilon-greedy exploration.

        Args:
            observations: The observations to sample actions for.
            deterministic: Whether to sample actions deterministically or not.

        Returns: The sampled actions as a torch tensor

        """
        observations = self.to_gpu(observations)

        q_values = self.predict(observations=observations)

        if deterministic:
            actions = torch.argmax(q_values, dim=1)
        else:
            if self.exploration_method == "epsilon_greedy":
                # perform epsilon greedy exploration with probability self.exploration_rate for each agent.
                # This is done by sampling a random action for each agent, and replacing the greedy action for this
                # agent with probability self.exploration_rate
                # Taking random actions per agent rather than per graph is more robust, because it allows the agent to
                # explore the graph more thoroughly
                actions = torch.argmax(q_values, dim=1)
                num_agents = self._get_num_agents(observations)
                random_actions = torch.randint(low=0,
                                               high=self._action_dimension,
                                               size=(num_agents,)
                                               )
                random_actions = self.to_gpu(random_actions)

                random_action_map = torch.rand(num_agents) < self.exploration_rate
                actions[random_action_map] = random_actions[random_action_map]
            elif self.exploration_method == "boltzmann":
                # exploration_rate is the temperature
                pseudo_probabilities = F.softmax(q_values / self.exploration_rate, dim=1)
                actions = torch.multinomial(pseudo_probabilities, num_samples=1).squeeze(
                    1)
            else:
                raise ValueError("Unknown exploration method: {}".format(self.exploration_method))

        return actions

    def _get_num_agents(self, observations) -> int:
        """

        Args:
            observations:

        Returns:

        """
        return self._get_agent_attribute(observations, "x").shape[0]

    def _get_agent_batch_idx(self, observations) -> torch.Tensor:
        """
        Get the batch indices of the agents in the batch. This is needed to create correspondences between
        graphs and agents in each batch
        Args:
            observations:

        Returns: A tensor of shape (num_agents,) containing the batch indices of the agents

        """
        return self._get_agent_attribute(observations, "batch")

    def _get_agent_batch_ptr(self, observations) -> torch.Tensor:
        """
        Get the batch indices of the agents in the batch. This is needed to create correspondences between
        graphs and agents in each batch
        Args:
            observations:

        Returns: A tensor of shape (num_agents,) containing the batch indices of the agents

        """
        return self._get_agent_attribute(observations, "ptr")

    def _get_agent_attribute(self, observations, attribute: str) -> torch.Tensor:
        if hasattr(observations, attribute):
            observation_attr = getattr(observations, attribute)
        else:
            agent_node_position = observations.node_types.index(self._agent_node_type)
            observation_attr = observations.node_stores[agent_node_position][attribute]
        return observation_attr

    def predict(self, observations: InputBatch) -> torch.Tensor:
        """
        Predicts the actions for the given observations by
        * feeding them through the graph base to get latent *node/agent* features
        * using the value mlp to get an action distribution per *node*
        Args:
            observations:

        Returns: A tensor of shape (batch_size, action_dimension) containing the action distribution for each agent

        """
        return self.q_network.forward(observations=observations)

    def _extract_replay_data(self, replay_data: DQNBufferSamples, prioritized_replay: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = replay_data.observations.to(self._gpu_device)
        actions = replay_data.actions.to(self._gpu_device)
        rewards = replay_data.rewards.to(self._gpu_device)
        next_observations = replay_data.next_observations.to(self._gpu_device)
        dones = replay_data.dones.to(self._gpu_device)
        if prioritized_replay:
            assert isinstance(replay_data, DQNPrioritizedBufferSamples)
            weights = replay_data.weights.to(self._gpu_device)
        else:
            weights = None
        return observations, actions, rewards, next_observations, dones, weights

    def gradient_step(self, replay_data) -> Tuple[torch.Tensor, np.array]:
        prioritized_replay = isinstance(replay_data, DQNPrioritizedBufferSamples)
        observations, actions, rewards, next_observations, dones, weights = self._extract_replay_data(replay_data,
                                                                                                      prioritized_replay)
        with torch.no_grad():
            if self.double_q_learning:
                next_actions = torch.argmax(self.q_network.forward(next_observations), dim=1)
                next_q_values = self.target_q_network.forward(next_observations)
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions.reshape(-1, 1).long()).flatten()
            else:
                # Compute the next Q-values using the target network
                next_q_values = self.target_q_network.forward(next_observations)
                # Follow greedy policy for the next step, i.e., use the action with the highest value hereafter
                next_q_values, _ = next_q_values.max(dim=1)
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1)  # , 1)
            # 1-step TD target

            # next_q_values is a tensor of shape (next_num_agents, 1). This needs to be mapped back to the original
            # number of agents in the batch. This is done by using the agent mappings

            agent_mappings = observations.agent_mapping
            agent_mapping_batch = observations.agent_mapping_batch
            agent_mappings = agent_mappings + self._get_agent_batch_ptr(observations)[agent_mapping_batch]

            projected_q_values = project_q_values_to_previous_step(next_q_values=next_q_values,
                                                                   agent_mappings=agent_mappings,
                                                                   aggregation_method=self._project_to_previous_step_aggregation)

            target_q_values = rewards + (1 - dones) * self.discount_factor * projected_q_values

        # Get current Q-values estimates
        current_q_values = self.q_network.forward(observations)

        # Retrieve the q-values for the actions from the replay buffer.
        # This is the Q-value that we want to update, i.e., the one that was used for the action selection
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.reshape(-1, 1).long()).flatten()

        # Compute Huber loss (less sensitive to outliers than l2)
        if prioritized_replay:
            loss_per_agent = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            # sum over agents per batch --> shape (batch_size,)
            priorities = scatter_sum(loss_per_agent, self._get_agent_batch_idx(observations))

            # total division by num_agents results in the total mean
            loss = (priorities * weights).sum() / current_q_values.shape[0]
            priorities = priorities.detach().cpu().numpy() + 1.0e-6
        else:
            priorities = None
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self._apply_loss(loss)
        return loss.item(), priorities

    def _apply_loss(self, loss: torch.Tensor) -> None:
        # Optimize the policy
        self._optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self._optimizer.step()


    @property
    def optimizers(self) -> List[optim.Optimizer]:
        return [self._optimizer]

    @property
    def exploration_rate(self) -> float:
        """
        Scheduler for exploration rate. Returns a scalar in [0, 1] that is used to scale the exploration noise.
        If the exploration type is "epsilon_greedy", the exploration rate is used as the probability of taking a random
        action. If it is "boltzmann", the exploration rate is used as the temperature parameter for the Boltzmann
        distribution.
        Returns: The current exploration rate

        """
        rate = max((self._num_exploration_decay_steps - self._step) / self._num_exploration_decay_steps, 0)
        return self._exploration_rate_init * rate + self._exploration_rate_final * (1 - rate)

    @property
    def current_iteration(self) -> int:
        return self._step
