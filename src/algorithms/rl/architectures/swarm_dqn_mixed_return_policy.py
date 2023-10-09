import torch
import numpy as np
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import DQNPrioritizedBufferSamples
from src.algorithms.rl.architectures.swarm_dqn_policy import SwarmDQNPolicy, project_q_values_to_previous_step
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.types import *


class SwarmDQNMixedRewardPolicy(SwarmDQNPolicy):
    """
    Policy that behaves like a DQN but uses a mixed_return learning approach to train the agent on a mixture of local
    and global rewards.
    """

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
        super().__init__(environment=environment,
                         algorithm_config=algorithm_config,
                         iterations=iterations,
                         use_gpu=use_gpu)

        # mixed_return learning parameters
        mixed_return_config = algorithm_config.get("mixed_return")
        self._global_weight: float = mixed_return_config.get("global_weight")

    def gradient_step(self, replay_data) -> Tuple[torch.Tensor, np.array]:
        prioritized_replay = isinstance(replay_data, DQNPrioritizedBufferSamples)
        observations, actions, rewards, next_observations, dones, weights = self._extract_replay_data(replay_data,
                                                                                                      prioritized_replay)

        batch_indices = self._get_agent_batch_idx(observations)
        global_weight = self.global_weight
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
            next_q_values = next_q_values.reshape(-1)

            # take average of q values and rewards over all agents in a graph
            next_batch_indices = self._get_agent_batch_idx(next_observations)
            global_next_q_values = scatter_mean(next_q_values, next_batch_indices, dim=0)
            global_next_q_values = global_next_q_values[next_batch_indices]
            next_q_values = global_weight * global_next_q_values + (1 - global_weight) * next_q_values

            global_rewards = scatter_mean(rewards, batch_indices, dim=0)
            global_rewards = global_rewards[batch_indices]
            rewards = global_weight * global_rewards + (1 - global_weight) * rewards

            # 1-step TD target

            # next_q_values is a tensor of shape (next_num_agents, 1). This needs to be mapped back to the original
            # number of agents in the batch. This is done by using the agent mappings
            agent_mappings = observations.agent_mapping
            # add size of all graphs up to this point as offset to make mappings unique
            agent_mapping_batch = observations.agent_mapping_batch
            agent_mappings = agent_mappings + self._get_agent_batch_ptr(observations)[agent_mapping_batch]

            projected_q_values = project_q_values_to_previous_step(next_q_values=next_q_values,
                                                                   agent_mappings=agent_mappings,
                                                                   aggregation_method=self._project_to_previous_step_aggregation
                                                                   )

            target_q_values = rewards + (1 - dones) * self.discount_factor * projected_q_values

        # Get current Q-values estimates
        current_q_values = self.q_network.forward(observations)

        # Retrieve the q-values for the actions from the replay buffer.
        # This is the Q-value that we want to update, i.e., the one that was used for the action selection
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.reshape(-1, 1).long()).flatten()

        global_current_q_values = scatter_mean(current_q_values, batch_indices, dim=0)
        global_current_q_values = global_current_q_values[batch_indices]
        current_q_values = global_weight * global_current_q_values + (1 - global_weight) * current_q_values

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

    @property
    def global_weight(self) -> np.array:
        """
        Returns the global weight for the current iteration. This is used to weight the global q-values in the
        aggregation.
        Returns: A float in [0, 1] that is used to weight the global q-values in the aggregation.

        """
        return self._global_weight
