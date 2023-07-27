import torch
from torch.nn import functional as F

from src.algorithms.rl.architectures.swarm_dqn_policy import SwarmDQNPolicy
from src.algorithms.rl.architectures.dqn_modules.discrete_graph_q_net import DiscreteGraphQNet
from util.types import *
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import DQNPrioritizedBufferSamples
import numpy as np


class DiscreteSweepDQNNet(DiscreteGraphQNet):
    def forward(self, observations: InputBatch) -> torch.Tensor:
        """
        Predicts the actions for the given observations by
        * feeding them through the graph base to get latent *node/agent* features
        * using the value mlp to get an action distribution per *node*
        Args:
            observations: A batch of observation graphs

        Returns: A tensor of shape (num_total_agents, action_dimension)
            containing the action distribution for each agent
        """
        value_node_features = self.graph_base(observations)

        if self.dueling:
            value = self.value_mlp(value_node_features)
            advantage = self.advantage_mlp(value_node_features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_value_mlp(value_node_features)
        return q_values


class SweepDQNPolicy(SwarmDQNPolicy):

    def _initialize_q_network(self, dqn_config, environment, network_config):
        # q network, target network
        dueling = dqn_config.get("dueling")
        self.q_network = DiscreteSweepDQNNet(environment=environment,
                                             network_config=network_config,
                                             dueling=dueling,
                                             device=self._gpu_device)
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.trainable = False

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

            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_network.forward(observations)

        # Retrieve the q-values for the actions from the replay buffer.
        # This is the Q-value that we want to update, i.e., the one that was used for the action selection
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.reshape(-1, 1).long()).flatten()

        # Compute Huber loss (less sensitive to outliers than l2)
        if prioritized_replay:
            priorities = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            # sum over agents per batch --> shape (batch_size,)

            # total division by num_agents results in the total mean
            loss = (priorities * weights).sum() / current_q_values.shape[0]
            priorities = priorities.detach().cpu().numpy() + 1.0e-6
        else:
            priorities = None
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self._apply_loss(loss)
        return loss.item(), priorities
