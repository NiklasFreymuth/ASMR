import torch
from torch.nn import functional as F
from torch_scatter import scatter_sum

from src.algorithms.rl.architectures.swarm_dqn_policy import SwarmDQNPolicy
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import DQNPrioritizedBufferSamples


class VDGNPolicy(SwarmDQNPolicy):

    def gradient_step(self, replay_data):
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

            # calculate batched TD error for each graph by summing over the q-values of all agents in the graph
            next_q_values = next_q_values.reshape(-1)
            next_batch_indices = self._get_agent_batch_idx(next_observations)
            next_q_values = scatter_sum(next_q_values, next_batch_indices, dim=0)
            target_q_values = rewards + torch.logical_not(dones) * self.discount_factor * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_network.forward(observations)

        # Retrieve the q-values for the actions from the replay buffer.
        # This is the Q-value that we want to update, i.e., the one that was used for the action selection
        current_q_values = torch.gather(current_q_values, dim=1, index=actions.reshape(-1, 1).long()).flatten()

        # this scatter_sum is the value decomposition at the heart of the VDGN, i.e., Q(s,a) = sum_i Q_i(s,a)
        batch_indices = self._get_agent_batch_idx(observations)
        current_q_values = scatter_sum(current_q_values, batch_indices, dim=0)

        # Compute Huber loss (less sensitive to outliers than l2)
        if prioritized_replay:
            priorities = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            loss = (priorities * weights).mean()
            priorities = priorities.detach().cpu().numpy() + 1.0e-6
        else:
            priorities = None
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self._apply_loss(loss=loss)

        return loss.item(), priorities
