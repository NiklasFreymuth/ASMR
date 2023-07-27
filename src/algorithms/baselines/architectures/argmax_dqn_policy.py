import torch
from torch.nn import functional as F
from torch_scatter import scatter_max

from src.algorithms.rl.architectures.swarm_dqn_policy import SwarmDQNPolicy
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import DQNPrioritizedBufferSamples
from util.types import *


class ArgmaxDQNPolicy(SwarmDQNPolicy):

    def sample_actions(self, observations: InputBatch, deterministic: bool = False) -> torch.Tensor:
        """
        For the argmax baseline, we only have a single agent whose action space is a choice over the number of elements,
        so we can simply sample the argmax of the q values over dimension 0 instead of 1

        Args:
            observations: The observations to sample actions for.
            deterministic: Whether to sample actions deterministically or not.

        Returns: The sampled actions as a torch tensor

        """
        observations = self.to_gpu(observations)

        q_values = self.predict(observations=observations)

        if deterministic:
            actions = torch.argmax(q_values, dim=0)
        else:
            if self.exploration_method == "epsilon_greedy":
                # perform epsilon greedy exploration with probability self.exploration_rate for each agent.
                # This is done by sampling a random action for each agent, and replacing the greedy action for this
                # agent with probability self.exploration_rate
                # Taking random actions per agent rather than per graph is more robust, because it allows the agent to
                # explore the graph more thoroughly
                actions = torch.argmax(q_values, dim=0)
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
                actions = torch.multinomial(F.softmax(q_values / self.exploration_rate, dim=0).squeeze(1),
                                            num_samples=1)
            else:
                raise ValueError("Unknown exploration method: {}".format(self.exploration_method))

        return actions

    def gradient_step(self, replay_data):
        prioritized_replay = isinstance(replay_data, DQNPrioritizedBufferSamples)
        observations, actions, rewards, next_observations, dones, weights = self._extract_replay_data(replay_data,
                                                                                                      prioritized_replay)
        with torch.no_grad():
            next_batch_indices = self._get_agent_batch_idx(next_observations)
            if self.double_q_learning:
                next_q_values = self.target_q_network.forward(next_observations)

                next_actions = self.q_network.forward(next_observations)
                _, next_actions = scatter_max(src=next_actions, index=next_batch_indices, dim=0)  # get element ids
                next_q_values = torch.gather(next_q_values, dim=0, index=next_actions.reshape(-1, 1).long()).flatten()
            else:
                # Compute the next Q-values using the target network
                next_q_values = self.target_q_network.forward(next_observations)
                # Follow greedy policy for the next step, i.e., use the action with the highest value hereafter
                next_q_values, _ = scatter_max(src=next_q_values, index=next_batch_indices, dim=0)  # get element ids
            # Avoid potential broadcast issue
            next_q_values = next_q_values.reshape(-1)
            # have #batch_size q_values, each of which correspond to the q-value of the marked element

            # calculate 1-step TD target
            target_q_values = rewards + torch.logical_not(dones) * self.discount_factor * next_q_values

        # Get current Q-values estimates
        current_q_values = self.q_network.forward(observations)

        # Retrieve the q-values for the actions from the replay buffer.
        # This is the Q-value that we want to update, i.e., the one that was used for the action selection
        current_q_values = torch.gather(current_q_values, dim=0, index=actions.reshape(-1, 1).long()).flatten()

        # Compute Huber loss (less sensitive to outliers than l2)
        if prioritized_replay:
            priorities = F.smooth_l1_loss(current_q_values, target_q_values, reduction="none")
            loss = (priorities * weights).mean()
            priorities = priorities.detach().cpu().numpy() + 1.0e-6
        else:
            priorities = None
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self._apply_loss(loss)

        return loss.item(), priorities
