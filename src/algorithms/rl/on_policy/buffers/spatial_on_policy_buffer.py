import numpy as np
import torch
from torch_scatter import scatter_sum

from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import AbstractMultiAgentOnPolicyBuffer
from util import keys as k
from util.types import *


class SpatialOnPolicyBuffer(AbstractMultiAgentOnPolicyBuffer):
    """
    Multi Agent On-Policy Rollout Buffer acting on values/advantages/returns per *node*. Node that this assumes
    that the agents are persistent within individual environment rollouts, i.e., no agents are added or removed within
    a single rollout.
    """

    def __init__(self, buffer_size: int, gae_lambda: float, discount_factor: float,
                 device: Optional[torch.device] = None, **kwargs):
        super().__init__(buffer_size=buffer_size, gae_lambda=gae_lambda, discount_factor=discount_factor,
                         device=device, **kwargs)
        self._agent_mappings: List[torch.Tensor] = []  # list of agent indices corresponding to previous agent indices
        # per step. This is used to project the values/advantages/returns to the previous step
        self._reward_per_agents: List[torch.Tensor] = []  # list of rewards per agent per step

    def reset(self) -> None:
        super().reset()
        self._agent_mappings = []
        self.rewards = []

    def _reset_values(self) -> None:
        self.values: List[torch.Tensor] = []

    def add(self, *, observation: Data, actions: torch.Tensor, reward: np.array, done: float,
            value: torch.Tensor, log_probabilities: torch.Tensor, **kwargs) -> None:
        assert "additional_information" in kwargs, "additional_information must be provided"
        additional_information = kwargs["additional_information"]

        assert k.AGENT_MAPPING in additional_information, "agent_mapping must be provided"
        agent_mapping = additional_information[k.AGENT_MAPPING]
        self._agent_mappings.append(torch.tensor(agent_mapping).to(self.device))

        super().add(observation=observation, actions=actions, reward=reward, done=done,
                    value=value, log_probabilities=log_probabilities)

    def _add_values(self, current_value: torch.Tensor):
        self.values.append(current_value.to(self.device))

    def _add_reward(self, reward: np.array) -> None:
        self.rewards.append(torch.tensor(reward.astype(np.float32)).to(self.device))

    def _convert_scalar_entries(self):
        """
        Cast scalar entries (one scalar per timestep/graph) into a graph for easier handling.
        Depending on whether this is a graph- or node-based policy buffer, the values are either per graph or per node
        and are handled accordingly
        Returns:

        """
        self.dones = torch.tensor(self.dones, device=self.device)

    def _convert_rewards(self):
        """
        Convert rewards to torch tensors and project them to the previous step if necessary
        Returns:
        """
        pass  # per-agent rewards have irregular shape, so can not be converted

    def _get_values_advantages_returns(self,
                                       batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = []
        advantages = []
        returns = []
        for index in batch_indices:
            values.extend(self.values[index])
            advantages.extend(self.advantages[index])
            returns.extend(self.returns[index])
        # flatten over values, advantages and returns
        values = torch.tensor(values, device=self.device)
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        return values, advantages, returns

    def compute_returns_and_advantage(self, last_value: torch.Tensor) -> None:
        """
        Taken from Stable Baselines 3
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438) to compute the advantage as
        GAE: h_t^V = r_t + y * V(s_t+1) - V(s_t)
        with
        V(s_t+1) = {0 if s_t is terminal (where V(s) may be bootstrapped in the algorithm sampling for terminal dones)
                   {V(s_t+1) if s_t not terminal
        where T is the last step of the rollout.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        Args:
            last_value: state value estimation for the last step of shape (#agents, )

        Returns:

        """
        last_value = last_value.to(self.device)  # move to correct device

        self.advantages: List[torch.Tensor]
        self.returns: List[torch.Tensor]
        self.advantages, self.returns = self._get_agent_wise_advantages_and_returns(last_value=last_value)

    def _get_agent_wise_advantages_and_returns(self, last_value: torch.Tensor) -> \
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Calculates the advantages for each agent in each step (i.e. for each timestep in each graph)
        Args:
            last_value: the value of the last state

        Returns:

        """
        advantages: Union[List[None], List[torch.Tensor]] = [None] * self.buffer_size
        # projected difference in values between r+V(s_{t+1}) and V(s)
        deltas = []
        next_values = self.values[1:] + [last_value]
        projected_next_values = self._project_to_previous_step(next_values, self.dones)
        for step in range(self.buffer_size):
            if self.dones[step]:
                delta = self.rewards[step] - self.values[step]
            else:
                delta = self.rewards[step] + self.discount_factor * projected_next_values[step] - self.values[step]
            deltas.append(delta)
        # a vectorized version of this would be
        # deltas = self.rewards + self.discount_factor * next_values * not_dones - self.values
        # note that this is not applicable due to different shapes/numbers of agents over different rollouts
        # generalized advantage estimate
        last_gae = torch.zeros(self._agent_mappings[-1].shape, device=self.device)
        for step in reversed(range(self.buffer_size)):
            # accumulate for generalized advantage estimate
            # we reverse the indices here to make use of the dynamic programming structure of the gae

            if self.dones[step]:
                last_gae = deltas[step]
            else:
                # project to previous step
                projected_last_gae = scatter_sum(last_gae, index=self._agent_mappings[step],
                                                 dim=0)

                last_gae = deltas[step] + self.discount_factor * self.gae_lambda * projected_last_gae

            advantages[step] = last_gae

        # calculate returns as advantage plus value for each step
        returns = [advantage + value for advantage, value in zip(advantages, self.values)]
        return advantages, returns

    def _project_to_previous_step(self, values: List[torch.Tensor], dones: torch.Tensor) -> List[torch.Tensor]:
        """
        Projects values to previous agents if the number of agents mismatches

        Args:
            values: values for each step. List of [#steps, #agents(step)]
            dones: List of dones. 1 if the env is done at that step, 0 else

        Returns: Projected values. List of [#steps, #agents(step-1)], i.e., a projection to the previous step

        """
        projected_values = []
        for step in range(len(values)):
            if dones[step]:
                projected_value = values[step]
            else:
                projected_value = scatter_sum(values[step], index=self._agent_mappings[step],
                                              dim=0)
            projected_values.append(projected_value)
        return projected_values

    @property
    def explained_variance(self) -> np.ndarray:
        from stable_baselines3.common.utils import explained_variance as explained_variance_function
        from util.torch_util.torch_util import detach
        # calculate explained variance for the full buffer rather than individual batches
        values = []
        returns = []
        for index in range(self.buffer_size):
            values.extend(self.values[index])
            returns.extend(self.returns[index])
        values = torch.tensor(values)
        returns = torch.tensor(returns)
        return explained_variance_function(detach(values.flatten()),
                                           detach(returns.flatten()))
