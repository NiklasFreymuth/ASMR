import numpy as np
import torch

from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import AbstractMultiAgentOnPolicyBuffer
from util.types import *


class GraphOnPolicyBuffer(AbstractMultiAgentOnPolicyBuffer):
    """
    Multi Agent On-Policy Rollout Buffer acting on a single value/advantage/return per *graph*.
    """

    def _reset_values(self) -> None:
        self.values = torch.empty(self.buffer_size, device=self.device)

    def _add_values(self, current_value: torch.Tensor) -> None:
        self.values[self.current_position] = current_value.to(self.device)

    def _get_values_advantages_returns(self,
                                       batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values: torch.Tensor = self.values[batch_indices]
        advantages: torch.Tensor = self.advantages[batch_indices]
        returns: torch.Tensor = self.returns[batch_indices]
        return values, advantages, returns

    def _convert_rewards(self):
        self.rewards = torch.tensor(self.rewards, device=self.device)

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

        To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        Args:
            last_value: state value estimation for the last step (one for each env)

        Returns:

        """
        self.advantages = torch.empty(self.buffer_size, device=self.device)
        not_dones = 1.0 - self.dones  # 1.0 if not done, 0.0 if done

        # difference in values between r+V(s_{t+1}) and V(s), i.e., delta = r + gamma * V(s_{t+1}) - V(s)
        next_values = torch.cat((self.values[1:], torch.tensor([last_value], device=self.device)), dim=0)
        deltas = self.rewards + self.discount_factor * next_values * not_dones - self.values

        # generalized advantage estimate
        last_gae = 0
        for step in reversed(range(self.buffer_size)):
            # accumulate for generalized advantage estimate
            last_gae = deltas[step] + self.discount_factor * self.gae_lambda * not_dones[step] * last_gae
            self.advantages[step] = last_gae

        self.returns = self.advantages + self.values
        self.returns = self.returns.float()

    @property
    def explained_variance(self) -> np.ndarray:
        """
        Explanation that the value function gives about the actual return.
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
        Returns:

        """
        from stable_baselines3.common.utils import explained_variance as explained_variance_function
        from util.torch_util.torch_util import detach
        # calculate explained variance for the full buffer rather than individual batches
        return explained_variance_function(detach(self.values.flatten()),
                                           detach(self.returns.flatten()))
