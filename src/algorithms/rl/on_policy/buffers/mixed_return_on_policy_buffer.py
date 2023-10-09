import numpy as np
import torch

from src.algorithms.rl.on_policy.buffers.spatial_on_policy_buffer import SpatialOnPolicyBuffer
from util import keys as k
from util.types import *


class MixedRewardOnPolicyBuffer(SpatialOnPolicyBuffer):
    """
    Multi Agent On-Policy Rollout Buffer acting on values/advantages/returns per *node* and *graph* via a mixed_return
    learning approach.
    """

    def __init__(self, buffer_size: int, gae_lambda: float, discount_factor: float, mixed_return_config: ConfigDict,
                 device: Optional[torch.device] = None, **kwargs):
        assert mixed_return_config is not None, f"Need to provide a mixed_return dictionary, given '{kwargs}'"
        super().__init__(buffer_size=buffer_size, gae_lambda=gae_lambda, discount_factor=discount_factor,
                         device=device, **kwargs)
        self._global_rewards: Union[torch.Tensor, List[torch.Tensor]] = []  # list of global rewards per step

        self._global_weight = mixed_return_config.get("global_weight")
        self._iteration = 0

    def reset(self) -> None:
        super().reset()
        self._global_rewards = []

    def add(self, *, observation: Data, actions: torch.Tensor, reward: np.array, done: float,
            value: torch.Tensor, log_probabilities: torch.Tensor, **kwargs) -> None:
        assert "additional_information" in kwargs, "additional_information must be provided"
        additional_information = kwargs["additional_information"]

        assert k.GLOBAL_REWARD in additional_information, "global_reward must be provided"
        self._global_rewards.append(additional_information[k.GLOBAL_REWARD])

        super().add(observation=observation, actions=actions, reward=reward, done=done,
                    value=value, log_probabilities=log_probabilities, **kwargs)

    def _convert_rewards(self):
        """
        Convert rewards to torch tensors and project them to the previous step if necessary
        Returns:
        """
        if len(self._global_rewards) > 0:
            self._global_rewards = torch.tensor(self._global_rewards, device=self.device)
        # per-agent rewards have irregular shape, so can not be converted

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
        super().compute_returns_and_advantage(last_value=last_value)  # sets local self.advantages, self.returns

        last_value = last_value.to(self.device)  # move to correct device

        self.advantages: List[torch.Tensor]  # local advantages
        self.returns: List[torch.Tensor]  # local returns

        # if global rewards are available, use them together with the local rewards to get a more hollistic view
        # on the reward.
        # for this,
        #  1) calculate the global returns. This is done by calculating the cumulative sum of the global rewards
        #  2) compute a weighted sum of the local and global returns

        # 1: calculate global returns and advantage
        global_advantages, global_returns = self._get_global_advantages_and_returns(last_value=last_value)

        # 2: compute weighted sum of local and global returns
        # compute the local vs global weight
        local_weight = 1 - self.global_weight

        # replace local returns and advantages by the weighted sum of local and global returns and advantages
        self.returns = [local_weight * return_tensor + self.global_weight * global_return_tensor
                        for return_tensor, global_return_tensor in zip(self.returns, global_returns)]

        self.advantages = [_return - value for value, _return in zip(self.values, self.returns)]

        self._iteration += 1

    def _get_global_advantages_and_returns(self, last_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the advantages for the global rewards.
        """
        global_advantages = torch.empty(self.buffer_size, device=self.device)
        not_dones = 1.0 - self.dones  # 1.0 if not done, 0.0 if done

        global_last_value = last_value.mean()
        global_values = torch.tensor([values.mean() for values in self.values], device=self.device)
        # difference in values between r+V(s_{t+1}) and V(s), i.e., delta = r + gamma * V(s_{t+1}) - V(s)
        next_values = torch.cat((global_values[1:], torch.tensor([global_last_value])), dim=0)
        deltas = self._global_rewards + self.discount_factor * next_values * not_dones - global_values

        # generalized advantage estimate
        last_gae = 0
        for step in reversed(range(self.buffer_size)):
            # accumulate for generalized advantage estimate
            last_gae = deltas[step] + self.discount_factor * self.gae_lambda * not_dones[step] * last_gae
            global_advantages[step] = last_gae

        global_returns = global_advantages + global_values
        return global_advantages, global_returns.float()

    @property
    def global_weight(self):
        return self._global_weight
