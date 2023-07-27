import torch

from src.algorithms.baselines.architectures.argmax_dqn_policy import ArgmaxDQNPolicy
from src.algorithms.rl.off_policy.swarm_dqn import SwarmDQN
from src.modules.mpn.common.hmpn_util import make_batch
from util.types import *


class ArgmaxDQN(SwarmDQN):

    def _get_random_actions(self):
        # environment action space must be discrete
        actions = torch.randint(low=0,
                                high=self._environment.num_agents,
                                size=(1,)).to(self.device)
        return actions

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

        values, actions = q_values.max(dim=0)
        return actions, values

    @property
    def scalar_rewards_and_dones(self):
        """
        DQN uses rewards/dones per agent instead of per graph.
        Returns:

        """
        return True

    @property
    def policy_class(self):
        return ArgmaxDQNPolicy
