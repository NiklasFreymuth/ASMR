from src.algorithms.baselines.architectures.vdgn_policy import VDGNPolicy
from src.algorithms.rl.off_policy.swarm_dqn import SwarmDQN


class VDGN(SwarmDQN):

    @property
    def scalar_rewards_and_dones(self):
        """
        VDGN uses scalar rewards and dones, and learns the credit assignment problem via VDNs
        Returns:

        """
        return True

    @property
    def policy_class(self):
        return VDGNPolicy
