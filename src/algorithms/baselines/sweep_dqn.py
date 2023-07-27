import torch

from src.algorithms.baselines.architectures.sweep_dqn_architectures import SweepDQNPolicy
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from src.algorithms.rl.off_policy.buffers.swarm_dqn_buffer import SwarmDQNBuffer
from src.algorithms.rl.off_policy.buffers.swarm_dqn_prioritized_buffer import SwarmDQNPrioritizedBuffer
from src.algorithms.rl.off_policy.swarm_dqn import SwarmDQN
from util.types import *


class SweepDQN(SwarmDQN):
    def _kickoff_environments(self):
        # tell environments whether they are in training or inference mode
        # for this baseline, the training environments only mark a single element in each step, while the inference
        # sweeps over all elements in the graph in parallel
        self._environment.train(True)
        for evaluation_env in self.evaluation_environments:
            evaluation_env.train(False)

        super()._kickoff_environments()

    def _build_buffer(self, dqn_config: ConfigDict) -> Union[SwarmDQNBuffer, SwarmDQNPrioritizedBuffer]:
        sample_buffer_on_gpu = self.algorithm_config.get("sample_buffer_on_gpu")
        buffer_size = dqn_config.get("max_replay_buffer_size")
        buffer_device = self.device if sample_buffer_on_gpu else torch.device("cpu")
        if self.prioritized_buffer:
            buffer_config = dqn_config.get("prioritized_buffer")
            alpha = buffer_config.get("alpha")
            self._prioritized_buffer_beta_init = buffer_config.get("beta_init")
            self._prioritized_buffer_beta_final = buffer_config.get("beta_final")
            return SwarmDQNPrioritizedBuffer(buffer_size=buffer_size, device=buffer_device, alpha=alpha,
                                             scalar_reward_and_done=self.scalar_rewards_and_dones)
        else:
            return SwarmDQNBuffer(buffer_size=buffer_size,
                                  device=buffer_device, scalar_reward_and_done=self.scalar_rewards_and_dones)

    def _build_normalizer(self, dqn_config: ConfigDict) -> AbstractEnvironmentNormalizer:
        normalize_observations = dqn_config["normalize_observations"]
        if normalize_observations:
            from src.algorithms.baselines.normalizers.sweep_environment_normalizer import SweepEnvironmentNormalizer
            environment_normalizer = SweepEnvironmentNormalizer(graph_environment=self._environment,
                                                                normalize_nodes=normalize_observations)
        else:
            from src.algorithms.rl.normalizers.dummy_swarm_environment_normalizer import DummySwarmEnvironmentNormalizer
            environment_normalizer = DummySwarmEnvironmentNormalizer()
        return environment_normalizer

    def _get_random_actions(self):
        """
        For this baseline, we sample a random agent and a random action for that agent. All other agents are masked out,
        i.e., get a zero action.
        Returns:

        """
        action = torch.randint(low=0,
                               high=self._environment.action_dimension,
                               size=(1,)).to(self.device)
        return action

    @property
    def scalar_rewards_and_dones(self):
        """
        DQN uses rewards/dones per agent instead of per graph.
        Returns:

        """
        return True

    @property
    def policy_class(self):
        return SweepDQNPolicy
