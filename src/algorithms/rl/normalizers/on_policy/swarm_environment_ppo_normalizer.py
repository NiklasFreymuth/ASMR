import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.algorithms.rl.normalizers.swarm_environment_observation_normalizer import SwarmEnvironmentObservationNormalizer
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.types import *


class SwarmEnvironmentPPONormalizer(SwarmEnvironmentObservationNormalizer):
    """
    Extends the observation normalizer to also normalize the rewards via their returns
    """

    def __init__(self,
                 graph_environment: AbstractSwarmEnvironment,
                 discount_factor: float,
                 normalize_rewards: bool,
                 normalize_nodes: bool,
                 normalize_edges: bool,
                 normalize_globals: bool,
                 reward_clip: float = 5,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6
                 ):
        super().__init__(graph_environment=graph_environment, normalize_edges=normalize_edges,
                         normalize_globals=normalize_globals, normalize_nodes=normalize_nodes,
                         observation_clip=observation_clip, epsilon=epsilon)

        if normalize_rewards:
            self.reward_normalizer = RunningMeanStd(epsilon=epsilon, shape=(1,))
            self.returns = np.array([0])
        else:
            self.reward_normalizer = None
            self.returns = None

        self.discount_factor = discount_factor
        self.reward_clip = reward_clip

    def reset(self, observations: InputBatch) -> InputBatch:
        """
        To be called after the reset() of the environment is called. Used to update the normalizer statistics
        with the initial observations of the environment, and potentially reset parts of the normalizer that
        depend on the environment episode
        Args:
            observations: the initial observations of the environment

        Returns: the normalized observations

        """
        if self.returns is not None:
            self.returns = np.array([0])
        return super().reset(observations=observations)

    def update_and_normalize(self, observations: InputBatch, reward: float = None) -> Tuple[InputBatch, float]:
        """
        Normalize the given observations and the given reward with the current statistics. Normalized nodes, edges,
        globals and the reward independently. Adapts the statistics if train==True.
        Args:
            observations: Current observation graph(s)
            reward: Reward of the current step. Can be None for reset() steps.

        Returns: A tuple of the normalized observations and the normalized reward

        """
        if reward is not None:
            self._update_reward(reward)
            reward = self._normalize_reward(reward)

        self.update_observations(observations=observations)
        observations = self.normalize_observations(observations=observations)
        return observations, reward

    def _update_reward(self, reward):
        """
        Update the reward statistics with the given reward and the current returns
        """
        if self.reward_normalizer is not None and reward is not None:
            # normalize reward according to PPO update rule. Taken from stable_baselines3
            self.returns = self.returns * self.discount_factor + np.mean(reward)
            # we just take the mean over rewards for now, as there may be a different number of rewards each step
            self.reward_normalizer.update(self.returns)

    def _normalize_reward(self, reward: float) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize rewards statistics.
        Calling this method does not update statistics.
        """
        if self.reward_normalizer is not None:
            scaled_reward = reward / np.sqrt(self.reward_normalizer.var + self.epsilon)
            scaled_reward = np.clip(scaled_reward, -self.reward_clip, self.reward_clip)[0]
        else:
            scaled_reward = reward
        return scaled_reward
