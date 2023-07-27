import collections
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.algorithms.rl.off_policy.buffers.abstract_off_policy_buffer import AbstractOffPolicyBuffer, \
    AbstractBufferSamples
from src.modules.mpn.common.hmpn_util import make_batch
from util.types import *


@dataclass
class DQNBufferSamples(AbstractBufferSamples):
    # has same fields as AbstractBufferSamples
    pass


# custom type for transitions stored in the Buffer
# observation, action, reward, next_observation, done, optional agent_mapping
DQNTransition = Tuple[
    InputBatch, Tensor, Union[np.array, float, torch.Tensor], InputBatch, Union[np.array, bool, torch.Tensor], Optional[
        Union[np.array, torch.Tensor]]]


class SwarmDQNBuffer(AbstractOffPolicyBuffer):
    def __init__(self, buffer_size: int, device: torch.device, scalar_reward_and_done: bool = False):
        super().__init__(buffer_size, device)
        # if scalar_reward_and_done is True, then the reward and done are scalars and not arrays.
        # This is the case for VDGN.
        self.scalar_reward_and_done = scalar_reward_and_done

    def _build_data_structure(self, buffer_size: int) -> collections.deque:
        return collections.deque(maxlen=buffer_size)

    def put(self, observation: InputBatch, actions: Tensor, reward: Union[float, np.array],
            next_observation: InputBatch, done: Union[bool, np.array],
            agent_mapping: Optional[np.array] = None) -> None:
        """
        Add a single graph-based update step
        Args:
            observation: Dictionary describing the relation between the agents before the update step
            actions: Actions taken by each agent
            reward: Scalar reward for performing the specified action in the current observation
            next_observation: Dictionary describing the relation between the agents after the update step
            done: Boolean flag on whether the environment ended after performing this action or not. Repeated for
            each agent of the environment
            agent_mapping: Mapping from the agent index in the observation/action to the agent index
              in the next observation

        Returns:

        """
        transition = self._transition_to_device((observation, actions, reward, next_observation, done, agent_mapping))
        self._put_in_buffer(transition)

    def _transition_to_device(self, transition: DQNTransition) -> DQNTransition:
        observation, actions, reward, next_observation, done, agent_mapping = transition
        observation = observation.to(self.device)
        actions = actions.to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_observation = next_observation.to(self.device)
        done = torch.tensor(done).to(self.device)
        if agent_mapping is not None:
            agent_mapping = torch.tensor(agent_mapping).to(self.device)
        return observation, actions, reward, next_observation, done, agent_mapping

    def _put_in_buffer(self, transition: DQNTransition) -> None:
        self.buffer.append(transition)

    def sample(self, num_samples: int) -> DQNBufferSamples:
        """
        Sample num_samples samples from the replay buffer
        Args:
            num_samples: int, number of samples to sample

        Returns: A DQNBufferSamples object containing the sampled transitions
        """
        mini_batch = self._sample_transitions(num_samples)
        return self._create_buffer_samples(mini_batch)

    def _sample_transitions(self, num_samples: int) -> Sequence[DQNTransition]:
        return random.sample(self.buffer, num_samples)

    def _create_buffer_samples(self, mini_batch: Sequence[DQNTransition]) -> DQNBufferSamples:
        observations, actions, rewards, next_observations, dones, agent_mappings = (list(x) for x in zip(*mini_batch))

        # convert to flat tensors
        if self.scalar_reward_and_done:
            rewards = torch.tensor(rewards)
            dones = torch.tensor(dones)
        else:
            rewards = torch.cat(rewards, dim=0)
            dones = torch.cat(dones, dim=0)
        actions = torch.cat(actions, dim=0)

        if agent_mappings is not None:
            for i in range(len(observations)):
                observations[i].agent_mapping = agent_mappings[i]
            observations = make_batch(observations, follow_batch=['agent_mapping'])
        else:
            observations = make_batch(observations)
        next_observations = make_batch(next_observations)

        return DQNBufferSamples(observations, actions, rewards, next_observations, dones)
