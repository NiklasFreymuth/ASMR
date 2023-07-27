from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from util.types import *


@dataclass
class AbstractBufferSamples:
    observations: InputBatch  # also includes agent mappings
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: InputBatch
    dones: torch.Tensor


class AbstractOffPolicyBuffer:
    def __init__(self, buffer_size, device: torch.device):
        self.device = device
        self.buffer = self._build_data_structure(buffer_size)

    @abstractmethod
    def _build_data_structure(self, buffer_size: int) -> Sequence:
        raise NotImplementedError

    @abstractmethod
    def put(self, observation: InputBatch, actions: Tensor, reward: np.array,
            next_observation: InputBatch, done: np.array) -> None:
        """
        Add a single graph-based update step
        Args:
            observation: Dictionary describing the relation between the agents before the update step
            actions: Actions taken by each agent
            reward: Scalar reward for performing the specified action in the current observation
            next_observation: Dictionary describing the relation between the agents after the update step
            done: Boolean flag on whether the environment ended after performing this action or not. Repeated for
            each agent of the environment

        Returns:

        """
        raise NotImplementedError

    def sample(self, num_samples: int) -> AbstractBufferSamples:
        """
        Sample num_samples samples from the replay buffer
        Args:
            num_samples:

        Returns: AbstractBufferSamples
        """
        raise NotImplementedError

    @property
    def size(self):
        return len(self.buffer)
