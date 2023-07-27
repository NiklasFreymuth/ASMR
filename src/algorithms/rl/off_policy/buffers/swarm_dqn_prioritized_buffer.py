import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.algorithms.rl.off_policy.buffers.swarm_dqn_buffer import DQNBufferSamples, SwarmDQNBuffer, DQNTransition
from src.algorithms.rl.off_policy.util.segment_tree import SumSegmentTree, MinSegmentTree
from util.types import *


@dataclass
class DQNPrioritizedBufferSamples(DQNBufferSamples):
    weights: torch.Tensor
    indices: List[int]


class SwarmDQNPrioritizedBuffer(SwarmDQNBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            buffer_size: int,
            device: torch.device,
            scalar_reward_and_done: bool = False,
            alpha: float = 0.6
    ):
        """Initialization."""
        super().__init__(buffer_size, device, scalar_reward_and_done)
        assert alpha >= 0
        self.max_size = buffer_size
        self.ptr = 0
        self._size = 0

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def _build_data_structure(self, buffer_size: int) -> list[Optional[DQNTransition]]:
        return [None] * buffer_size

    def _put_in_buffer(self, transition: DQNTransition) -> None:
        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, num_samples: int, beta: float = 0.4) -> DQNPrioritizedBufferSamples:
        """Sample a batch of experiences."""
        mini_batch, indices = self._sample_transitions_and_indices(num_samples, beta)
        return self._create_prioritized_buffer_samples(mini_batch, indices, beta)

    def _sample_transitions_and_indices(self, num_samples: int, beta: float) -> Tuple[
        Sequence[DQNTransition], List[int]]:
        assert self.size >= num_samples
        assert beta > 0

        indices = self._sample_proportional(num_samples)
        return [self.buffer[idx] for idx in indices], indices

    def _create_prioritized_buffer_samples(self, mini_batch: Sequence[DQNTransition], indices: List[int],
                                           beta: float) -> DQNPrioritizedBufferSamples:
        dqn_buffer_sample = self._create_buffer_samples(mini_batch)
        weights = torch.tensor([self._calculate_weight(i, beta) for i in indices])
        return DQNPrioritizedBufferSamples(dqn_buffer_sample.observations, dqn_buffer_sample.actions,
                                           dqn_buffer_sample.rewards,
                                           dqn_buffer_sample.next_observations, dqn_buffer_sample.dones, weights,
                                           indices)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.size

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, num_samples) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, self.size - 1)
        segment = p_total / num_samples

        for i in range(num_samples):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size) ** (-beta)
        weight = weight / max_weight

        return weight

    @property
    def size(self):
        return self._size
