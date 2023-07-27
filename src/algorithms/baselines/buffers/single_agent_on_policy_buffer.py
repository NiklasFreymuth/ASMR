import numpy as np
import torch

from src.algorithms.rl.on_policy.buffers.graph_on_policy_buffer import GraphOnPolicyBuffer


class SingleAgentOnPolicyBuffer(GraphOnPolicyBuffer):
    """
    Rollout buffer used in the Deep Reinforcement Learning for Adaptive Mesh Refinement
    (https://arxiv.org/pdf/2209.12351.pdf) baseline.
    Rollout transitions consist of graph observation, an index indicating which node within the graph is seen as agent
    and an action i.e. a scalar value for this agent.
    """

    def _get_observation_batch(self, batch_indices: np.ndarray) -> torch.Tensor:
        observations = [self.observations[index] for index in batch_indices]
        return torch.cat(observations, dim=0)
