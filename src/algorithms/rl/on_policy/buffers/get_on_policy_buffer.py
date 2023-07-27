from typing import Optional

import torch

from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import AbstractMultiAgentOnPolicyBuffer


def get_on_policy_buffer(buffer_size: int, gae_lambda: float, discount_factor: float,
                         value_function_scope: str, use_mixed_reward: bool = False,
                         device: Optional[torch.device] = None, **kwargs) -> AbstractMultiAgentOnPolicyBuffer:
    """

    Args:
        buffer_size:
        gae_lambda:
        discount_factor:
        value_function_scope: The scope of the value function. Either "agent" for a value per agent/node or "graph"
        for a single value for each graph/set of agents
        use_mixed_reward: Whether to use a mixed reward function

    Returns: A corresponding OnPolicyBuffer that can deal with either node- or graph-wise value functions

    """
    if value_function_scope == "spatial":
        if use_mixed_reward:
            from src.algorithms.rl.on_policy.buffers.mixed_reward_on_policy_buffer import MixedRewardOnPolicyBuffer
            return MixedRewardOnPolicyBuffer(buffer_size=buffer_size,
                                             gae_lambda=gae_lambda,
                                             discount_factor=discount_factor,
                                             device=device,
                                             **kwargs)
        else:
            from src.algorithms.rl.on_policy.buffers.spatial_on_policy_buffer import SpatialOnPolicyBuffer
            on_policy_buffer = SpatialOnPolicyBuffer
    elif value_function_scope == "graph":
        from src.algorithms.rl.on_policy.buffers.graph_on_policy_buffer import GraphOnPolicyBuffer

        on_policy_buffer = GraphOnPolicyBuffer
    else:
        raise NotImplementedError(f"Unknown value_function_scope '{value_function_scope}'")
    return on_policy_buffer(buffer_size=buffer_size,
                            gae_lambda=gae_lambda,
                            discount_factor=discount_factor,
                            device=device,
                            **kwargs)
