from typing import Optional

import torch

from src.algorithms.rl.on_policy.buffers.abstract_multi_agent_on_policy_buffer import AbstractMultiAgentOnPolicyBuffer


def get_on_policy_buffer(buffer_size: int, gae_lambda: float, discount_factor: float,
                         value_function_scope: str, mixed_return_config: Optional[dict] = None,
                         device: Optional[torch.device] = None) -> AbstractMultiAgentOnPolicyBuffer:
    """

    Args:
        buffer_size:
        gae_lambda:
        discount_factor:
        value_function_scope: The scope of the value function. Either "agent" for a value per agent/node or "graph"
        for a single value for each graph/set of agents
        mixed_return_config: A dictionary containing the configuration for the mixed_return learning. If this is None or,
        the maximum global weight for the mixed_return is 0, no mixed_return learning is used

    Returns: A corresponding OnPolicyBuffer that can deal with either node- or graph-wise value functions

    """
    use_mixed_return = mixed_return_config.get("global_weight", 0) > 0
    if value_function_scope == "spatial":
        if use_mixed_return:
            from src.algorithms.rl.on_policy.buffers.mixed_return_on_policy_buffer import MixedRewardOnPolicyBuffer
            return MixedRewardOnPolicyBuffer(buffer_size=buffer_size,
                                             gae_lambda=gae_lambda,
                                             discount_factor=discount_factor,
                                             device=device,
                                             mixed_return_config=mixed_return_config)
        else:
            from src.algorithms.rl.on_policy.buffers.spatial_on_policy_buffer import SpatialOnPolicyBuffer
            on_policy_buffer = SpatialOnPolicyBuffer
    elif value_function_scope in ["graph", "vdn"]:
        from src.algorithms.rl.on_policy.buffers.graph_on_policy_buffer import GraphOnPolicyBuffer

        on_policy_buffer = GraphOnPolicyBuffer
    else:
        raise NotImplementedError(f"Unknown value_function_scope '{value_function_scope}'")
    return on_policy_buffer(buffer_size=buffer_size,
                            gae_lambda=gae_lambda,
                            discount_factor=discount_factor,
                            device=device)
