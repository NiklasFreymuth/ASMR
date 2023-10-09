import torch
from torch.nn import functional as F


def get_entropy_loss(entropy, log_probabilities):
    """
    Entropy total_loss to favor exploration
    Args:
        entropy:
        log_probabilities:

    Returns:

    """
    if entropy is None:  # Approximate entropy when no analytical form is available
        entropy_loss = -torch.mean(-log_probabilities)
    else:
        entropy_loss = -torch.mean(entropy)
    return entropy_loss


def get_policy_loss(advantages: torch.Tensor, ratio: torch.Tensor, clip_range: float):
    """
    clipped surrogate total_loss for the policy.
    Note that we return the negative loss
    Args:
        advantages:
        ratio:
        clip_range: In [0,1]. Range to clip the ratio to.
            The ratio will be clipped between (1-clip_range, 1+clip_range).

    Returns:

    """
    # Normalize advantage
    advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)
    advantages = advantages.detach()

    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    return policy_loss


def get_value_loss(returns: torch.Tensor, values: torch.Tensor,
                   old_values: torch.Tensor, clip_range: float) -> torch.Tensor:
    """
    Value total_loss using the TD(gae_lambda) target for the value function
    Args:
        returns:
        values:
        old_values: Previous value function estimates. Used for clipping if value_function-clip_range > 0
        clip_range:

    Returns:

    """
    value_loss = F.mse_loss(returns, values)
    if clip_range > 0:
        # In OpenAI's PPO implementation, the value function is clipped around the previous value estimate
        # the worst of the clipped and unclipped versions is then used to train the value function
        values_clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
        value_loss_clipped = F.mse_loss(returns, values_clipped)
        value_loss = torch.max(value_loss, value_loss_clipped)
    return value_loss
