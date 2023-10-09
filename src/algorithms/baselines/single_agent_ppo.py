import torch
from torch.nn import functional as F

from src.algorithms.rl.architectures.swarm_ppo_actor_critic import SwarmPPOActorCritic
from src.algorithms.rl.on_policy.swarm_ppo import SwarmPPO
from util.types import *


class SingleAgentPPO(SwarmPPO):
    """
    Baseline Implementation of the paper Deep Reinforcement Learning for Adaptive Mesh Refinement.
    (https://arxiv.org/pdf/2209.12351.pdf) compatible with our GraphEnvironments.
    As underlying Reinforcement Learning algorithm, this implementation uses Proximal Policy Optimization (PPO).

    Implementation differs from our implementation in the following points:
    - Action strategy
    - Reward calculation
    - Policy network (& no GNN message passing).
    - Observation space i.e. features an agent gets/uses for decision-making.
    """

    def _build_policy(self, ppo_config: ConfigDict) -> SwarmPPOActorCritic:
        return SingleAgentPPOActorCritic(environment=self._environment,
                                    network_config=self._network_config,
                                    use_gpu=self.algorithm_config.get("use_gpu"),
                                    ppo_config=ppo_config)


class SingleAgentPPOActorCritic(SwarmPPOActorCritic):

    def forward(self, observations: InputBatch, deterministic: bool = False,
                value_function_scope: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: A tuple
            action: The sampled action as an integer (index of the action)
            values: The value function
            log_probabilities: The log probability of the sampled action under the current policy

        """
        observations = self.to_gpu(observations)
        distribution, values = self._get_values_and_distribution(observations,
                                                                 value_function_scope=value_function_scope)
        one_probabilities = distribution.distribution.probs
        action_probabilities = F.softmax(one_probabilities, dim=0).squeeze(1)

        if deterministic:  # choose the action with the highest probability
            action = torch.argmax(one_probabilities, dim=0)
        else:  # sample an action from the distribution
            action = torch.multinomial(action_probabilities, num_samples=1)
        log_probabilities = torch.log(action_probabilities[action])
        return action, values, log_probabilities

    def evaluate_actions(self, observations: InputBatch,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: A tuple
            values: The value function estimate for the current observations
            log_probabilities: The log probability of the sampled action under the current policy
            entropy: The entropy of the action distribution

        """
        observations = self.to_gpu(observations)
        actions = self.to_gpu(actions)
        distribution, values = self._get_values_and_distribution(observations)
        one_probabilities = distribution.distribution.probs

        ptr = observations.ptr
        log_probabilities = torch.zeros(len(ptr) - 1, dtype=torch.float32)
        entropies = torch.zeros(len(ptr) - 1, dtype=torch.float32)
        for idx, (batch_start, batch_end) in enumerate(zip(ptr[:-1], ptr[1:])):
            batch_one_probabilities = one_probabilities[batch_start:batch_end]
            batch_action = actions[idx]
            action_probabilities = F.softmax(batch_one_probabilities, dim=0).squeeze(1)
            log_probabilities[idx] = torch.log(action_probabilities[batch_action])
            entropies[idx] = -torch.sum(action_probabilities * torch.log(action_probabilities))
        return values, log_probabilities, entropies
