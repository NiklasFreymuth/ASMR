import torch
from stable_baselines3.common.distributions import Distribution

from src.algorithms.rl.architectures.swarm_ppo_actor_critic import SwarmPPOActorCritic
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.types import *


class SweepPPOActorCritic(SwarmPPOActorCritic):
    def __init__(self, environment: AbstractSwarmEnvironment,
                 network_config: ConfigDict, ppo_config: ConfigDict, use_gpu: bool):
        super(SweepPPOActorCritic, self).__init__(environment=environment,
                                                  network_config=network_config,
                                                  use_gpu=use_gpu,
                                                  ppo_config=ppo_config)
        self._value_function_scope = "agent"

    def _get_values_and_distribution(self, observations: InputBatch,
                                     value_function_scope: Optional[str] = None) -> Tuple[Distribution, torch.Tensor]:
        """
        Processes the observations by
        * feeding them through the shared graph base to get latent *node/agent* features
        * using the policy_mlp and a linear mapping to get an action distribution per *node*
        * using the value_mlp and an aggregation to get a value.
          This value may either be per *graph* ([self._]value_function_scope=="graph") or
          per *node* ([self._]value_function_scope=="agent")

        Args:
            observations:
            value_function_scope: [Optional] Scope of the value function. If None, will use self._value_function_scope.
                May be either "graph" or "agent"

        Returns: A tuple
            distribution: One diagonal Gaussian distribution per *agent*
            values: One value per *graph* or per *node*.
            For graphs, the value is the aggregation over the values for all nodes/agents.

        """
        if self.share_base:  # one shared base for value and policy
            node_features = self.graph_base(observations)
            value_node_features = node_features
            policy_node_features = node_features
        else:  # a base each for value and policy
            value_node_features = self.value_base(observations)
            policy_node_features = self.policy_base(observations)

        values = self.value_mlp(value_node_features)
        latent_policy_features = self.policy_mlp(policy_node_features)
        distribution = self._get_action_distribution_from_latent(latent_policy_features)

        return distribution, values
