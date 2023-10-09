import torch
import torch.nn as nn

from src.algorithms.rl.architectures.get_swarm_base import get_swarm_base
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase
from src.modules.mlp import MLP
from util.types import *


class DiscreteGraphQNet(nn.Module):
    """
    Q Network that evaluates an observation-action pair for a batch of agents. In our setting,
    each observation is a graph where each node corresponds to an agent. This network assumes that these graphs are
     preprocessed using a shared message passing GNN, and that it receives an agent observation of a given dimension
     for each agent.
    """

    def __init__(self,
                 environment: AbstractSwarmEnvironment,
                 network_config: ConfigDict,
                 dueling: bool,
                 device: torch.device):
        """

        Args:
            environment:
            network_config:
        """
        super(DiscreteGraphQNet, self).__init__()
        mlp_input_dimension = network_config.get("latent_dimension")
        latent_dimension = network_config.get("latent_dimension")
        critic_mlp = network_config.get("critic").get("mlp")

        self.graph_base: AbstractMessagePassingBase = get_swarm_base(graph_env=environment,
                                                                     network_config=network_config,
                                                                     device=device)

        # dueling: use a different architecture and compose the q-values from value and advantage
        self.dueling = dueling

        if self.dueling:
            self.value_mlp = MLP(in_features=mlp_input_dimension,
                                 config=critic_mlp,
                                 latent_dimension=latent_dimension,
                                 out_features=1,
                                 device=device)
            self.advantage_mlp = MLP(in_features=mlp_input_dimension,
                                     config=critic_mlp,
                                     latent_dimension=latent_dimension,
                                     out_features=environment.action_dimension,
                                     device=device)
        else:
            self.q_value_mlp = MLP(in_features=mlp_input_dimension,
                                   config=critic_mlp,
                                   latent_dimension=latent_dimension,
                                   out_features=environment.action_dimension,
                                   device=device)

        self._agent_node_type = environment.agent_node_type
        self._environment = environment

    def forward(self, observations: InputBatch) -> torch.Tensor:
        """
        Predicts the actions for the given observations by
        * feeding them through the graph base to get latent *node/agent* features
        * using the value mlp to get an action distribution per *node*
        Args:
            observations: A batch of observation graphs

        Returns: A tensor of shape (num_total_agents, action_dimension)
            containing the action distribution for each agent
        """
        node_features, _, _, batches = self.graph_base(observations)
        value_node_features = node_features.get(self._agent_node_type)

        if self.dueling:
            value = self.value_mlp(value_node_features)
            advantage = self.advantage_mlp(value_node_features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_value_mlp(value_node_features)
        return q_values
