import gym
import torch
import torch.optim as optim
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution, \
    Distribution, BernoulliDistribution
from torch_scatter import scatter_mean

from src.algorithms.rl.architectures.get_swarm_base import get_swarm_base
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.modules.abstract_architecture import AbstractArchitecture
from src.modules.mlp import MLP
from src.modules.mpn.message_passing_base import MessagePassingBase
from util.types import *


class SwarmPPOActorCritic(AbstractArchitecture):
    def __init__(self, environment: AbstractSwarmEnvironment,
                 network_config: ConfigDict, ppo_config: ConfigDict, use_gpu: bool):
        """
        Initializes the actor-critic network for PPO, i.e., the value network and the policy network.
        Args:
            environment: The environment to train the reinforcement algorithm on.
            network_config: Configuration for the policy and value networks.
            ppo_config: Configuration for the PPO algorithm
            use_gpu: Whether to use the GPU
        """
        super(SwarmPPOActorCritic, self).__init__(use_gpu=use_gpu,
                                                  network_config=network_config,
                                                  ppo_config=ppo_config)
        self._agent_node_type = environment.agent_node_type

        self._value_function_scope = ppo_config.get("value_function_scope")

        self.policy_base: MessagePassingBase = get_swarm_base(graph_env=environment,
                                                              network_config=network_config,
                                                              device=self._gpu_device)
        self.value_base: MessagePassingBase = get_swarm_base(graph_env=environment,
                                                             network_config=network_config,
                                                             device=self._gpu_device)

        training_config = network_config.get("training")
        latent_dimension = network_config.get("latent_dimension")

        mlp_config = {"activation_function": "tanh",
                      "num_layers": 2}
        self.value_mlp = MLP(in_features=latent_dimension,
                             config=mlp_config,
                             latent_dimension=latent_dimension,
                             out_features=1,
                             device=self._gpu_device)
        self.policy_mlp = MLP(in_features=latent_dimension,
                              config=mlp_config,
                              latent_dimension=latent_dimension,
                              device=self._gpu_device)

        action_dimension = environment.action_dimension
        if isinstance(environment.agent_action_space, gym.spaces.Box):
            self.action_distribution = DiagGaussianDistribution(action_dim=action_dimension)
            self.action_out_embedding, self.log_std = self.action_distribution.proba_distribution_net(
                latent_dim=self.policy_mlp.out_features, log_std_init=0.0
            )
            self.proba_distribution = partial(self.action_distribution.proba_distribution, log_std=self.log_std)
        elif isinstance(environment.agent_action_space, gym.spaces.Discrete):
            if action_dimension == 1:
                self.action_distribution = BernoulliDistribution(action_dims=action_dimension)
                self.proba_distribution = self.action_distribution.proba_distribution
            else:
                self.action_distribution = CategoricalDistribution(action_dim=action_dimension)
                self.proba_distribution = self.action_distribution.proba_distribution
            self.action_out_embedding = self.action_distribution.proba_distribution_net(
                latent_dim=self.policy_mlp.out_features
            )
            self.log_std = None
        else:
            raise NotImplementedError(f"Action type '{environment.agent_action_type} not implemented")

        self._initialize_optimizer_and_scheduler(training_config=training_config)

        self.to(self._gpu_device)

    def forward(self, observations: InputBatch, deterministic: bool = False,
                value_function_scope: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            observations:  Observation graph
            deterministic: Whether to sample or use deterministic actions
            value_function_scope: [Optional] Scope of the value function. If None, will use self._value_function_scope.
                May be either "graph" or "agent"

        Returns:
            -An action *per agent*,
            -A scalar *per graph* iff self.value_function_scope is set to "graph", and per agent otherwise
            - A log_probability of taking this action  *per agent*

        """
        observations = self.to_gpu(observations)
        distribution, values = self._get_values_and_distribution(observations,
                                                                 value_function_scope=value_function_scope)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probability = distribution.log_prob(actions)
        return actions, values, log_probability

    def evaluate_actions(self, observations: InputBatch,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy given the observations.

        :param observations: Observation Graph(s)
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        observations = self.to_gpu(observations)
        actions = self.to_gpu(actions)
        distribution, values = self._get_values_and_distribution(observations)

        log_probabilities = distribution.log_prob(actions)
        return values, log_probabilities, distribution.entropy()

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
        node_features, _, _, batches = self.value_base(observations)
        batch = batches.get(self._agent_node_type)
        value_node_features = node_features.get(self._agent_node_type)
        node_features, _, _, _ = self.policy_base(observations)
        policy_node_features = node_features.get(self._agent_node_type)
        latent_policy_features = self.policy_mlp(policy_node_features)
        distribution = self._get_action_distribution_from_latent(latent_policy_features)

        values = self.value_mlp(value_node_features)
        if value_function_scope is None:
            value_function_scope = self._value_function_scope
        if value_function_scope == "graph":
            # aggregate over evaluations per node to get one evaluation per graph
            values = scatter_mean(values, batch, dim=0)
        elif value_function_scope in ["spatial"]:
            # keep one evaluation per node
            pass
        else:
            raise ValueError(f"Unknown value function scope '{value_function_scope}'")
        return distribution, values

    def _get_action_distribution_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve the action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_out_embedding(latent_pi)
        return self.proba_distribution(mean_actions)

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer
