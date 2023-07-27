import torch

from src.algorithms.baselines.architectures.sweep.sweep_ppo_actor_critic import SweepPPOActorCritic
from src.algorithms.baselines.buffers.single_agent_on_policy_buffer import SingleAgentOnPolicyBuffer
from src.algorithms.rl.architectures.swarm_ppo_actor_critic import SwarmPPOActorCritic
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from src.algorithms.rl.on_policy.swarm_ppo import SwarmPPO
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.environments.mesh.mesh_refinement.sweep.sweep_mesh_refinement import SweepMeshRefinement
from util.types import *


class SweepPPO(SwarmPPO):
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

    def __init__(self, config: ConfigDict,
                 environment: Optional[AbstractSwarmEnvironment] = None,
                 evaluation_environments: Optional[List[AbstractSwarmEnvironment]] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(config=config,
                         environment=environment,
                         evaluation_environments=evaluation_environments,
                         seed=seed)
        self._value_function_scope = "agent"

    def _kickoff_environments(self):
        # tell environments whether they are in training or inference mode
        # for this baseline, the training environments only mark a single element in each step, while the inference
        # sweeps over all elements in the graph in parallel
        self._environment.train(True)
        for evaluation_env in self.evaluation_environments:
            evaluation_env: SweepMeshRefinement
            evaluation_env.train(False)

        super()._kickoff_environments()

    def _build_normalizer(self, ppo_config: ConfigDict) -> AbstractEnvironmentNormalizer:
        normalize_observations = ppo_config.get("normalize_observations")
        if normalize_observations:
            from src.algorithms.baselines.normalizers.sweep_environment_normalizer import SweepEnvironmentNormalizer
            environment_normalizer = SweepEnvironmentNormalizer(graph_environment=self._environment,
                                                                normalize_nodes=normalize_observations)
        else:
            from src.algorithms.rl.normalizers.dummy_swarm_environment_normalizer import DummySwarmEnvironmentNormalizer
            environment_normalizer = DummySwarmEnvironmentNormalizer()
        return environment_normalizer

    def _build_buffer(self, ppo_config):
        sample_buffer_on_gpu = self.config["algorithm"]["sample_buffer_on_gpu"]
        gae_lambda: float = ppo_config.get("gae_lambda", 0.95)
        buffer_device = self.device if sample_buffer_on_gpu else torch.device("cpu")
        rollout_buffer = SingleAgentOnPolicyBuffer(buffer_size=self._num_rollout_steps,
                                                   gae_lambda=gae_lambda,
                                                   discount_factor=self._discount_factor,
                                                   agent_node_type=self.environment.agent_node_type,
                                                   device=buffer_device)
        return rollout_buffer

    def _build_policy(self, ppo_config: ConfigDict) -> SwarmPPOActorCritic:
        return SweepPPOActorCritic(environment=self._environment,
                                   network_config=self._network_config,
                                   use_gpu=self.algorithm_config.get("use_gpu"),
                                   ppo_config=ppo_config)
