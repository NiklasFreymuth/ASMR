import torch

from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.types import *
from util.torch_util.torch_running_mean_std import TorchRunningMeanStd


class SweepEnvironmentNormalizer(AbstractEnvironmentNormalizer):

    def __init__(self, graph_environment: AbstractSwarmEnvironment,
                 normalize_nodes: bool,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6, *args, **kwargs):
        if normalize_nodes:
            num_node_features = graph_environment.num_node_features
            if isinstance(graph_environment.num_node_features, int):
                pass
            elif isinstance(graph_environment.num_node_features, dict):
                num_node_features = num_node_features[graph_environment.agent_node_type]
            else:
                raise ValueError(f"Unknown type for num_node_features: {type(graph_environment.num_node_features)}")

            self.node_normalizers = TorchRunningMeanStd(epsilon=epsilon,
                                                        shape=(num_node_features,))

        self.epsilon = epsilon
        self.observation_clip = observation_clip
        super().__init__(*args, **kwargs)

    def reset(self, observations: InputBatch) -> InputBatch:
        self.update_observations(observations=observations)
        return self.normalize_observations(observations=observations)

    def update_and_normalize(self, observations: InputBatch, **kwargs) -> Union[InputBatch, Tuple]:
        self.update_observations(observations=observations)
        observations = self.normalize_observations(observations=observations)
        if len(kwargs) > 0:
            return observations, *kwargs.values()
        else:
            return observations

    def update_observations(self, observations: InputBatch) -> None:
        self.node_normalizers.update(observations)

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        """
        Normalize observations using this instances current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        return self._normalize_observation(observations, self.node_normalizers)

    def _normalize_observation(self, observation: Tensor, normalizer: TorchRunningMeanStd) -> Tensor:
        """
        Helper to normalize a given observation.
        * param observation:
        * param normalizer: associated statistics
        * return: normalized observation
        """
        scaled_observation = (observation - normalizer.mean) / torch.sqrt(normalizer.var + self.epsilon)
        scaled_observation = torch.clip(scaled_observation, -self.observation_clip, self.observation_clip)
        return scaled_observation.float()
