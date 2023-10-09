import torch

from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from util.torch_util.torch_running_mean_std import TorchRunningMeanStd
from util.types import *


class SwarmEnvironmentObservationNormalizer(AbstractEnvironmentNormalizer):

    def __init__(self,
                 graph_environment: AbstractSwarmEnvironment,
                 normalize_nodes: bool,
                 normalize_edges: bool,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6
                 ):
        """
        Normalizes the observations of a graph environment
        Args:
            graph_environment: the graph environment to normalize the observations of
            normalize_nodes: whether to normalize the node features
            normalize_edges: whether to normalize the edge features
            observation_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """
        super().__init__()
        if normalize_nodes:
            self.node_normalizers = TorchRunningMeanStd(epsilon=epsilon,
                                                        shape=(graph_environment.num_node_features,))

        else:
            self.node_normalizers = None

        if normalize_edges:
            self.edge_normalizers = TorchRunningMeanStd(epsilon=epsilon,
                                                        shape=(graph_environment.num_edge_features,))

        else:
            self.edge_normalizers = None

        self.epsilon = epsilon
        self.observation_clip = observation_clip

    def reset(self, observations: InputBatch) -> InputBatch:
        """
        To be called after the reset() of the environment is called. Used to update the normalizer statistics
        with the initial observations of the environment, and potentially reset parts of the normalizer that
        depend on the environment episode
        Args:
            observations: the initial observations of the environment

        Returns: the normalized observations

        """
        self.update_observations(observations=observations)
        return self.normalize_observations(observations=observations)

    def update_and_normalize(self, observations: InputBatch, **kwargs) -> Union[InputBatch, Tuple]:
        """
        Update the normalizer statistics with the given observations and return the normalized observations
        Args:
            observations: the observations to update the normalizer with
            **kwargs: additional arguments

        Returns: the normalized observations. Also returns all additional arguments that were passed in **kwargs
        """
        self.update_observations(observations=observations)
        observations = self.normalize_observations(observations=observations)
        if len(kwargs) > 0:
            return observations, *kwargs.values()
        else:
            return observations

    def update_observations(self, observations: InputBatch):
        # unpack
        if self.node_normalizers is not None:
            self.node_normalizers.update(observations.x)
        if self.edge_normalizers is not None:
            self.edge_normalizers.update(observations.edge_attr)

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        """
        Normalize observations using this instances current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        if self.node_normalizers is not None:
            observations.__setattr__("x", self._normalize_observation(observation=observations.x,
                                                                      normalizer=self.node_normalizers))
        if self.edge_normalizers is not None:
            observations.__setattr__("edge_attr", self._normalize_observation(observation=observations.edge_attr,
                                                                              normalizer=self.edge_normalizers))

        return observations

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
