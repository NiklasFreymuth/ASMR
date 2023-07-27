from abc import ABC

import gym
import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

from util.keys import AGENT
from util.types import *


class AbstractSwarmEnvironment(gym.Env, ABC):
    """
    AbstractSwarmEnvironment is an abstract gym environment designed to work on graph-based agents.
    Defines interface for pytorch geometric based Multi-Agent Reinforcement Learning algorithms while using numpy
    arrays for the internal computation of the environment.
    For each of these environments, agents can be seen as nodes in a graph and act in a collaborative fashion to
    minimize some shared reward. The number of agents for the environment can change between rollouts, or even between
    steps of the same rollout. However, all agents/nodes are assumed to have the same sized feature vector. Similarly,
    interactions between agents and global contexts also have a constant dimension.
    These dimension must be specified in the num_node_features, num_edge_features, num_global_features properties below.
    """

    def __init__(self, environment_config: ConfigDict, seed: Optional[int] = None):
        """
        Initialize the environment.
        Args:
            environment_config: (potentially nested) dictionary detailing the configuration of the environment.
            seed: Optional random seed to use for this environment.
        """
        super().__init__()
        self._seed = seed
        self._environment_config = environment_config

        # other
        self._random_state: np.random.RandomState = np.random.RandomState(seed=seed)

        # set graph sizes
        self._num_node_features: Dict[str, int] = {}
        self._num_edge_features: Dict[Tuple[str, str, str], int] = {}
        self._num_global_features: Optional[int] = None

    @property
    def num_node_features(self) -> Union[int, Dict[str, int]]:
        """
        Number of features per node. These can be anything, such as e.g., the velocity and angle of agents, embedded
        positions, a color/type of the agent, ...
        Returns: The features per agent. Can be None

        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement num_node_features")

    @property
    def num_edge_features(self) -> Union[int, Dict[Tuple[str, str, str], int]]:
        """
        Returns: The dimension of the interaction between agents.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement num_edge_features")

    @property
    def num_global_features(self) -> Optional[int]:
        """
        Returns: The dimension of the global vector of this environment. May describe something like boundary
          conditions or a global context.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement num_global_features")

    @property
    def num_agents(self) -> int:
        """
        Returns: The number of agents currently in the environment.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement num_global_features")

    def render(self, mode="human", *args, **kwargs) -> Union[np.array, Tuple[List[BaseTraceType], Dict[Key, Any]]]:
        """
        Renders the *current step* of the environment and returns it as a list of traces. Overlaying these traces
        in a plotly figure results in the full render for the rollout.
        Assumes that render() is called after the initial reset(), and after every step()-call of a rollout
        Args:
            mode: Either "human" or "rgb_array".
            *args:
            **kwargs:

        Returns: Either a numpy array of the current figure if mode=="rgb_array", or a tuple
           (traces, additional_render_information) otherwise
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement render")

    @property
    def agent_node_type(self) -> str:
        """
        Returns: The name of the node type that acts as an agent.
        """
        return AGENT

    @property
    def agent_action_space(self) -> gym.Space:
        """
        Returns: The gym.Space object that defines the action space for the agent.
        """
        return gym.spaces.Box(low=-np.infty, high=np.infty, shape=(self.action_dimension,), dtype=np.float32)

    @property
    def action_dimension(self) -> int:
        """
        Get the current action dimension.
        Returns:
            The action dimension.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement action_dimension")

    @property
    def is_truncated(self) -> bool:
        """
        Has the environment been truncated due to a time limit?
        Returns:
            True = The environment has reached a time limit.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement is_truncated")

    @property
    def is_terminal(self) -> bool:
        """
        Is the current episode over?
        Returns:
            True = The environment is done, i.e., reached a terminal state.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement is_truncated")

    @property
    def last_observation(self) -> Data:
        """
        Create the observation graph.

        Returns:
            The graph that was created.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement last_observation")

    def _set_graph_sizes(self):
        """
        This helper function is the central place for an environment to fill out the number of features for the graph.
        It should fill _num_node_features, _num_edge_features, _num_global_features. These are later used,
        when creating the observation graph.
        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement last_observation")

    ####################
    # additional plots #
    ####################
    def plot_value_per_agent(self, value_per_agent: np.array, title: str = "Value per Agent") -> go.Figure:
        """
        Plot/visualize a scalar value per agent.
        Args:
            value_per_agent: A (flat) numpy array of shape (num_agents,).
            title: Title of the plot

        Returns: A plotly figure.

        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement plot_value_per_agent")

    def additional_plots(self, iteration: int, policy_step_function: Optional[callable] = None) -> Dict[Key, go.Figure]:
        """
        Function that takes an algorithm iteration as input and returns a number of additional plots about the
        current environment as output. Some plots may be always selected, some only on e.g., iteration 0.
        Args:
            iteration: The current iteration of the algorithm.
            policy_step_function: (Optional)
                A function that takes a graph as input and returns the action(s) and (q)-value(s) for each agent.

        """
        raise NotImplementedError("AbstractSwarmEnvironment does not implement additional_plots")
