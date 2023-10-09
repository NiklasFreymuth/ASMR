from typing import Dict, Any, Union, Callable, Optional, Tuple

import numpy as np
import torch

from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement import MeshRefinement
from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_util import get_aggregation_per_element
from modules.swarm_environments.mesh.mesh_refinement.sweep.sweep_mesh_refinement_util import get_resource_budget, \
    get_neighbors_area, \
    get_edge_attributes, get_average_with_same_shape
from modules.swarm_environments.util import keys


class SweepMeshRefinement(MeshRefinement):
    """
    Environment Wrapper of MeshRefinement that adds features & functionalities relevant for the implementation of
    the paper Deep Reinforcement Learning for Adaptive Mesh Refinement (https://arxiv.org/pdf/2209.12351.pdf). The
    paper does not use message passing, such that relevant information needs to be added directly to the node i.e.
    element features.
    The following features are added for each element:
        - A Resource budget
        - The mean, min, max area of the neighbor elements
        - The overall average solution
        - Mean of all incoming element2element-edge features (which can me Euclidean distances & distance vectors)
    """

    def __init__(self, environment_config: Dict[Union[str, int], Any], seed: Optional[int] = None):
        self._previous_refined_elements = None
        self._num_evaluation_timesteps = environment_config.get("num_evaluation_timesteps")
        self._num_training_timesteps = environment_config.get("num_training_timesteps")
        self._buffered_graph_edges = None
        self._single_agent_mode = None
        self._current_agent_index = None

        super().__init__(environment_config, seed)

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self._buffered_graph_edges = None
        if self._single_agent_mode:
            self._current_agent_index = self._random_state.randint(1, self.num_agents - 1)

    def get_buffered_graph_edges(self) -> Dict[Union[str, Tuple[str, str, str]], torch.Tensor]:
        if self._buffered_graph_edges is None:
            self._buffered_graph_edges = self._get_graph_edges()
        return self._buffered_graph_edges

    def train(self, train: bool):
        """
        Set environment either in training or in evaluation mode. This is necessary since we use single agent training
        and multi-agent evaluation in the sweep baseline. Therefore, the number of time steps within one episode has
        to be significantly higher in training than in evaluation.
        """
        if train:
            self._max_timesteps = self._num_training_timesteps
            self._single_agent_mode = True
        else:
            self._max_timesteps = self._num_evaluation_timesteps
            self._single_agent_mode = False

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Overwrites the step function that it returns the reward such as in the baseline Deep Reinforcement Learning
        for Adaptive Mesh Refinement (https://arxiv.org/pdf/2209.12351.pdf).

        Args:
            action: The action that the agent took in the current state. In the case of single agent training, this is
            an array of length 1 with a single integer.
            In the case of multi-agent evaluation, this is a list of integers with length equal to the number of agents.
        """
        assert not self.is_terminal, f"Tried to perform a step on a terminated environment. Currently on step " \
                                     f"{self._timestep:} of {self._max_timesteps:} " \
                                     f"with {self._num_elements}/{self._maximum_elements} elements."

        if self._single_agent_mode:
            # in training mode, we return a single node that contains all the information, and likewise receive a single
            # action in return.
            # Set all actions except the action of the selected agent to zero to perform an environment step.
            actions = np.zeros(self.num_agents, dtype=np.int64)
            actions[self._current_agent_index] = action  # works due to broadcasting
        else:
            actions = action

        self._buffered_graph_edges = None
        self._timestep += 1
        self._set_previous_step()

        # refine mesh and store which element has become which set of new elements
        self._previous_refined_elements = self._get_elements_to_refine(actions)
        self._agent_mapping = self.fem_problem.refine_mesh(self._previous_refined_elements)

        # solve equation and calculate error per element/element
        self._previous_error_per_element = self.error_per_element
        self._error_estimation_dict = self.fem_problem.calculate_solution_and_get_error()

        # set new agent index if in single agent mode
        if self._single_agent_mode:
            self._current_agent_index = self._random_state.randint(1, self.num_agents - 1)

        # query returns
        observation = self.last_observation
        reward_dict = self._get_reward_dict()
        metric_dict = self._get_metric_dict()

        # done after a given number of steps or if the mesh becomes too large
        done = self.is_terminal
        info = reward_dict | metric_dict | {keys.IS_TRUNCATED: self.is_truncated,
                                            keys.RETURN: self._cumulative_return}
        return observation, self._reward, done, info

    @property
    def last_observation(self) -> torch.Tensor:
        """
        Retrieve an observation graph for the current state of the environment.

        We use an additional self.last_observation wrapper to make sure that classes that inherit from this
        one have access to node and edge features outside the Data() structure
        Returns: A HeteroData() object of the graph that describes the current state of this environment

        """
        assert self._single_agent_mode is not None, "Environment has not been set to training or evaluation mode yet."
        graph_nodes = self._get_graph_nodes()["x"]

        return graph_nodes[self._current_agent_index].reshape(1, -1) if self._single_agent_mode else graph_nodes

    def _get_reward_by_type(self) -> Tuple[float, Dict]:
        """
        Calculate the reward for the current timestep depending on the environment states and the action
        the agents took.
        Args:

        Returns: A tuple of the element limit penalty, the element penalty, the reward and a dictionary of partial rewards.
        """

        if self._reward_type == "sweep":
            if self.num_agents > self._maximum_elements:
                reward = -np.array(self._element_limit_penalty)
                return reward, {keys.REWARD: reward,
                                keys.APPROXIMATION_GAIN: 0,
                                keys.ELEMENT_PENALTY: 0,
                                keys.ELEMENT_LIMIT_PENALTY: -np.array(self._element_limit_penalty)}
            else:
                return self._get_sweep_reward()
        else:
            return super()._get_reward_by_type()

    def _get_sweep_reward(self):
        # Store remaining error, number of agents and elements that are refined for later on reward calculation.
        previous_remaining_error_per_dimension = self._previous_error_per_element.sum(axis=0)
        previous_num_agents = self._previous_num_elements

        # Calculate reward as log ratio of change in mesh minus barrier penalty for the use of a resource budget.
        remaining_error_per_dimension = np.sum(self.error_per_element, axis=0)
        approximation_gain_per_dimension = previous_remaining_error_per_dimension - remaining_error_per_dimension
        # gain is equivalent to error reduction

        # We sometimes *increase* the error, so for the log scaling to work we have to adapt the reward by clipping
        # to zero
        scaled_approximation_gain_per_dim = np.log(np.maximum(0, approximation_gain_per_dimension) + 1.0e-12) - \
                                            np.log(1.0e-12)
        scaled_approximation_gain = self.project_to_scalar(scaled_approximation_gain_per_dim)

        previous_budget_use = np.minimum(previous_num_agents, self._maximum_elements - 1) / self._maximum_elements
        budget_use = np.minimum(self.num_agents, self._maximum_elements - 1) / self._maximum_elements
        # budget is the maximum number of elements.

        barrier_previous_budget_use = np.sqrt(previous_budget_use) / (1 - previous_budget_use)
        barrier_budget_use = np.sqrt(budget_use) / (1 - budget_use)
        barrier_budget_penalty_coefficient = barrier_budget_use - barrier_previous_budget_use
        barrier_budget_penalty = self._element_penalty_lambda * barrier_budget_penalty_coefficient

        reward = scaled_approximation_gain - barrier_budget_penalty
        return reward, {keys.REWARD: reward,
                        keys.APPROXIMATION_GAIN: scaled_approximation_gain,
                        keys.ELEMENT_PENALTY: barrier_budget_penalty,
                        keys.ELEMENT_LIMIT_PENALTY: 0}

    def _register_element_features(self) -> Dict[str, Callable[[], np.array]]:
        """
        Returns a dictionary of functions that return the features of the elements. We return a dictionary of functions
        instead of a dictionary of values to allow for lazy evaluation of the features (e.g. if the features are
        expensive to compute or change between iterations, we always want to compute them when requested to).
        Returns:
        """
        element_feature_config = self._environment_config.get("element_features")
        element_feature_names = [feature_name
                                 for feature_name, include_feature in element_feature_config.items()
                                 if include_feature]
        element_features = {}

        # Additional element features
        if "resource_budget" in element_feature_names:
            # Get resource budget (as in sweep paper).
            element_features["resource_budget"] = lambda: get_resource_budget(num_agents=self.num_agents,
                                                                              num_max_agents=self._maximum_elements)

        if "average_error" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_error"] = lambda i_=position: get_average_with_same_shape(
                    self.error_per_element[:, i_])

        if "average_solution" in element_feature_names:
            for position, name in enumerate(self._solution_dimension_names):
                element_features[f"{name}_solution"] = lambda i_=position: get_average_with_same_shape(
                    input_array=get_aggregation_per_element(self.solution[:, i_],
                                                            self._element_indices,
                                                            aggregation_function_str="mean"))

        if "mean_area_neighbors" in element_feature_names:
            # Get mean of minimum area of neighbor elements.
            element_features["mean_area_neighbors"] = lambda: \
                get_neighbors_area(
                    self.get_buffered_graph_edges().get("edge_index"),
                    self.element_areas, aggregation="mean")

        if "mean_edge_attributes" in element_feature_names:
            # compute dimensionality of edge features (used later on)
            edge_feature_config = self._environment_config.get("edge_features")
            edge_feature_names = [feature_name for feature_name, include_feature in edge_feature_config.items()
                                  if include_feature]
            edge_feature_dimensions = 0
            if "distance_vector" in edge_feature_names:
                edge_feature_dimensions += 2
            if "euclidean_distance" in edge_feature_names:
                edge_feature_dimensions += 1
            # Get mean of all incoming edge attributes for each element.
            for i in range(edge_feature_dimensions):
                element_features[f"mean_edge_attributes_{i}"] = lambda i_=i: \
                    get_edge_attributes(self.get_buffered_graph_edges(),
                                        feature_position=i_, aggregation="mean")

        return element_features | super()._register_element_features()
