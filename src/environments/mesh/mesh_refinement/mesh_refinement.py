import gym
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.basedatatypes import BaseTraceType
from skfem import Mesh

import util.keys as Keys
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.environments.mesh.mesh_refinement.mesh_refinement_util import get_line_segment_distances, \
    get_aggregation_per_element
from src.environments.mesh.mesh_refinement.mesh_refinement_visualization import get_mesh_traces, \
    contour_trace_from_element_values
from src.environments.mesh.mesh_refinement.problems.fem_buffer import FEMBuffer
from util.function import get_triangle_areas_from_indices, save_concatenate
from util.types import *
from util.visualization import get_layout


class MeshRefinement(AbstractSwarmEnvironment):
    """
    Gym environment of a graph-based 2d mesh refinement task using scikit-FEM as a backend.
    Given some underlying finite element task and a coarse initial mesh, the agents must in each step select a number
     of elements of the mesh that will be refined.
     Optionally, a smoothing operator is called after the refinement.
    The goal of the task is to automatically create meshes of adaptive resolution that focus on the "interesting"
    areas of the task with only as many resources as necessary. For this, the mesh is used for a simulation in every
    timestep, and the result of this simulation is compared to some ground truth. The simulation error is then
    combined with a cost for the size of the mesh to produce a reward.
    Intuitively, the task is to highlight important/interesting nodes of the graph, such that the subsequent remeshing
    algorithm assigns more resources to this area than to the other areas. This is iterated over, giving the agent
    the option to iteratively refine the remeshing.
    """

    def __init__(self, environment_config: ConfigDict, seed: Optional[int] = None):
        """
        Args:
            environment_config: Config for the environment.
                Details can be found in the configs/references/mesh_refinement_reference.yaml example file
            seed: Optional seed for the random number generator.
        """
        super().__init__(environment_config=environment_config, seed=seed)

        self.fem_problem = FEMBuffer(fem_config=environment_config.get("fem"),
                                     random_state=np.random.RandomState(seed=seed))

        ################################################
        #        general environment parameters        #
        ################################################
        self._refinement_strategy: str = environment_config.get("refinement_strategy")
        self._max_timesteps = environment_config.get("num_timesteps")
        self._element_limit_penalty = environment_config.get("element_limit_penalty")
        self._maximum_elements = environment_config.get("maximum_elements", 20000)

        ################################################
        # graph connectivity, feature and action space #
        ################################################
        self._reward_type = environment_config.get("reward_type")
        self._include_globals = environment_config.get("include_globals")
        self._set_graph_sizes()

        ################################################
        #          internal state and cache            #
        ################################################
        self._timestep: int = 0
        self._element_penalty_lambda = self._environment_config.get("element_penalty")
        self._initial_approximation_errors: Optional[Dict[Key, float]] = None
        self._reward = None
        self._cumulative_return: np.array = 0  # return of the environment
        # dictionary containing the error estimation for the current solution for different error evaluation metrics
        self._error_estimation_dict: Optional[Dict[str, np.array]] = None
        self._initial_error_norm = None

        # last-step history for delta-based rewards and plotting
        self._previous_error_per_element: Optional[np.array] = None
        self._previous_num_elements: Optional[int] = None
        self._previous_element_mapping = None
        self._previous_element_areas = None
        self._previous_std_per_element = None

        # fields/internal variables for spatial mesh refinement, especially a spatial reward
        self._element_mapping = None  # mapping List[old_element_indices] of size new_element_indices that maps
        self._reward_per_agent: Optional[np.array] = 0  # cumulative return of the environment per agent
        self._cumulative_reward_per_agent: Optional[np.array] = 0  # cumulative reward of the environment per agent

        ################################################
        #            recording and plotting            #
        ################################################
        self._initial_num_elements = None

    def _set_graph_sizes(self):
        """
        Internally sets the
        * action dimension
        * number of node types and node features for each type
        * number of edge types and edge features for each type
        * number of global features (if used)
        depending on the configuration. Uses the same edge features for all edge types.
        Returns:

        """
        # set number of edge features
        self._num_edge_features = 1

        # element features
        self._element_feature_functions = self._register_element_features()
        self._num_node_features = len(self._element_feature_functions)
        # in addition to the features above, we may include fem-problem specific features for the elements.
        # these are specified in the fem_problem:$Problem:element_features config and registered in the fem_problem.
        self._num_node_features += self.fem_problem.num_pde_element_features

        # global features
        if self._include_globals:
            self._global_feature_functions = self._register_global_features()
            self._num_global_features = len(self._global_feature_functions)
            self._num_global_features += self.fem_problem.num_pde_global_features
        else:
            self._global_feature_functions = None
            self._num_global_features = None

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

        if "x_position" in element_feature_names:
            element_features["x_position"] = lambda: self.element_midpoints[:, 0]
        if "y_position" in element_feature_names:
            element_features["y_position"] = lambda: self.element_midpoints[:, 1]
        if "area" in element_feature_names:
            element_features["area"] = lambda: self.element_areas
        if "solution_std" in element_feature_names:
            for position, name in enumerate(self.solution_dimension_names):
                element_features[f"{name}_solution_std"] = lambda i_=position: self.solution_std_per_element[:, i_]
        if "solution_mean" in element_feature_names:
            for position, name in enumerate(self.solution_dimension_names):
                element_features[f"{name}_solution_mean"] = lambda i_=position: get_aggregation_per_element(
                    self.solution[:, i_], self.element_indices, aggregation_function_str="mean")
        if "distance_to_boundary" in element_feature_names:
            fn = lambda: get_line_segment_distances(points=self.element_midpoints,
                                                    projection_segments=self.fem_problem.boundary_line_segments,
                                                    return_minimum=True)
            element_features["distance_to_boundary"] = fn
        return element_features

    def _register_global_features(self) -> Dict[str, Callable[[], np.array]]:
        """
        Returns a dictionary of functions that return the global features.
        Returns:

        """
        global_feature_config = self._environment_config.get("global_features")
        global_feature_names = [feature_name
                                for feature_name, include_feature in global_feature_config.items()
                                if include_feature]
        global_features = {}
        if "num_vertices" in global_feature_names:
            global_features["num_vertices"] = lambda: self.num_vertices
        if "num_elements" in global_feature_names:
            global_features["num_elements"] = lambda: self.num_elements
        if "timestep" in global_feature_names:
            global_features["timestep"] = lambda: self._timestep
        return global_features

    def reset(self) -> Data:
        """
        Resets the environment and returns an (initial) observation of the next rollout
        according to the reset environment state

        Returns:
            The observation of the initial state.
        """
        # get the next fem problem. This samples a new domain and new load function, resets the mesh and the solution.
        self.fem_problem.next()

        # calculate the solution of the finite element problem for the initial mesh and retrieve an error per element
        self._error_estimation_dict = self.fem_problem.calculate_solution_and_get_error()

        # reset the internal state of the environment. This includes the current timestep, the current element penalty
        # and some values for calculating the reward and plotting the env
        self._reset_internal_state()

        observation = self.last_observation
        return observation

    def _reset_internal_state(self):
        """
        Resets the internal state of the environment
        Returns:

        """
        self._element_mapping = np.arange(self.num_elements).astype(np.int64)  # map to identity at first step
        self._previous_element_mapping = np.arange(self.num_elements).astype(np.int64)  # map to identity at first step
        self._previous_element_areas = self.element_areas
        self._previous_std_per_element = self.solution_std_per_element
        self._reward_per_agent = np.zeros(self.num_agents)
        self._cumulative_reward_per_agent = np.zeros(self.num_elements)

        # reset timestep and rewards
        self._timestep = 0
        self._reward = 0
        self._cumulative_return = 0

        # reset internal state that tracks statistics over the episode
        self._previous_error_per_element = self.error_per_element

        # collect a dictionary of initial errors to normalize them when calculating metrics during evaluation
        self._initial_approximation_errors = {}
        for error_name, element_errors in self.error_estimation_dict.items():
            if "maximum" in error_name:
                self._initial_approximation_errors[error_name] = np.max(element_errors, axis=0) + 1.0e-12
            else:
                self._initial_approximation_errors[error_name] = np.sum(element_errors, axis=0) + 1.0e-12

        self._previous_num_elements = self.num_elements
        self._initial_num_elements = self.num_elements
        self._initial_error_norm = np.linalg.norm(self.error_per_element, axis=0)

    def step(self, action: np.ndarray) -> Tuple[Data, np.array, bool, Dict[str, Any]]:
        """
        Performs a step of the Mesh Refinement task
        Args:
            action: the action the agents will take in this step. Has shape (num_agents, action_dimension)
            Given as an array of shape (num_agents, action_dimension)

        Returns: A 4-tuple (observations, reward, done, info), where
            * observations is a graph of the agents and their positions, in this case of the refined mesh
            * reward is a single scalar shared between all agents, i.e., per **graph**
            * done is a boolean flag that says whether the current rollout is finished or not
            * info is a dictionary containing additional information
        """
        assert not self.is_terminal, f"Tried to perform a step on a terminated environment. Currently on step " \
                                     f"{self._timestep:} of {self._max_timesteps:} " \
                                     f"with {self.num_elements}/{self._maximum_elements} elements."

        self._timestep += 1

        self._set_previous_step()

        self._element_mapping = self._refine_mesh(action=action)  # refine mesh and store which element has become which
        # set of new elements

        # solve equation and calculate error per element/element
        self._previous_error_per_element = self.error_per_element

        self._error_estimation_dict = self.fem_problem.calculate_solution_and_get_error()

        # query returns
        observation = self.last_observation

        reward_dict = self._get_reward_dict()
        metric_dict = self._get_metric_dict()

        # done after a given number of steps or if the mesh becomes too large
        done = self.is_terminal
        info = reward_dict | metric_dict | {Keys.IS_TRUNCATED: self.is_truncated,
                                            Keys.RETURN: self._cumulative_return}
        return observation, self._reward, done, info

    def inference_step(self, action: np.ndarray) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        Performs a step of the Mesh Refinement task *without* calculating the reward or difference to the fine-grained
        reference. This is used for inference

        Args:
            action: the action the agents will take in this step. Has shape (num_agents, action_dimension)
            Given as an array of shape (num_agents, action_dimension)

        Returns: A 4-tuple (observations, reward, done, info), where
            * observations is a graph of the agents and their positions, in this case of the refined mesh
            * reward is a single scalar shared between all agents, i.e., per **graph**
            * done is a boolean flag that says whether the current rollout is finished or not
            * info is a dictionary containing additional information
        """
        assert not self.is_terminal, f"Tried to perform a step on a terminated environment. Currently on step " \
                                     f"{self._timestep:} of {self._max_timesteps:} " \
                                     f"with {self.num_elements}/{self._maximum_elements} elements."

        self._timestep += 1
        self._element_mapping = self._refine_mesh(action=action)
        # solve equation
        self.fem_problem.calculate_solution()
        observation = self.last_observation
        done = self.is_terminal
        info = {}
        return observation, self._reward, done, info

    def _set_previous_step(self):
        """
        Sets variables for the previous timestep. These are used for the reward function, as well as for different
        kinds of plots and metrics
        """
        # refine mesh based on selected faces/elements and re-build basis
        self._previous_num_elements = self.num_elements
        self._previous_element_mapping = self._element_mapping
        self._previous_element_areas = self.element_areas
        self._previous_std_per_element = self.solution_std_per_element

    def _refine_mesh(self, action: np.array) -> np.array:
        """
        Refines fem_problem.mesh by splitting all faces/elements for which the average of agent activation surpasses a
        threshold.
        If this refinement exceeds the maximum number of nodes allowed in the environment, we return a boolean flag
        that indicates so and stops the environment

        Optionally smoothens the newly created mesh as a post-processing step

        Args:
            action: An action/activation per element.
                Currently, a scalar value that is interpreted as a refinement threshold.
        Returns: An array of mapped element indices

        """
        elements_to_refine = self._get_elements_to_refine(action)

        # updates self.fem_problem.mesh
        element_mapping = self.fem_problem.refine_mesh(elements_to_refine)
        return element_mapping

    def _get_elements_to_refine(self, action: np.array) -> np.array:
        """
        Calculate which elements to refine based on the action, refinement strategy and the
        maximum number of elements allowed in the environment
        Args:
            action: An action/activation per agent, i.e., per element or element

        Returns: An array of ids corresponding to elements_to_refine

        """
        action = action.flatten()  # make sure that action is 1d, i.e., decides on a refinement threshold per agent
        # select elements to refine based on the average actions of its surrounding agents/nodes

        if self._refinement_strategy == "discrete":
            # refine elements w/ action > 0
            elements_to_refine = np.argwhere(action > 0.0).flatten()
        elif self._refinement_strategy in ["argmax", "best", "max"]:
            # refine elements w/ action > 0
            if action.size == 1 and isinstance(action.item(), int):
                elements_to_refine = action
            else:
                elements_to_refine = np.argmax(action).flatten()
            if self.refinements_per_element[elements_to_refine] > 30:
                # if we have refined this element too often, we skip this action
                elements_to_refine = np.array([])
        else:
            raise ValueError(f"Unknown refinement strategy '{self._refinement_strategy}")
        return elements_to_refine

    def render(self, mode: str = "human", render_intermediate_steps: bool = False,
               *args, **kwargs) -> Union[np.array, Tuple[List[BaseTraceType], Dict[Key, Any]]]:
        """
        Renders and returns a list of plotly traces  for the *current state* of this environment.

        Args:
            mode: How to render the figure. Must either be "human" or "rgb_array"
            render_intermediate_steps: Whether to render steps that are non-terminal or not. Defaults to False, as
              we usually want to visualize the result of a full rollout
            *args: additional arguments (not used).
            **kwargs: additional keyword arguments (not used).

        Returns: Either a numpy array of the current figure if mode=="rgb_array", or a tuple
           (traces, additional_render_information) otherwise.
        """
        if render_intermediate_steps or self.is_terminal:
            # only return traces if told to do so or if at the last step to avoid overlay of multiple steps

            traces = self.fem_problem.approximated_weighted_solution_traces()

            remaining_error = self._get_remaining_error(return_dimensions=False)
            title = f"Interpolated Solution. " \
                    f"Reward: {np.sum(self._reward):.3f} " \
                    f"Return: {np.sum(self._cumulative_return):.3f} " \
                    f"Agents: {self.num_agents} " \
                    f"Remaining Error: {remaining_error:.3f}"
            layout = get_layout(boundary=self.fem_problem.plot_boundary, title=title)
            return traces, {"layout": layout}
        else:
            return [], {}

    def _get_remaining_error(self, return_dimensions: bool = False):
        """
        Get the remaining error by aggregating over all elements and taking the convex sum of all solution dimensions
        """
        if "maximum" in self.error_metric:
            remaining_error_per_dimension = np.max(self.error_per_element, axis=0)
        else:
            remaining_error_per_dimension = np.sum(self.error_per_element, axis=0)
        remaining_error_per_dimension = remaining_error_per_dimension / self.initial_approximation_error  # normalize
        remaining_error = np.dot(remaining_error_per_dimension, self.solution_dimension_weights)

        if return_dimensions:
            return remaining_error, remaining_error_per_dimension
        else:
            return remaining_error

    @property
    def last_observation(self) -> Data:
        """
        Retrieve an observation graph for the current state of the environment.

        We use an additional self.last_observation wrapper to make sure that classes that inherit from this
        one have access to node and edge features outside the Data() structure
        Returns: A Data() object of the graph that describes the current state of this environment

        """
        graph_dict = {}
        graph_dict = graph_dict | self._get_graph_nodes()
        graph_dict = graph_dict | self._get_graph_edges()
        if self._include_globals:
            graph_dict = graph_dict | self._get_graph_globals()

        return Data(x=graph_dict["x"],
                    edge_index=graph_dict["edge_index"],
                    edge_attr=graph_dict["edge_attr"],
                    u=graph_dict["u"])

    def _get_graph_nodes(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of node features that are used to describe the current state of this environment.

        Returns: A dictionary of node features. This dictionary has the format
        {
        [Optional] Keys.ELEMENT: {"x": element_features}
        },
        where element and node features depend on the context, but include things like the evaluation of the target
        function, the degree of the node, etc.
        """

        # Builds feature matrix of shape (#elements, #features)
        # by iterating over the functions in self._element_feature_functions.
        element_features = np.array([fn() for key, fn in self._element_feature_functions.items()]).T
        element_features = save_concatenate([element_features, self.fem_problem.element_features()], axis=1)
        node_dict = {"x": torch.tensor(element_features, dtype=torch.float32)}
        return node_dict

    def _get_graph_edges(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of edge features that are used to describe the current state of this environment.
        Note that we always use symmetric graphs and self edges

        Returns: A dictionary of edge features. This dictionary has the format
        {
        RelationName1: {"edge_index": indices, "edge_attr": features},
        RelationName2: {"edge_index": indices, "edge_attr": features},
        ...
        }
        """
        # concatenate incoming, outgoing and self edges of each node to get an undirected graph
        src_nodes = np.concatenate((self.element_edges[0], self.element_edges[1], np.arange(self.num_elements)),
                                   axis=0)
        dest_nodes = np.concatenate((self.element_edges[1], self.element_edges[0], np.arange(self.num_elements)),
                                    axis=0)

        num_edges = self.element_edges.shape[1] * 2 + self.num_elements
        edge_features = np.empty(shape=(num_edges, self._num_edge_features))
        euclidean_distances = np.linalg.norm(self.element_midpoints[dest_nodes] -
                                             self.element_midpoints[src_nodes], axis=1)
        edge_features[:, 0] = euclidean_distances
        edge_dict = {
            "edge_index": torch.tensor(np.vstack((src_nodes, dest_nodes))).long(),
            "edge_attr": torch.tensor(edge_features, dtype=torch.float32),
        }
        return edge_dict

    def _get_graph_globals(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of global features that are used to describe the current state of this environment.
        Returns: A dictionary of global features. Contains a key "u" that contains the global features as a torch.Tensor
        of shape (1, #global_features)
        """
        global_features = np.array([fn() for key, fn in self._global_feature_functions.items()]).T
        global_features = save_concatenate([global_features, self.fem_problem.global_features()], axis=0)
        return {"u": torch.tensor(global_features, dtype=torch.float32).reshape(1, -1)}

    def _get_reward_dict(self) -> Dict[Key, np.float32]:
        """
        Calculate the reward for the current timestep depending on the environment states and the action
        the agents took.
        Args:

        Returns:
            Dictionary that must contain "reward" as well as partial reward data

        """
        reward, reward_dict = self._get_reward_by_type()

        self._reward = reward
        self._cumulative_return = self._cumulative_return + np.sum(self._reward)

        return reward_dict

    def _get_metric_dict(self) -> ValueDict:
        remaining_error, remaining_error_per_dimension = self._get_remaining_error(return_dimensions=True)

        metric_dict = {Keys.REMAINING_ERROR: remaining_error,
                       Keys.ERROR_TIMES_AGENTS: remaining_error * self.num_agents,
                       Keys.DELTA_ELEMENTS: self.num_elements - self._previous_num_elements,
                       Keys.AVG_TOTAL_REFINEMENTS: np.log(self.num_elements / self._initial_num_elements) / np.log(4),
                       Keys.AVG_STEP_REFINEMENTS: np.log(self.num_elements / self._previous_num_elements) / np.log(4),
                       Keys.NUM_AGENTS: self.num_agents,
                       Keys.REACHED_ELEMENT_LIMITS: self.reached_element_limits,
                       Keys.REFINEMENT_STD: self.refinements_per_element.std()}
        if 1 < len(self.solution_dimension_names) < 5:
            # provide individual metrics for the different solution dimensions
            for position, name in enumerate(self.solution_dimension_names):
                metric_dict[f"{name}_remaining_error"] = remaining_error_per_dimension[position]
        elif len(self.solution_dimension_names) >= 5:
            # provide a single metric that is the weighted sum of the different solution dimensions
            last_dimension = self.solution_dimension_names[-1]
            metric_dict[f"{last_dimension}_remaining_error"] = remaining_error_per_dimension[-1]

        for error_metric, element_errors in self.error_estimation_dict.items():
            if element_errors.shape[0] == self.num_elements:
                # need to aggregate
                if "maximum" in error_metric:
                    error_per_dimension = np.max(element_errors, axis=0)
                else:
                    error_per_dimension = np.sum(element_errors, axis=0)

            else:
                error_per_dimension = element_errors
            error_per_dimension = error_per_dimension / self._initial_approximation_errors[error_metric]
            remaining_error = np.dot(error_per_dimension, self.solution_dimension_weights)
            metric_dict[f"{error_metric}_error"] = remaining_error
        return metric_dict

    def _get_reward_by_type(self) -> Tuple[np.array, Dict]:
        """
        Calculate the reward for the current timestep depending on the environment states and the action
        the agents took.

        Args:

        Returns: A Tuple (reward, dictionary of partial rewards).

        """

        reward_dict = {}

        if self._reward_type in ["spatial", "spatial_area", "spatial_max"]:
            reward_per_agent_and_dim = np.copy(self._previous_error_per_element)

            if self._reward_type == "spatial_max":
                # calculate the difference between an elements error and the maximum error of its childs.
                # Only makes sense if we have the "maximum" as an error metric per element.
                assert self.error_metric == "maximum", f"Must use maximum error metric for spatial_max reward type" \
                                                       f"but got {self.error_metric}"
                from torch_scatter import scatter_max
                agent_mapping = torch.tensor(self.agent_mapping)
                error_per_element = torch.tensor(self.error_per_element)
                max_mapped_error, _ = scatter_max(src=error_per_element, index=agent_mapping,
                                                  dim=0, dim_size=reward_per_agent_and_dim.shape[0])
                max_mapped_error = max_mapped_error.numpy()
                reward_per_agent_and_dim = reward_per_agent_and_dim - max_mapped_error

            elif self._reward_type == "spatial":
                np.add.at(reward_per_agent_and_dim, self.agent_mapping, -self.error_per_element)  # scatter add

            elif self._reward_type == "spatial_area":
                # normalize reward per element by its inverse area. I.e., give more weight to smaller areas
                np.add.at(reward_per_agent_and_dim, self.agent_mapping, -self.error_per_element)  # scatter add
                reward_per_agent_and_dim = reward_per_agent_and_dim / self._previous_element_areas[:, None]

            else:
                raise ValueError(f"Unknown reward type {self._reward_type}")

            reward_per_agent_and_dim = reward_per_agent_and_dim / self.initial_approximation_error  # normalize to 1

            reward_per_agent = np.dot(reward_per_agent_and_dim, self.solution_dimension_weights)

            # the element penalty per agent depends on how many new elements are created by this agent
            element_counts = np.unique(self.agent_mapping, return_counts=True)[1]
            element_counts = element_counts - 1  # -1 because of the original element
            element_penalty = self._element_penalty_lambda * element_counts
            element_limit_penalty = (self._element_limit_penalty / self._previous_num_elements) \
                if self.reached_element_limits else 0

            reward_per_agent = reward_per_agent - element_penalty - element_limit_penalty

            self._reward_per_agent = reward_per_agent
            self._cumulative_reward_per_agent = self._cumulative_reward_per_agent[self.previous_agent_mapping] \
                                                + reward_per_agent
            reward = reward_per_agent

        elif self._reward_type == "vdgn":
            # this uses the multi-objective VDGN reward function of https://arxiv.org/pdf/2211.00801v3.pdf,
            # which is a log difference of the error per dimension. The element penalty is defined as
            # (d_t-1 - d_t)/d_thresh*w, which equals
            # self._element_penalty_lambda * (self.num_elements - self._previous_num_elements) due to linearity
            previous_errors = np.linalg.norm(self._previous_error_per_element, axis=0)
            current_errors = np.linalg.norm(self.error_per_element, axis=0)
            reward_per_dimension = np.log(previous_errors + 1.0e-12) - np.log(current_errors + 1.0e-12)
            # 1.0e-12 for stability reasons, as the error may become 0 for coarse reference solutions

            reward = np.dot(reward_per_dimension, self.solution_dimension_weights)

            element_penalty = self._element_penalty_lambda * (self.num_elements - self._previous_num_elements)
            element_limit_penalty = self._element_limit_penalty if self.reached_element_limits else 0

            reward = reward - element_penalty - element_limit_penalty

        elif self._reward_type == "argmax":
            previous_errors = np.linalg.norm(self._previous_error_per_element, axis=0)
            current_errors = np.linalg.norm(self.error_per_element, axis=0)
            reward_per_dimension = (previous_errors - current_errors) / self._initial_error_norm
            reward = np.dot(reward_per_dimension, self.solution_dimension_weights)
            element_penalty = 0
            element_limit_penalty = 0
        else:
            raise ValueError(f"Unknown reward type '{self._reward_type}'")

        reward_dict[Keys.REWARD] = reward
        reward_dict[Keys.PENALTY] = -reward
        reward_dict[Keys.ELEMENT_LIMIT_PENALTY] = element_limit_penalty
        reward_dict[Keys.ELEMENT_PENALTY] = element_penalty
        return reward, reward_dict

    @property
    def mesh(self) -> Mesh:
        """
        Returns the current mesh.
        """
        return self.fem_problem.mesh

    @mesh.setter
    def mesh(self, mesh: Mesh):
        self.fem_problem.mesh = mesh

    @property
    def agent_node_type(self) -> str:
        return Keys.ELEMENT

    @property
    def vertex_edges(self) -> np.array:
        return self.mesh.facets

    @property
    def vertex_positions(self) -> np.array:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)

        """
        return self.fem_problem.vertex_positions

    @property
    def element_indices(self) -> np.array:
        return self.mesh.t.T

    @property
    def element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements/faces.
        Returns: np.array of shape (num_elements, 2)

        """
        return self.fem_problem.element_midpoints

    @property
    def element_edges(self) -> np.array:
        # f2t are element/face neighborhoods, which are set to -1 for boundaries
        return self.mesh.f2t[:, self.mesh.f2t[1] != -1]

    @property
    def element_areas(self) -> np.array:
        return get_triangle_areas_from_indices(positions=self.vertex_positions, triangle_indices=self.element_indices)

    @property
    def num_elements(self) -> int:
        return self.mesh.t.shape[1]

    @property
    def num_vertices(self) -> int:
        return self.mesh.p.shape[1]

    @property
    def num_node_features(self) -> Union[int, Dict[str, int]]:
        return self._num_node_features

    @property
    def num_edge_features(self) -> Union[int, Dict[Tuple[str, str, str], int]]:
        return self._num_edge_features

    @property
    def num_global_features(self) -> Optional[int]:
        return self._num_global_features

    @property
    def action_dimension(self) -> int:
        """
        Returns: The dimensionality of the action space. Note that for discrete action spaces
        (i.e., refinement_strategy == "absolute_discrete")  this is the number of possible actions,
        which are all represented in a single scalar

        """
        if self._refinement_strategy == "discrete":
            return 2
        else:  # single continuous value
            return 1

    @property
    def num_agents(self) -> int:
        return self.num_elements

    @property
    def agent_action_space(self) -> gym.Space:
        """

        Returns:

        """
        if self._refinement_strategy == "discrete":
            return gym.spaces.Discrete(self.action_dimension)
        else:
            return gym.spaces.Box(low=-np.infty, high=np.infty, shape=(self.action_dimension,), dtype=np.float32)

    @property
    def agent_mapping(self) -> np.array:
        assert self._element_mapping is not None, "Element mapping not initialized"
        return self._element_mapping

    @property
    def previous_agent_mapping(self) -> np.array:
        assert self._previous_element_mapping is not None, "Previous element mapping not initialized"
        return self._previous_element_mapping

    @property
    def reached_element_limits(self) -> bool:
        """
        True if the number of elements/faces in the mesh is above the maximum allowed value.
        Returns:

        """
        return self.num_elements > self._maximum_elements

    @property
    def is_truncated(self) -> bool:
        return self._timestep >= self._max_timesteps

    @property
    def is_terminal(self) -> bool:
        return self.reached_element_limits or self.is_truncated

    @property
    def solution(self) -> np.array:
        """
        Returns: solution vector per *vertex* of the mesh.
            An array (num_vertices, solution_dimension),
            where every entry corresponds to the solution of the parameterized fem_problem
            equation at the position of the respective node/vertex.

        """
        return self.fem_problem.solution

    @property
    def solution_dimension_names(self) -> List[str]:
        return self.fem_problem.solution_dimension_names

    @property
    def solution_dimension_weights(self) -> List[str]:
        return self.fem_problem.solution_dimension_weights

    @property
    def error_metric(self) -> str:
        """
        'Main' error estimation method used for the algorithm.
        This is the method that is used for the reward calculation.
        Other methods can be used for debugging and logging purposes.
        Returns:

        """
        return self.fem_problem.error_metric

    @property
    def error_per_element(self) -> np.array:
        """
        Returns: error per element of the mesh. np.array of shape (num_elements, solution_dimension)

        """
        return self._error_estimation_dict.get(self.error_metric)

    @property
    def initial_approximation_error(self) -> np.array:
        """
        Returns: error per element of the mesh. np.array of shape (num_elements, solution_dimension)

        """
        return self._initial_approximation_errors.get(self.error_metric)

    @property
    def error_estimation_dict(self) -> Dict[str, np.array]:
        """
        Returns a dictionary of all error estimation methods and their respective errors.
        These errors may be per element/face, or per integration point, depending on the metric.
        Returns:

        """
        return self._error_estimation_dict

    @property
    def reward_type(self) -> str:
        return self._reward_type

    @property
    def refinements_per_element(self) -> np.array:
        return self.fem_problem.refinements_per_element

    @property
    def solution_std_per_element(self) -> np.array:
        """
        Computes the standard deviation of the solution per element.
        Returns: np.array of shape (num_elements, solution_dimension)

        """
        return get_aggregation_per_element(self.solution, self.element_indices, aggregation_function_str="std")

    ####################
    # additional plots #
    ####################

    def plot_value_per_agent(self, value_per_agent: np.array, title: str = "Value per Agent",
                             mesh: Optional[Mesh] = None) -> go.Figure:
        """
        Plot/visualize a scalar value per agent.
        Args:
            value_per_agent: A numpy array of shape (num_agents,).
            title: The title of the plot.
            mesh: The mesh to be used for the plot. If None, the mesh of the environment is used.

        Returns: A plotly figure.

        """
        if mesh is None:
            assert len(value_per_agent) == self.num_agents, f"Need to provide a value per agent, given " \
                                                            f"'{value_per_agent.shape}' and '{self.num_agents}'"
            mesh = self.mesh

        return self._plot_value_per_element(value_per_element=value_per_agent, title=title, mesh=mesh)

    def _plot_value_per_element(self, value_per_element: np.array, title: str, normalize_by_element_area: bool = False,
                                mesh: Optional[Mesh] = None) -> go.Figure:
        """
        only return traces if asked or at the last step to avoid overlay of multiple steps
        Args:
            value_per_element: A numpy array of shape (num_elements,).
            title: The title of the plot.
            normalize_by_element_area: If True, the values are normalized by the element area as value /= element_area.
            mesh: The mesh to plot the values on. If None, the mesh of the current state is used.

        Returns: A plotly figure with an outline of the mesh and value per element in the element midpoints.

        """
        if mesh is None:
            mesh = self.mesh
        if normalize_by_element_area:
            value_per_element = value_per_element / self.element_areas
        element_midpoint_trace = contour_trace_from_element_values(mesh=mesh,
                                                                   element_evaluations=value_per_element.flatten(),
                                                                   trace_name=title)
        mesh_trace = get_mesh_traces(mesh)
        traces = element_midpoint_trace + mesh_trace
        layout = get_layout(boundary=self.fem_problem.plot_boundary,
                            title=title)
        value_per_element_plot = go.Figure(data=traces,
                                           layout=layout)
        return value_per_element_plot

    def plot_cumulative_reward_per_agent(self) -> go.Figure:
        return self.plot_value_per_agent(value_per_agent=self._cumulative_reward_per_agent,
                                         title="Cumulative Reward",
                                         mesh=self.fem_problem.previous_mesh)

    def plot_reward_per_agent(self) -> go.Figure:
        return self.plot_value_per_agent(value_per_agent=self._reward_per_agent,
                                         title="Final Reward",
                                         mesh=self.fem_problem.previous_mesh)

    def plot_error_per_element(self, normalize_by_element_area: bool = True) -> go.Figure:

        weighted_remaining_error = self._get_remaining_error(return_dimensions=False)
        return self._plot_value_per_element(
            value_per_element=np.dot(self.error_per_element, self.solution_dimension_weights),
            normalize_by_element_area=normalize_by_element_area,
            title=f"Element Errors. Remaining total error: {weighted_remaining_error:.4f}")

    def additional_plots(self, iteration: int, policy_step_function: Optional[callable] = None) -> Dict[Key, go.Figure]:
        """
        Function that takes an algorithm iteration as input and returns a number of additional plots about the
        current environment as output. Some plots may be always selected, some only on e.g., iteration 0.
        Args:
            iteration: The current iteration of the algorithm.
            policy_step_function: (Optional)
                A function that takes a graph as input and returns the action(s) and (q)-value(s)
                for each agent.

        """
        _, remaining_error_per_solution_dimension = self._get_remaining_error(return_dimensions=True)
        additional_plots = {
            "refinements_per_element": self._plot_value_per_element(value_per_element=self.refinements_per_element,
                                                                    title="Refinements per element")
        }

        subsampled_solution_dimensions = self.solution_dimension_names
        if len(self.solution_dimension_names) > 5:
            subsampled_solution_dimensions = [self.solution_dimension_names[0], self.solution_dimension_names[-1]]
        for position, solution_dimension_name in enumerate(subsampled_solution_dimensions):
            additional_plots[f"{solution_dimension_name}_std_per_element"] = self._plot_value_per_element(
                value_per_element=self.solution_std_per_element[:, position],
                title=f"Element Std for '{solution_dimension_name}'")

            additional_plots[f"{solution_dimension_name}_error_per_element"] = self._plot_value_per_element(
                value_per_element=self.error_per_element[:, position],
                normalize_by_element_area=False,
                title=f"Error per element of '{solution_dimension_name}'. "
                      f"Remaining error: {remaining_error_per_solution_dimension[position]:.4f}")

        if len(self.solution_dimension_names) > 1:
            additional_plots["weighted_solution_std_per_element"] = self._plot_value_per_element(
                value_per_element=np.dot(self.solution_std_per_element, self.solution_dimension_weights),
                title=f"Element Std of Solution Norm")

            additional_plots["weighted_solution_error_per_element"] = self.plot_error_per_element(
                normalize_by_element_area=False)

        if policy_step_function is not None:
            from util.torch_util.torch_util import detach
            actions, values = policy_step_function(observation=self.last_observation)
            if len(actions) == self.num_agents:
                additional_plots["final_actor_evaluation"] = self.plot_value_per_agent(detach(actions),
                                                                                       title=f"Action per Agent at "
                                                                                             f"Step {self._timestep}")
            if len(values) == self.num_agents:
                additional_plots["final_critic_evaluation"] = self.plot_value_per_agent(detach(values),
                                                                                        title=f"Critic Evaluation at "
                                                                                              f"Step {self._timestep}")

        if self._reward_type in ["spatial", "spatial_area"]:
            additional_plots["cumulative_reward_per_agent"] = self.plot_cumulative_reward_per_agent()
            additional_plots["reward_per_agent"] = self.plot_reward_per_agent()

        additional_plots |= self.fem_problem.additional_plots()
        return additional_plots

    def __deepcopy__(self, memo):
        from copy import deepcopy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))

        # reinitialize stateless (lambda-) functions
        # it is sufficient to call the register functions,
        # as only new objects for the stateless lambda functions have to be created
        setattr(result, '_element_feature_functions', result._register_element_features())
        if self._include_globals:
            setattr(result, '_global_feature_functions', result._register_global_features())

        return result
