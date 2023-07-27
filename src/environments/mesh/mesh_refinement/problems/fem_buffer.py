r"""
Buffer of finite element problems. This class stores several finite element problems and allows to sample from them.
In particular, each FEM Problem consists of an original coarse mesh and basis, and a fine-grained mesh, basis, 
and solution.
The meshes are skfem Mesh objects, each basis is a skfem Basis object, and the solution is a PDESolution object.
This class further gives interfaces to interact with the finite element problems, e.g., to calculate a reward or to plot
them.
"""
import os

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Basis, Mesh

from src.environments.mesh.mesh_refinement.mesh_refinement_visualization import get_evaluation_heatmap_from_basis, \
    get_mesh_traces
from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from src.environments.mesh.mesh_refinement.problems.get_finite_element_problem import get_finite_element_problem_class
from src.environments.mesh.mesh_refinement.problems.problem_util.finite_element_util import scalar_linear_basis
from util.function import filter_included_fields
from util.index_sampler import IndexSampler
from util.types import *

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FEMBuffer:
    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Initializes the AbstractFiniteElementProblem.

        Args:
            fem_config: Config containing additional details about the finite element method. Contains
                domain_config: Configuration describing the family of geometries for this problem
                plot_resolution: The resolution of the plots in x and y direction
                error_metric:
                 The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean error is used
            random_state: A random state to use for drawing random numbers
        """
        self._fem_config = fem_config

        #################
        # problem state #
        #################
        self._random_state = random_state

        # parameters for a buffer of pdes to avoid calculating the same pde multiple times
        num_pdes = fem_config.get("num_pdes")  # number of pdes to store. None, 0 or -1 means infinite
        self._use_buffer = num_pdes is not None and num_pdes > 0

        num_pdes = num_pdes if self._use_buffer else 1
        self._index_sampler = IndexSampler(num_pdes,
                                           random_state=self._random_state)
        self._fem_problems: Optional[List[AbstractFiniteElementProblem]] = [None] * num_pdes
        self._current_fem_problem = None

        # parameters for the partial differential equation
        self._fem_problem_class: Type[AbstractFiniteElementProblem] \
            = get_finite_element_problem_class(fem_config=fem_config)
        pde_config = fem_config.get(fem_config.get("pde_type"))
        self._pde_element_feature_names = filter_included_fields(pde_config.get("element_features", {}))
        self._pde_global_feature_names = filter_included_fields(pde_config.get("global_features", {}))

        # parameters for the remeshing
        self._refinements_per_element = None

        #  The metric to use for the error estimation. Can be either 'squared', 'mean' or 'maximum'
        self._error_metric = fem_config.get("error_metric", "mean")

        ###################
        # mesh parameters #
        ###################
        self._current_mesh = None  # current mesh
        self._current_basis = None  # current basis
        self._current_solution = None  # solution vector or tensor for the current basis
        self._previous_basis = None  # basis of the previous mesh/step

        #####################
        # plotting utility #
        ####################
        # plot the mesh by interpolating on a (by default) 101x101 grid
        self._plot_resolution = fem_config.get("plot_resolution", 101)
        self._plot_boundary = np.array([0, 0, 1, 1])  # uniform rectangle

    def next(self) -> None:
        """
        Draws the next finite element problem. This method is called at the beginning of each episode and draws a
        (potentially new) finite element problem from the buffer.
        Returns:

        """
        pde_idx = self._index_sampler.next()

        self._next_from_idx(pde_idx=pde_idx)

    def _next_from_idx(self, pde_idx: int):
        if (not self._use_buffer) or self._fem_problems[pde_idx] is None:
            # draw a new fem_problem from the given distribution if we are not using a buffer or if the buffer entry
            # is empty
            new_seed = self._random_state.randint(0, 2 ** 31)
            new_problem = self._fem_problem_class(fem_config=self._fem_config,
                                                  random_state=np.random.RandomState(seed=new_seed))
            self._fem_problems[pde_idx] = new_problem
        self._current_fem_problem = self._fem_problems[pde_idx]
        self.mesh = self.current_fem_problem.initial_mesh
        self._previous_basis = copy.deepcopy(self._current_basis)  # set previous basis to current basis after reset
        self._refinements_per_element = np.zeros(self.num_elements, dtype=np.int)

    def calculate_solution_and_get_error(self) -> np.array:
        """
        Calculates a solution of the underlying PDE for the given finite element basis, and uses this solution
        to estimate an error per vertex.
        Args:
        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.

        """
        self.calculate_solution()

        error_estimation_dict = self.get_error_estimate_per_element(error_metric=self._error_metric)
        return error_estimation_dict

    def calculate_solution(self) -> None:
        """
        Calculates a solution of the underlying PDE for the given finite element basis, and caches the solution
        for plotting.
        Args:

        """
        self._current_solution = self.current_fem_problem.calculate_solution(basis=self._current_basis, cache=True)

    def get_error_estimate_per_element(self, error_metric: str) -> np.array:
        """
        Wrapper for the element-wise error estimate of the current fem Problem.
        Args:
            error_metric:
                The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean/average error is used
                    "maximum":
        Returns:
            The error estimate per element/face of the mesh. This error estimate is an entity per element, and is
            calculated by integrating the error over the element.

        """
        return self.current_fem_problem.get_error_estimate_per_element(error_metric=error_metric,
                                                                       basis=self._current_basis,
                                                                       solution=self._current_solution)

    def refine_mesh(self, elements_to_refine: np.array) -> np.array:
        """
        Refines the mesh by splitting the given elements.
        Args:
            elements_to_refine: An array of element indices to refine. May be empty, in which case no refinement happens

        Returns: A mapping from the old element indices to the new element indices.

        """
        if len(elements_to_refine) > 0:  # do refinement
            refined_mesh = self.mesh.refined(elements_to_refine)

            # track how often each element has been refined, update internal number of refinements
            new_element_midpoints = refined_mesh.p[:, refined_mesh.t].mean(axis=1).T
            corresponding_elements = self.mesh.element_finder()(x=new_element_midpoints[:, 0],
                                                                y=new_element_midpoints[:, 1])
            element_indices, inverse_indices, counts = np.unique(corresponding_elements, return_counts=True,
                                                                 return_inverse=True)
            self._update_refinements_per_element(counts, element_indices, inverse_indices)
        else:
            refined_mesh = self.mesh

            # default element mapping if no refinement happens
            inverse_indices = np.arange(self.mesh.t.shape[1]).astype(np.int64)
        self.mesh = refined_mesh  # update here because this also updates the basis and previous mesh
        return inverse_indices

    def _update_refinements_per_element(self, counts, element_indices, inverse_indices) -> None:
        """
        Updates the number of times each element has been refined
        Args:
            counts: The number of new elements corresponding to each old one
            element_indices: The indices of the old elements
            inverse_indices: The indices of the new elements

        Returns:

        """
        # mark all elements that were split as refined, whether the action directly selected them or not
        self._refinements_per_element[element_indices] += counts - 1
        # update refinements by assigning the previous number of refinements to each child
        self._refinements_per_element = self._refinements_per_element[inverse_indices]

    ##############################
    #         Observations       #
    ##############################

    @property
    def num_pde_element_features(self) -> int:
        return len(self._pde_element_feature_names)

    @property
    def num_pde_global_features(self) -> int:
        return len(self._pde_global_feature_names)

    def element_features(self) -> np.array:
        """
        Returns a dictionary of element features that are used as observations for the  RL agents.
        Args:

        Returns: An array (num_elements, num_features) that contains the features for each element of the mesh

        """
        return self.current_fem_problem.element_features(basis=self._current_basis,
                                                         element_feature_names=self._pde_element_feature_names)

    def global_features(self) -> Dict[str, callable]:
        """
        Returns a dictionary of global features that are used as observations for the  RL agents.
        Args:

        Returns: A dictionary of global features that are used as observations for the  RL agents.
        """
        return self.current_fem_problem.global_features(basis=self._current_basis,
                                                        global_feature_names=self._pde_global_feature_names)

    @property
    def solution_dimension_names(self) -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return self._fem_problem_class.solution_dimension_names()

    @property
    def solution_dimension_weights(self) -> np.array:
        """
        Returns a list of weights for the solution dimensions. This is used to weight the solution dimensions
        when calculating the error.
        Returns: A list of weights for the solution dimensions

        """
        return self.current_fem_problem.solution_dimension_weights

    ##############
    # properties #
    ##############

    @property
    def mesh(self) -> Optional[Mesh]:
        """
        The mesh to use for the finite element problem. This is a skfem.Mesh object.
        Does not include the boundary conditions.
        Returns:

        """
        return self._current_mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        """
        Update the mesh and its corresponding basis from the outside. This happens e.g., when the algorithm chooses
        to refine the mesh
        Args:
            mesh: The new mesh. Will overwrite the old mesh and is used to build a new basis, which may contain
            additional boundary conditions.

        Returns: None

        """
        self._previous_basis: Basis = copy.deepcopy(self._current_basis)
        self._current_mesh = mesh
        # build basis by assigning an element type to the mesh elements whenever we change the mesh
        self._current_basis: Basis = self.current_fem_problem.add_boundary_conditions_and_create_basis(self.mesh)

    @property
    def current_fem_problem(self) -> AbstractFiniteElementProblem:
        """
        The current finite element problem, i.e., the one that is currently being solved.
        If the current problem does not exist, it is created.
        Returns:

        """
        return self._current_fem_problem

    @property
    def previous_mesh(self) -> Mesh:
        return self._previous_basis.mesh

    @property
    def num_elements(self) -> int:
        return self.mesh.t.shape[1]

    @property
    def refinements_per_element(self) -> np.array:
        return self._refinements_per_element

    @property
    def solution(self) -> np.array:
        """

        Returns: solution vector per *vertex* of the mesh.
            An array (num_vertices, solution_dimension),
            where every entry corresponds to the solution of the underlying PDE at the position of the
            respective node/vertex.
            For problems with a one-dimensional solution per vertex, we return an array of shape (num_vertices, 1)

        """
        assert self._current_solution is not None, "The solution has not been calculated yet"
        return self._current_solution

    @property
    def boundary_line_segments(self) -> np.array:
        """
        The boundary of the domain, represented by line segements
        Returns: an array of shape (#line_segments, 4), where the last dimension is over (x0, y0, x1, y1)

        """
        return self.current_fem_problem.boundary_line_segments

    @property
    def element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements.
        Returns: np.array of shape (num_elements, 2)

        """
        from src.environments.mesh.mesh_refinement.mesh_refinement_util import element_midpoints
        return element_midpoints(self.mesh)

    @property
    def vertex_positions(self) -> np.array:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)

        """
        return self.mesh.p.T

    @property
    def error_metric(self) -> str:
        """
        Method to use for error estimation. Current error estimation methods may be
            "squared": the squared error is used
            "mean": the mean error is used
            "maximum": The maximum error is used
        Returns:

        """
        return self._error_metric

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots(self) -> Dict[str, go.Figure]:
        """
        This function can be overwritten to add additional plots specific to the current FEM problem.
        Returns:

        """
        return self.current_fem_problem.additional_plots_from_basis(self._current_basis)

    def approximated_weighted_solution_traces(self) -> List[BaseTraceType]:
        """

        Returns: plotly traces of the approximated solution of the underlying PDE.

        """
        weighted_solution = np.dot(self._current_solution, self.solution_dimension_weights)
        # equivalent to np.einsum('ij, j -> i', self._current_solution, self.solution_dimension_weights)
        evaluation_function = scalar_linear_basis(self._current_basis).interpolator(y=weighted_solution)
        # evaluation function is interpolated solution
        contour_trace = self.contour_trace_on_domain(evaluation_function=evaluation_function)
        mesh_trace = get_mesh_traces(self.mesh)
        traces = contour_trace + mesh_trace
        return traces

    def contour_trace_on_domain(self, evaluation_function: callable,
                                normalize_by_element_area: bool = False) -> List[BaseTraceType]:
        """
        the evaluation function takes a list of points as, and outputs some value for each point.
        We can feed a grid into it to get a grid of values, which we can then plot as a contour plot.
        Args:
            evaluation_function:
            normalize_by_element_area:

        Returns:

        """
        return get_evaluation_heatmap_from_basis(basis=self._current_basis,
                                                 evaluation_function=evaluation_function,
                                                 resolution=self._plot_resolution,
                                                 normalize_by_element_area=normalize_by_element_area)

    @property
    def current_size(self):
        return sum([1 for _ in self._fem_problems if _ is not None])

    @property
    def plot_boundary(self):
        return self._plot_boundary
