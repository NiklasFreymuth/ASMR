r"""
Buffer of finite element problems. This class stores several finite element problems and allows to sample from them.
In particular, each FEM Problem consists of an original coarse mesh and basis, and a fine-grained mesh, basis,
and solution.
The meshes are skfem Mesh objects, each basis is a skfem Basis object, and the solution is a PDESolution object.
This class further gives interfaces to interact with the finite element problems, e.g., to calculate a reward or to plot
them.
"""
import copy
import os
from typing import Dict, Any, List, Union, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
from skfem import Basis, Mesh

from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_visualization import get_evaluation_heatmap_from_basis, \
    get_mesh_traces
from modules.swarm_environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import \
    AbstractFiniteElementProblem
from modules.swarm_environments.mesh.mesh_refinement.problems.problem_util.finite_element_util import to_scalar_basis

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FEMProblemWrapper:
    def __init__(self, *,
                 fem_config: Dict[Union[str, int], Any],
                 fem_problem: AbstractFiniteElementProblem,
                 pde_features: Dict[str, List[str]], ):
        """
        Wrapper around a finite element problem.
        This class stores all the information about the problem itself, as well as temporary information about the
        current mesh, basis, and solution. It also provides interfaces to interact with the problem, e.g., to calculate
        a reward or to plot it.
        """
        self._fem_config = fem_config

        #################
        # problem state #
        #################
        # parameters for the partial differential equation
        self.fem_problem = fem_problem

        self._pde_element_feature_names = pde_features["element_features"]

        #  The metric to use for the error estimation. Can be either 'indicator', 'squared' or 'absolute'
        self._error_metric = fem_config.get("error_metric", "mean")

        ###################
        # mesh parameters #
        ###################
        self._refinements_per_element = None
        self._mesh = None  # current mesh
        self._basis = None  # current basis
        self._solution = None  # solution vector or tensor for the current basis
        self._nodal_solution = None  # solution vector or tensor for the current basis per node
        self._previous_basis = None  # basis of the previous mesh/step

        #####################
        # plotting utility #
        ####################
        # plot the mesh by interpolating on a (by default) 101x101 grid
        self._plot_resolution = fem_config.get("plot_resolution", 101)
        self._plot_boundary = np.array(fem_config.get("domain").get("boundary", [0, 0, 1, 1]))

    def reset(self):
        """
        Resets the finite element problem to its initial state.
        Args:

        """
        self.mesh = self.fem_problem.initial_mesh
        self._previous_basis = copy.deepcopy(self._basis)  # set previous basis to current basis after reset
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
        self._solution = self.fem_problem.calculate_solution(basis=self._basis, cache=True)
        if self.fem_problem.has_different_nodal_solution:
            self._nodal_solution = self.fem_problem.nodal_solution
        else:
            self._nodal_solution = self._solution

    def get_error_estimate_per_element(self, error_metric: str) -> np.array:
        """
        Wrapper for the element-wise error estimate of the current fem Problem.
        Args:
            error_metric:
                The metric to use for the error estimation. Can be either
                    "mean": the mean/average error is used
                    "indicator": an error indicator is used. Only works for problems that define an error indicator,
                        i.e., Poisson. Should only be used for debugging purposes
        Returns:
            The error estimate per element/face of the mesh. This error estimate is an entity per element, and is
            calculated by integrating the error over the element.

        """
        return self.fem_problem.get_error_estimate_per_element(error_metric=error_metric,
                                                               basis=self._basis,
                                                               solution=self._solution)

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

    def element_features(self) -> np.array:
        """
        Returns a dictionary of element features that are used as observations for the  RL agents.
        Args:

        Returns: An array (num_elements, num_features) that contains the features for each element of the mesh

        """
        return self.fem_problem.element_features(basis=self._basis,
                                                 element_feature_names=self._pde_element_feature_names)

    def project_to_scalar(self, values: np.array) -> np.array:
        """
        Projects a value per node and solution dimension to a scalar value per node.
        Args:
            values: A vector of shape ([num_vertices,] solution_dimension)

        Returns: A scalar value per vertex
        """
        # equivalent to np.einsum('ij, j -> i', self._solution, self.fem_problem.solution_dimension_weights)
        # if simple weights are used
        return self.fem_problem.project_to_scalar(values=values)

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
        return self._mesh

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
        self._previous_basis: Basis = copy.deepcopy(self._basis)
        self._mesh = mesh
        # build basis by assigning an element type to the mesh elements whenever we change the mesh
        # The element type determines which (local) function is used for each element, i.e., it is used to determine
        # which calculations are performed
        self._basis: Basis = self.fem_problem.add_boundary_conditions_and_create_basis(self.mesh)

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
    def nodal_solution(self) -> np.array:
        """

        Returns: solution vector per *vertex* of the mesh.
            An array (num_vertices, solution_dimension),
            where every entry corresponds to the solution of the underlying PDE at the position of the
            respective node/vertex.
            For problems with a one-dimensional solution per vertex, we return an array of shape (num_vertices, 1)

        """
        assert self._nodal_solution is not None, "The solution has not been calculated yet"
        return self._nodal_solution

    @property
    def boundary_line_segments(self) -> np.array:
        """
        The boundary of the domain, represented by line segements
        Returns: an array of shape (#line_segments, 4), where the last dimension is over (x0, y0, x1, y1)

        """
        return self.fem_problem.boundary_line_segments

    @property
    def element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements.
        Returns: np.array of shape (num_elements, 2)

        """
        from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_util import element_midpoints
        return element_midpoints(self.mesh)

    @property
    def element_indices(self) -> np.array:
        return self.mesh.t.T

    @property
    def vertex_positions(self) -> np.array:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)

        """
        return self.mesh.p.T

    @property
    def mesh_edges(self) -> np.array:
        """
        Returns: the edges of all vertices/nodes of the mesh. Shape (2, num_edges)
        """
        return self.mesh.facets

    @property
    def element_neighbors(self) -> np.array:
        """
        Find neighbors of each element. Shape (2, num_neighbors)
        Returns:

        """
        # f2t are element/face neighborhoods, which are set to -1 for boundaries
        return self.mesh.f2t[:, self.mesh.f2t[1] != -1]

    @property
    def error_metric(self) -> str:
        """
        Method to use for error estimation. Current error estimation methods may be
            "squared": the squared error is used
            "mean": the absolute error is used
            "indicator": an error indicator is used. Only works for problems that define an error indicator
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
        return self.fem_problem.additional_plots_from_basis(self._basis)

    def approximated_weighted_solution_traces(self) -> List[BaseTraceType]:
        """

        Returns: plotly traces of the approximated solution of the underlying PDE.

        """
        weighted_solution = self.project_to_scalar(self._nodal_solution)
        evaluation_function = to_scalar_basis(self._basis, linear=True).interpolator(y=weighted_solution)
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
        return get_evaluation_heatmap_from_basis(basis=self._basis,
                                                 evaluation_function=evaluation_function,
                                                 resolution=self._plot_resolution,
                                                 normalize_by_element_area=normalize_by_element_area)

    @property
    def plot_boundary(self):
        return self._plot_boundary
