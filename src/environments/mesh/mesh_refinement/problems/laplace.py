r"""Laplacian task with an outer boundary with boundary condition Omega_0=0, 
and an inner boundary with condition Omega_1=1
"""

import os

import numpy as np
import skfem as fem
from plotly import graph_objects as go
from skfem import Mesh, Basis, ElementTriP1
from skfem.models.poisson import laplace

from src.environments.mesh.mesh_refinement.mesh_refinement_util import element_midpoints
from src.environments.mesh.mesh_refinement.mesh_refinement_util import get_line_segment_distances
from src.environments.mesh.mesh_refinement.mesh_refinement_visualization import contour_trace_from_element_values, \
    get_mesh_traces
from src.environments.mesh.mesh_refinement.domains.square_hole import SquareHole
from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from util.types import *
from util.visualization import get_layout

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def in_rectangle(x: np.array, rectangle: np.array) -> np.array:
    """
    Args:
        x: Points to query. Must have shape (2,...)
        rectangle: 4-tuple (x1, y1, x2, y2) that describes the rectangle boundaries
    Returns:
        A boolean array of shape (...)
    """
    return np.logical_and(np.logical_and(x[0] >= rectangle[0], x[0] <= rectangle[2]),
                          np.logical_and(x[1] >= rectangle[1], x[1] <= rectangle[3]))


class Laplace(AbstractFiniteElementProblem):
    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Args:
            fem_config: A dictionary containing the configuration for the finite element method.
            random_state: A random state to use for reproducibility.

        """
        self._temperature = 1.0  # fixed material parameter

        super().__init__(fem_config=fem_config,
                         random_state=random_state)  # also calls reset() and thus _set_pde()
        assert isinstance(self._domain, SquareHole), f"Laplace task currently only defined " \
                                                     f"for SquareHole domain, given '{type(self._domain)}'"

    def _set_pde(self) -> None:
        """
        This function is called to draw a new PDE from a family of available PDEs.
        Since the PDE does not change for this task, we do not do anything here except to grab the current hole boundary

        """
        self._source_rectangle = self._domain.hole_boundary

    def add_boundary_conditions_and_create_basis(self, mesh: Mesh) -> Basis:
        source_facets = mesh.facets_satisfying(partial(in_rectangle, rectangle=self._source_rectangle),
                                               boundaries_only=True)
        mesh = mesh.with_boundaries(
            {
                "source": source_facets,
            }
        )

        basis = Basis(mesh, ElementTriP1())
        return basis

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution for the parameterized Poisson equation based on the given basis. The solution is
        calculated for every node/vertex of the underlying mesh, and the way it is calculated depends on the element
        used in the basis.
        For example, ElementTriP1() elements will draw 3 quadrature points for each face that lie in the middle of the
        edge between the barycenter of the face and its spanning nodes, and then linearly interpolate based on those
        elements.
        Args:
            cache:

        Returns: An array (num_vertices, 2), where every entry corresponds to a vector of the norm of the stress
        at the corresponding vertex, and the magnitude of the displacement at the corresponding vertex.

        """

        # set boundary conditions
        # Get all degrees of freedom and set appropriate entry to prescribed BCs.
        boundary_temperature = basis.zeros()
        boundary_temperature[basis.get_dofs({"source"})] = self._temperature

        # Assemble matrices, solve problem
        matrix = fem.asm(laplace, basis)
        solution = fem.solve(*fem.condense(matrix, x=boundary_temperature, I=basis.mesh.interior_nodes()))
        return solution

    ##############################
    #         Observations       #
    ##############################

    def element_features(self, basis: Basis, element_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_elements, num_features) containing the features for each element.
        Args:
            basis: The basis to use for the calculation
            element_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        Returns: An array of shape (num_elements, num_features) containing the features for each element.
        """
        if "distance_to_source" in element_feature_names:
            mesh = basis.mesh
            return np.array([get_line_segment_distances(points=element_midpoints(mesh),
                                                        projection_segments=self.source_line_segments,
                                                        return_minimum=True)]).T
        else:
            return None

    def global_features(self, basis: Basis, global_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_global_features,) containing the features for the entire mesh.
        Args:
            basis: The basis to use for the calculation
            global_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        """
        return None

    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["temperature"]

    @property
    def source_line_segments(self):
        source_facets = self.initial_mesh.facets_satisfying(partial(in_rectangle,
                                                                    rectangle=self._source_rectangle),
                                                            boundaries_only=True)
        boundary_node_indices = self.initial_mesh.facets[:, source_facets]
        line_segments = self.initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        return line_segments

    ###############################
    # plotting utility functions #
    ###############################

    def _minimum_source_distance_plot(self, mesh: Mesh) -> go.Figure:
        """
        Plots the minimum distance to the heat source for each face of the mesh
        Args:
            mesh:

        Returns:

        """
        element_midpoints = mesh.p[:, mesh.t].mean(axis=1).T
        face_distances = get_line_segment_distances(points=element_midpoints,
                                                    projection_segments=self.source_line_segments,
                                                    return_minimum=True)
        contour_trace = contour_trace_from_element_values(mesh=mesh, element_evaluations=face_distances,
                                                          trace_name="Source distance per agent")
        mesh_trace = get_mesh_traces(mesh)
        traces = contour_trace + mesh_trace

        boundary = np.concatenate((mesh.p.min(axis=1), mesh.p.max(axis=1)), axis=0)
        layout = get_layout(boundary=boundary,  # min, max of deformed mesh
                            title="Minimum Distance to Heat Source")
        mininum_source_distance_plot = go.Figure(data=traces,
                                                 layout=layout)
        return mininum_source_distance_plot

    def additional_plots_from_basis(self, basis) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            basis: 

        """
        additional_plots = {
            "mininum_source_distance": self._minimum_source_distance_plot(mesh=basis.mesh)
        }
        return additional_plots
