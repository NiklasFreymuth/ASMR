import numpy as np
from skfem import Basis, Mesh

from src.environments.mesh.mesh_refinement.problems.problem_util.error_integrator_util import get_integration_weights, \
    find_integration_points_on_edges, get_integrated_differences, probes_from_elements
from src.environments.mesh.mesh_refinement.problems.problem_util.finite_element_util import scalar_linear_basis
from src.environments.mesh.mesh_refinement.problems.problem_util.pde_solution import PDESolution
from util.types import *


class ErrorIntegrator:
    def __init__(self, error_metrics: Union[List[str], str],
                 integration_mesh: Mesh,
                 solution_calculation_function: Callable,
                 boundary_and_basis_creation_function: Callable
                 ):
        """
        Initialize the error integrator. The error integrator is used to calculate an error estimate per element/face
        of the mesh by integration over the domain using a fine-grained ground truth mesh.
        Args:
            error_metrics: The metric to use for the error estimation. Can be either
                "squared": the squared error is used
                "mean": the absolute error is used
            integration_mesh: The fine-grained ground truth mesh
            solution_calculation_function: The function to use to calculate the fine-grained ground truth solution
            boundary_and_basis_creation_function: The function to use to create the fine-grained ground truth basis
                Takes a mesh as input and returns a basis with a task-specific element for the mesh,
                as well as potential boundary conditions
        Returns:

        """
        if isinstance(error_metrics, str):
            error_metrics = [error_metrics]
        self._error_metrics = error_metrics
        self._integration_point_type = "elements"

        self._pde_solution = self._get_pde_solution(integration_mesh=integration_mesh,
                                                    solution_calculation_function=solution_calculation_function,
                                                    boundary_and_basis_creation_function=boundary_and_basis_creation_function)

    def get_error_estimate(self, coarse_basis: Basis,
                           coarse_solution: np.array) -> Dict[Key, np.array]:
        """
        Wrapper for get_integrated_differences
        Args:
            coarse_basis: The basis of the coarse mesh to calculate the error estimate for based on the fine integration
                mesh.
            coarse_solution: The solution of the coarse mesh to calculate the error estimate for based on the fine
                integration mesh.

        Returns:
            A dictionary
            {"mean": The absolute error estimate as a vector per element/face of the mesh,
            "squared": The squared error estimate as a vector per element/face of the mesh,
            }

        """
        integration_points = self._pde_solution.integration_points
        integration_weights = self._pde_solution.integration_weights
        reference_evaluation = self._pde_solution.reference_evaluation

        # calculate the interpolated solution of the coarse mesh at the integration points of the reference mesh
        # save the corresponding elements for each integration point, so that we do not need to calculate things twice
        scalar_basis = scalar_linear_basis(coarse_basis)
        corresponding_elements = scalar_basis.mesh.element_finder()(*integration_points)
        probes = probes_from_elements(scalar_basis, integration_points, corresponding_elements)  # coo-matrix
        coarse_evaluation = probes @ coarse_solution

        # assume that each midpoint of the reference mesh corresponds to exactly one face or one edge of the coarse mesh
        # This works since the meshes live on the same domain and the reference mesh is a refinement of the coarse mesh.
        # Notably, integration points can never fall on the boundary, as the boundaries are the same for both meshes,
        # and each integration point is the midpoint of a face of the reference mesh.
        edge_indices = find_integration_points_on_edges(coarse_basis=coarse_basis,
                                                        integration_points=integration_points,
                                                        corresponding_elements=corresponding_elements)

        # pre-calculate the pointwise differences between the reference and coarse solution for each integration point
        num_coarse_elements = coarse_basis.mesh.t.shape[1]
        pointwise_differences = np.abs(reference_evaluation - coarse_evaluation)

        invalid_elements = corresponding_elements == -1
        if invalid_elements.any():
            # fallback due to numerical inaccuracies for the corresponding elements.
            # This happens once in a couple million integration points, so we can just ignore these points entirely
            pointwise_differences[invalid_elements] = 0
            corresponding_elements[invalid_elements] = 0

        # mark points that lie on edges
        valid_edge_indices = edge_indices != -1  # !=-1 means that the point lies on an edge
        edge_elements = coarse_basis.mesh.f2t[:, edge_indices[valid_edge_indices]].T

        # mark remaining points that lie on elements/faces
        full_elements = corresponding_elements[~valid_edge_indices]

        element_wise_metrics = {error_metric: get_integrated_differences(pointwise_differences=pointwise_differences,
                                                                         integration_weights=integration_weights,
                                                                         corresponding_edge_elements=edge_elements,
                                                                         corresponding_full_elements=full_elements,
                                                                         valid_edge_indices=valid_edge_indices,
                                                                         num_coarse_elements=num_coarse_elements,
                                                                         error_metric=error_metric)
                                for error_metric in self._error_metrics
                                }

        error_histogram = np.sort(pointwise_differences, axis=0)
        num_integration_points = integration_points.shape[1]

        top_errors = {"top0.1": np.sum(error_histogram[-int(num_integration_points / 1000):], axis=0),
                      "top5": np.sum(error_histogram[-int(num_integration_points / 20):], axis=0)
                      }
        error_estimation_dict = element_wise_metrics | top_errors
        return error_estimation_dict

    def _get_pde_solution(self, integration_mesh: Mesh, solution_calculation_function: Callable,
                          boundary_and_basis_creation_function: Callable) -> PDESolution:
        """
        create the fine-grained ground truth mesh and basis. This is the mesh and basis that will be used for
        the integration of the error estimate.
        Args:
            boundary_and_basis_creation_function: The function to use to create the fine-grained ground truth basis
            integration_mesh: The mesh to use for the fine-grained ground truth mesh
            solution_calculation_function: The function to use to calculate the fine-grained ground truth solution

        Returns:

        """

        reference_basis = boundary_and_basis_creation_function(mesh=integration_mesh)
        integration_points = self._get_integration_points(reference_basis=reference_basis)
        integration_weights = self._get_integration_weights(reference_basis=reference_basis)
        reference_evaluation = self._get_reference_evaluation(reference_basis=reference_basis,
                                                              solution_calculation_function=solution_calculation_function)
        pde_solution = PDESolution(integration_points=integration_points,
                                   integration_weights=integration_weights,
                                   reference_evaluation=reference_evaluation)
        return pde_solution

    def _get_integration_points(self, reference_basis: Basis) -> np.array:
        """
        Returns the integration points used for error estimation via integration.
        Returns:

        """
        if self._integration_point_type == "vertices":
            integration_points = reference_basis.mesh.p
        elif self._integration_point_type == "elements":
            vertex_positions = reference_basis.mesh.p.T
            element_indices = reference_basis.mesh.t.T
            element_positions = vertex_positions[element_indices]
            elements_midpoints = np.mean(element_positions, axis=1)
            integration_points = elements_midpoints.T
        elif self._integration_point_type == "quadrature_points":
            integration_points = reference_basis.global_coordinates()[:].reshape(2, -1)
        else:
            raise ValueError(f"Unknown integration point type '{self._integration_point_type}'")
        return integration_points

    def _get_integration_weights(self, reference_basis: Basis) -> np.array:
        """
        Returns the integration weights used for error estimation via integration. The weights are the normalized
        areas of the faces of the integration points.
        This works because we have the same amount of integrations points for each face
        Returns: An array of normalized weights for each integration point

        """
        element_integration_weights = get_integration_weights(basis=reference_basis)

        if self._integration_point_type == "vertices":
            # get integration weights for each vertex
            # move the integration weights to the corresponding vertices
            # element_indices = self.reference_basis.mesh.t.T
            vertex_integration_weights = np.zeros(reference_basis.mesh.p.shape[1])
            # add all indices without buffering, i.e., for a[0,0,1], add something to the element 0 twice
            np.add.at(vertex_integration_weights, reference_basis.mesh.t, element_integration_weights)
            integration_weights = vertex_integration_weights
        elif self._integration_point_type == "elements":
            # get integration weights for each element
            integration_weights = element_integration_weights
        elif self._integration_point_type == "quadrature_points":
            raise NotImplementedError("Integration weights for quadrature points not implemented yet")
        else:
            raise ValueError(f"Unknown integration point type '{self._integration_point_type}'")
        return integration_weights

    def _get_reference_evaluation(self, reference_basis,
                                  solution_calculation_function: Callable,
                                  integration_points: Optional[np.array] = None) -> np.array:
        """
        gets the reference evaluation of the solution at the integration points.
        Args:
            reference_basis: The basis to use for the reference mesh
            solution_calculation_function: The function to use to calculate the fine-grained ground truth solution
            integration_points: The integration points to use for the reference evaluation

        Returns:

        """
        reference_solution = solution_calculation_function(basis=reference_basis)
        if self._integration_point_type == "vertices":
            # when using the vertices as integration points, the solution is also the reference evaluation
            reference_evaluation = reference_solution
        elif self._integration_point_type == "elements":
            # use the element midpoints as integration points
            element_indices = reference_basis.mesh.t.T
            element_evaluations = reference_solution[element_indices]
            element_averages = np.mean(element_evaluations, axis=1)
            reference_evaluation = element_averages
        else:
            assert integration_points is not None, "Integration points must be given when using quadrature " \
                                                   "points as integration type"
            # this always works, but is really slow
            reference_evaluation_function = scalar_linear_basis(reference_basis).interpolator(y=reference_solution)

            reference_evaluation = reference_evaluation_function(integration_points)
        return reference_evaluation
