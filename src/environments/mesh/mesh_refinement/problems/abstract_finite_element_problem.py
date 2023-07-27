r"""
Base class for an Abstract (Static) Finite Element Problem.
The problem specifies a partial differential equation to be solved, and the boundary conditions. It also specifies the
domain/geometry of the problem.
Currently, uses a triangular mesh with linear elements.
"""
import abc
import os

import numpy as np
from skfem import Basis, ElementTriP1, Mesh

from src.environments.mesh.mesh_refinement.domains.get_domain import get_domain
from src.environments.mesh.mesh_refinement.problems.problem_util.error_integrator import ErrorIntegrator
from util.types import *

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from plotly import graph_objects as go


class AbstractFiniteElementProblem(abc.ABC):
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
                    "mean": the mean/average error is used
            random_state: A random state to use for drawing random numbers
        """
        self._random_state = random_state
        self._integration_error_metrics = ["squared", "mean", "maximum"]

        # initialize domain, pde on domain, and possibly error integrator
        self._domain = get_domain(domain_config=fem_config.get("domain"),
                                  random_state=copy.deepcopy(random_state))
        self._set_pde()  # set pde after domain, since the domain may be used to generate the pde.
        self._set_error_integrator(fem_config=fem_config)

        self._plot_boundary = np.array([0, 0, 1, 1])  # uniform rectangle

    def _set_error_integrator(self, fem_config: ConfigDict) -> None:
        if fem_config.get("error_metric", "mean") in self._integration_error_metrics:  # do numerical integration
            self._error_integrator = ErrorIntegrator(
                error_metrics=self._integration_error_metrics,
                integration_mesh=self._domain.get_integration_mesh(),
                solution_calculation_function=self.calculate_solution,
                boundary_and_basis_creation_function=self.add_boundary_conditions_and_create_basis)
        else:
            self._error_integrator = None

    def _set_pde(self) -> None:
        """
        Resets the PDE to a new random PDE.
        Returns:

        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _set_pde()")

    def add_boundary_conditions_and_create_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh. By default, uses a linear triangular basis
        and no boundary conditions on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh
        """
        return Basis(mesh, ElementTriP1())

    def calculate_solution(self, basis: Basis, cache: bool = False) -> np.array:
        """
        Wrapper for _calculate_solution that makes sure that the solution is a numpy array of
        shape (num_vertices, solution_dimension).
        Args:
            basis: The basis to calculate the solution for
            cache: Whether to cache information about the solution

        Returns: The solution for the given basis

        """
        solution = self._calculate_solution(basis=basis, cache=cache)
        if solution.ndim == 1:  # add a dimension if the solution is one-dimensional
            solution = solution[:, None]
        return solution

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution of the underlying PDE for the given finite element basis.
        The solution is calculated for every *node/vertex* of the underlying mesh. The way it is calculated depends on
        the element used in the basis. Here, we use, ElementTriP1() elements, which are linear elements that will draw
        3 quadrature points for each element. These points lie in the middle of the edge between the barycenter of the
        element and its spanning nodes.
        The evaluation of the element is linearly interpolated based on those elements.

        Args:
            basis: The basis to calculate the solution for
            cache: Whether to cache information about the solution

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.


        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _calculate_solution")

    def get_error_estimate_per_element(self, error_metric: str, basis: Basis,
                                       solution: np.array) -> Dict[Key, np.array]:
        """
        Calculates an error per *element/face* of the mesh in the given scikit-FEM basis.
        The error may be a scalar or a vector, depending on the problem.
        This error is calculated via integration over the domain and comparison to a fine-grained ground truth mesh.
        Args:
            basis: The basis to calculate the error estimate for
            solution: The solution of the pde in the given basis
            error_metric:
                The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean/average error is used
                    "maximum": the maximum error is used
        Returns:
            The error estimate per element/face of the mesh. This error estimate is an entity per element, and is
            calculated by integrating the error over the element.

        """
        assert error_metric in self._integration_error_metrics, \
            f"Error metric {error_metric} not supported for error estimation per element"
        error_estimation_dict = self._error_integrator.get_error_estimate(coarse_basis=basis,
                                                                          coarse_solution=solution)

        return error_estimation_dict

    #############################
    #        Observations       #
    #############################

    def element_features(self, basis: Basis, element_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_elements, num_features) containing the features for each element.
        Args:
            basis: The basis to use for the calculation
            element_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        Returns: An array of shape (num_elements, num_features) containing the features for each element.

        """
        raise NotImplementedError

    def global_features(self, basis: Basis, global_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_global_features,) containing the features for the entire mesh.
        Args:
            basis: The basis to use for the calculation
            global_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        raise NotImplementedError("The solution dimension names are not defined for AbstractFiniteElementProblem")

    @property
    def solution_dimension_weights(self) -> np.array:
        """
        Returns a list of weights for the solution dimensions. This is used to weight the solution dimensions
        when calculating the error.
        Returns: A list of weights for the solution dimensions

        """
        if len(self.solution_dimension_names()) == 1:
            return np.array([1.0])
        else:
            raise NotImplementedError("The solution dimension weights for solutions with multiple dimensions"
                                      " are not defined for AbstractFiniteElementProblem")

    ##############
    # properties #
    ##############

    @property
    def boundary_line_segments(self) -> np.array:
        """
        The boundary of the domain, represented by line segements
        Returns: an array of shape (#line_segments, 4), where the last dimension is over (x0, y0, x1, y1)

        """
        return self._domain.boundary_line_segments

    @property
    def initial_mesh(self) -> Mesh:
        """
        Returns the initial mesh of the domain.
        Returns:

        """
        return self._domain.initial_mesh

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_basis(self, basis) -> Dict[str, go.Figure]:
        """
        Returns a dictionary of additional plots that are used for visualization.
        Args:
            basis:

        Returns:

        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement 'additional_plots_from_basis'")
