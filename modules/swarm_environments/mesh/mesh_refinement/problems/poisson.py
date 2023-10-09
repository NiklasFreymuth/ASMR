r"""
Abstract Base class for Poisson equations.
The poisson equation is given as \Delta u = f, where \Delta is the Laplacian, u is the solution, and f is the
load. We consider a 2D domain with zero boundary conditions.
"""
import os
from typing import Dict, Any, List, Union, Callable, Optional

import numpy as np
import plotly.graph_objects as go
from skfem import Basis, Functional, InteriorFacetBasis
from skfem import LinearForm, asm, solve, condense, BilinearForm
from skfem.helpers import dot
from skfem.helpers import grad

from modules.swarm_environments.mesh.functions.abstract_target_function import AbstractTargetFunction
from modules.swarm_environments.mesh.functions.get_target_function import get_target_function
from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_util import element_midpoints
from modules.swarm_environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import \
    AbstractFiniteElementProblem
from modules.swarm_environments.util.function import wrapped_partial
from modules.swarm_environments.util.visualization import get_layout

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_load(positions: np.array, load: AbstractTargetFunction):
    """
    Calculate the load for positions x and y. This is the function "f" of the rhs of the poisson equation. It
    essentially defines where the "mass" of a system lies, and the solution of the poisson equation says in which
    direction the flow of gravity should be.
    A positive load means that we have positive mass, i.e., something that attracts.
    A negative load would correspond to a sink in something like fluid flow.
    For this load function, we consider loads that are Gaussian Mixture Models
    Args:
        positions: positions to evaluate as shape ( #points, 2)
        load: An AbstractTargetFunction instance that defines the GMM model that specifies the load

    Returns:

    """

    load_eval = load.evaluate(positions, include_gradient=False)
    return load_eval


###
# lots of complicated helper functions that are used to calculate an error indicator
###

def _get_edge_residual(basis: Basis, solution: np.ndarray):
    """
    Calculates the error/residual for the edge of the mesh
    Args:
        basis:
        solution:

    Returns:

    """
    # facet jump
    mesh = basis.mesh
    element = basis.elem
    fbasis = [InteriorFacetBasis(mesh, element, side=i)
              for i in [0, 1]]
    # the interior facet basis computes the integral over each element
    w = {f"u{i + 1}": fbasis[i].interpolate(solution) for i in [0, 1]}
    eta_E = edge_jump.elemental(fbasis[0], **w)  # calculates error on the boundary based on w.n, w.u1 and w.u2
    tmp = np.zeros(mesh.facets.shape[1])

    # add all indices without buffering, i.e., for a[0,0,1], add something to the element 0 twice
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(.5 * tmp[mesh.t2f], axis=0)
    return eta_E


def _get_interior_residual(basis: Basis, solution: np.ndarray, load: AbstractTargetFunction) -> np.ndarray:
    """
    Calculates the residual/error for the interior of the mesh
    Args:
        basis: A scikit-FEM Basis instance that consists of a mesh and an element.
          Must use a triangle mesh and linear triangular elements.
        solution: A solution vector per *vertex* of the mesh
        load: A AbstractTargetFunction instance that defines the GMM model that specifies the load

    Returns:

    """
    approximated_solution_field = basis.interpolate(solution)
    interior_residual_ = wrapped_partial(interior_residual, load=load)

    eta_K = Functional(interior_residual_).elemental(basis, w=approximated_solution_field)
    # .elemental evaluates element-wise
    return eta_K


def interior_residual(w, load: AbstractTargetFunction):
    """
    Provides a closed-form solution for the interior residual
    May be arbitrarily wrong for small meshes and large coefficients :(
    Args:
        w:
        load: A AbstractTargetFunction instance that defines the GMM model that specifies the load
    Returns:

    """
    h = w.h
    x, y = w.x
    positions = np.stack((x, y), axis=-1)
    return (h ** 2) * (evaluate_load(positions, load=load) ** 2)


@Functional
def edge_jump(w):
    h = w.h
    n = w.n
    dw1 = grad(w['u1'])  # short for w['u1'].grad
    dw2 = grad(w['u2'])
    return h * ((dw1[0] - dw2[0]) * n[0] +
                (dw1[1] - dw2[1]) * n[1]) ** 2


@BilinearForm
def laplace(u, v, _):
    # equivalent to `return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]`
    return dot(grad(u), grad(v))


def wrap_load(v, w, evaluate_load: callable, *args, **kwargs) -> np.ndarray:
    """
    Calculate the load for positions x and y. This is the function "f" of the rhs of the poisson equation.
    """
    x, y = w.x
    positions = np.stack((x, y), axis=-1)
    return evaluate_load(positions, *args, **kwargs) * v


class Poisson(AbstractFiniteElementProblem):
    def __init__(self, *,
                 fem_config: Dict[Union[str, int], Any],
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Initializes a Poisson equation with asymmetric load f(x,y)=x^c_x*y^c_y based on coefficients c_x and c_y.
        The result is a Poisson equation whose "center of mass" is parameterized by the given coefficients, with
        positive coefficients shifting the center in the positive direction and vice versa.
        Args:
            fem_config: Configuration for the finite element method. Contains
                domain: Dictionary for the (family of) problem domain(s)
                poisson: Dictionary consisting of keys for the specific kind of load function to build
            random_state: Internally used random_state to generate x and y coefficients. Will be used to new
                coefficients for every reset() call.
        """
        poisson_config = fem_config.get("poisson")
        boundary = np.array(fem_config.get("domain").get("boundary", [0, 0, 1, 1]))
        self._load_function = get_target_function(target_function_name=poisson_config.get("load_type", "gmm"),
                                                  target_function_config=poisson_config,
                                                  fixed_target=poisson_config.get("fixed_load"),
                                                  boundary=boundary,
                                                  random_state=random_state)
        super().__init__(fem_config=fem_config,
                         random_state=random_state)  # also calls reset()

    def _set_pde(self) -> None:
        """
        Draw a new load function

        """
        self._load_function.reset(valid_point_function=self._points_in_domain)

    def _points_in_domain(self, candidate_points: np.array) -> np.array:
        """
        Returns a subset of points that are inside the current domain, i.e., that can be found in the mesh.
        Returns:

        """
        corresponding_elements = self._domain.initial_mesh.element_finder()(x=candidate_points[:, 0],
                                                                            y=candidate_points[:, 1])
        valid_points = candidate_points[corresponding_elements != -1]
        return valid_points

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

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.

        """
        K = asm(laplace, basis)  # finite element assembly. Returns a sparse matrix
        f = asm(LinearForm(self.load), basis)  # rhs of the linear system that matches the load function

        interior = basis.mesh.interior_nodes()  # mesh nodes that are not part of the boundary

        # enforce Dirichlet boundary conditions
        # from skfem import enforce
        # K, f = enforce(K, f, D=basis.mesh.boundary_nodes())

        condensed_system = condense(K, f, I=interior)  # condense system by zeroing out all nodes that lie on a boundary

        # "solve" just takes a sparse matrix and a right handside, i.e., it just solves a (linear) system of equations
        solution = solve(*condensed_system)
        return solution

    def _get_error_estimate_from_indicator(self, basis: Basis, solution: np.array) -> np.ndarray:
        """
        Calculates a numerically integrated error indicator per *face* of the mesh in the given scikit-FEM basis.
        May be arbitrarily wrong for complex load functions.
        Args:
            basis: The basis to calculate the error indicator for
            solution: The solution to the parameterized Poisson equation in the given basis
        Returns: An error/residual per *face* of the mesh

        """
        interior_error = _get_interior_residual(basis=basis, solution=solution[:, 0],
                                                load=self._load_function)
        edge_error = _get_edge_residual(basis=basis, solution=solution[:, 0])
        error_per_element = interior_error + edge_error
        return error_per_element  # error/residual per face

    # wrapper functions for the load function for the finite element assembly
    @property
    def load(self) -> callable:
        return wrapped_partial(wrap_load, evaluate_load=evaluate_load, load=self._load_function)

    @property
    def load_function(self) -> callable:
        return wrapped_partial(evaluate_load, load=self._load_function)

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
        if "load_function" in element_feature_names:
            mesh = basis.mesh
            return np.array([self.load_function(element_midpoints(mesh))]).T  # midpoints of the mesh
        else:
            return None

    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["poisson"]

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_basis(self, basis: Basis) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            basis:

        """
        additional_plots = {
            "load_function": self.load_function_plot(basis=basis),
            "log_load_function": self.log_load_function_plot(basis=basis),
        }
        return additional_plots

    def load_function_plot(self, basis: Basis) -> go.Figure:
        """
        Plot the load function on the current mesh.
        Args:
            basis: Basis to evaluate the function on
        Returns:
        """
        evaluation_function = lambda x: self.load_function(np.stack((x[0], x[1]), axis=-1))
        # evaluation function is load function of poisson equation
        return self._evaluation_function_plot(evaluation_function=evaluation_function,
                                              basis=basis,
                                              title="Load function")

    def log_load_function_plot(self, basis: Basis) -> go.Figure:
        """
        Plot the log of the load function on the current mesh.
        Args:
            basis: Basis to evaluate the function on
        Returns:

        """
        evaluation_function = lambda x: np.maximum(
            np.log(self.load_function(np.stack((x[0], x[1]), axis=-1)) + 1.0e-12), -10)
        # evaluation function is log of load function of poisson equation
        return self._evaluation_function_plot(evaluation_function=evaluation_function,
                                              basis=basis,
                                              title="Log load function")

    def _evaluation_function_plot(self, evaluation_function: Callable, basis: Basis, title: str) -> go.Figure:
        """
        Plot the evaluation function on the current mesh.
        Args:
            evaluation_function: Function to evaluate on the mesh
            basis: Basis to evaluate the function on
            title: Title of the plot
        Returns:

        """
        from modules.swarm_environments.mesh.mesh_refinement.mesh_refinement_visualization import \
            get_evaluation_heatmap_from_basis
        contour_traces = get_evaluation_heatmap_from_basis(basis=basis,
                                                           evaluation_function=evaluation_function,
                                                           normalize_by_element_area=False)

        layout = get_layout(boundary=self._boundary,
                            title=title)
        evaluation_function_plot = go.Figure(data=contour_traces,
                                             layout=layout)
        return evaluation_function_plot
