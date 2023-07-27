r"""Heat Diffusion task with an outer boundary with boundary condition Omega_0=0,
and some heat source
"""

import os

import numpy as np
import skfem as fem
from plotly import graph_objects as go
from scipy.sparse.linalg import splu
from skfem import Mesh, Basis, LinearForm
from skfem.models.poisson import laplace, mass

from src.environments.mesh.mesh_refinement.mesh_refinement_util import element_midpoints
from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from util.function import wrapped_partial
from util.types import *

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def heatsource_IE(v, w, current_timestep: int, pos_x: float, pos_y: float, heat: float, t0: int):
    """
    Heat source for the heat diffusion problem with implicit euler integration
    Args:
        v:
        w:
        current_timestep: The current timestep
        pos_x: Position of the heat source in x direction
        pos_y: Position of the heat source in y direction
        heat: Heat of the heat source
        t0: Delay before the heat source starts appearing. In timesteps

    Returns:

    """
    x, y = w.x

    has_heat = (current_timestep >= t0)
    heat = heat * has_heat
    x_distance = np.abs((x - pos_x))
    y_distance = np.abs((y - pos_y))

    f = heat * (1 / np.exp(100 * x_distance)) * (1 / np.exp(100 * y_distance))
    # the 100 scales the radius of the heat source
    return f * v


FINAL_STEP = 10
STEP_SIZE = 0.5  # doubles as the first timestep


class HeatDiffusion(AbstractFiniteElementProblem):
    """
    Heat Diffusion task with an outer boundary with boundary condition Omega_0=0,
    and some heat source that moves from one point on the geometry to another.
    Assumes a convex geometry/problem domain, since the heat source moves along a line between two points on the domain
    """

    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        self.heat = 1000.0

        self.final_timestep = FINAL_STEP
        self.step_size = STEP_SIZE
        self._duration = self.final_timestep - self.step_size

        diffusion_config = fem_config.get("heat_diffusion")
        self._fixed_diffusion = diffusion_config.get("fixed_diffusion")
        self.diffusivity = diffusion_config.get("diffusivity")

        super().__init__(fem_config=fem_config,
                         random_state=random_state)  # also calls reset() and thus _set_pde()

    def _set_pde(self) -> None:
        """
        This function is called to draw a new PDE from a family of available PDEs.
        """
        if self._fixed_diffusion:
            # have heat source move from one corner to the other
            boundary_coordinates = self._domain.initial_mesh.p[:, self._domain.initial_mesh.boundary_nodes()]
            midpoint = np.mean(boundary_coordinates, axis=1)
            self._start_pos = (9 * boundary_coordinates[:, 0] + midpoint) / 10
            self._end_pos = (9 * boundary_coordinates[:, int(boundary_coordinates.shape[1] / 2)] + midpoint) / 10
        else:
            # set start and end position randomly in the domain
            self._start_pos = self._get_valid_point_in_domain()
            self._end_pos = self._get_valid_point_in_domain()

    def _get_valid_point_in_domain(self):
        while True:
            candidate_points = self._random_state.uniform(0.0, 1.0,  # assume a domain in [0,1]^2
                                                          size=(10, 2))  # draw 10 points at once to speed up
            valid_points = self._points_in_domain(candidate_points)
            if len(valid_points) > 0:
                return valid_points[0]

    def _points_in_domain(self, candidate_points: np.array) -> np.array:
        """
        Returns a subset of points that are inside the current domain, i.e., that can be found in the mesh.
        Returns:

        """
        corresponding_elements = self._domain.initial_mesh.element_finder()(x=candidate_points[:, 0],
                                                                            y=candidate_points[:, 1])
        valid_points = candidate_points[corresponding_elements != -1]
        return valid_points

    def add_boundary_conditions_and_create_basis(self, mesh: Mesh) -> Basis:
        """
        Fetches boundary nodes of the given mesh and creates a basis with those boundaries.
        Args:
            mesh: The mesh used to create the basis.
        """
        # Function space
        basis = fem.Basis(mesh, fem.ElementTriP1())
        return basis

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates the solution and returns *?*
        """
        solution = basis.zeros()
        solutions = [solution]
        current_timestep = self.step_size
        previous_solution = solution

        mass_matrix = fem.asm(mass, basis)
        scaled_laplace_term = self.step_size * self.diffusivity * fem.asm(laplace, basis)
        A = mass_matrix + scaled_laplace_term

        interior_nodes = basis.mesh.interior_nodes()
        Ap = fem.penalize(A, I=interior_nodes)
        backsolve_function = splu(Ap.T).solve

        while current_timestep <= self.final_timestep:
            current_solution = self._solve_step(basis=basis,
                                                current_timestep=current_timestep,
                                                previous_solution=previous_solution,
                                                backsolve_function=backsolve_function,
                                                mass_matrix=mass_matrix,
                                                A=A)
            solutions.append(current_solution)
            previous_solution = current_solution
            current_timestep = current_timestep + self.step_size

        solutions = np.array(solutions[1:]).T  # transpose and ignore the first solution, which is all zeros
        return solutions[:, -1]

    def _solve_step(self, basis: Basis, current_timestep: int, previous_solution: np.array,
                    backsolve_function: callable, mass_matrix: np.array, A: np.array) -> np.array:
        """
        Solves the heat diffusion problem for one timestep.
        Args:
            basis: The basis used to solve the problem
            current_timestep: The current timestep
            previous_solution: The solution of the previous timestep

        Returns: The solution of the current timestep

        """

        heatsource_function = self._heatsource_function(current_timestep=current_timestep,
                                                        duration=self._duration)
        H_1 = fem.asm(LinearForm(heatsource_function), basis)
        M_0 = mass_matrix @ np.resize(previous_solution, mass_matrix.shape[0])
        B = M_0 + self.step_size * H_1
        interior_nodes = basis.mesh.interior_nodes()
        _, Bp = fem.penalize(A, B, I=interior_nodes)

        solution = backsolve_function(Bp)
        return solution

    def _heatsource_function(self, current_timestep, duration: float):
        # calculate position of the heat source at the current timestep assuming uniform movement
        normalized_timestep = (self.final_timestep - current_timestep) / duration
        pos_x = normalized_timestep * (self._start_pos[0] - self._end_pos[0]) + self._end_pos[0]
        pos_y = normalized_timestep * (self._start_pos[1] - self._end_pos[1]) + self._end_pos[1]

        # calculate heat source function from position
        heatsource_function = wrapped_partial(heatsource_IE,
                                              current_timestep=current_timestep + self.step_size,  # integrate from t+1
                                              pos_x=pos_x,
                                              pos_y=pos_y,
                                              heat=self.heat,
                                              t0=self.step_size)
        return heatsource_function

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
        features = []
        if "distance_to_start" in element_feature_names:
            mesh = basis.mesh
            start_distances = np.linalg.norm(element_midpoints(mesh) - self._start_pos, axis=1).T
            features.append(start_distances)

        if "distance_to_end" in element_feature_names:
            mesh = basis.mesh
            end_distances = np.linalg.norm(element_midpoints(mesh) - self._end_pos, axis=1).T
            features.append(end_distances)

        if len(features) > 0:
            return np.array(features).T
        else:
            return None

    def global_features(self, basis, global_feature_names) -> Optional[np.array]:
        return None

    @staticmethod
    def solution_dimension_names() -> List[str]:
        return ["final_step"]

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_basis(self, basis) -> Dict[str, go.Figure]:
        """
        """
        # no additional plots for now to keep the plots simple. Could consider adding a plot of the heat source position
        # over time, or a plot of the distance to the first/last heat source
        return {}

