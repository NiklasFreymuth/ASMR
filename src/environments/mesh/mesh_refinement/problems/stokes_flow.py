r"""Fluid Flow task.
Solves the Poiseuille flow problem (?) using the Navier-Stokes equations.
Here, we are interested in the steady state solution, i.e., the solution that is independent of time.
As such, the domain must always be a trapezoid, and the inlet velocity is parabolic.

We compute both the velocity and pressure fields, and weight their contributions to the reward function.
Further, we use quadratic elements for the velocity field, and linear elements for the pressure field.
"""

import os

import numpy as np
from plotly import graph_objects as go
from skfem import ElementTriP1, ElementTriP2, FacetBasis, Basis, ElementVector, bmat, solve, \
    asm, condense, DiscreteField, Mesh
from skfem.models.general import divergence, rot
from skfem.models.poisson import vector_laplace, laplace

from src.environments.mesh.mesh_refinement.domains.trapezoid import Trapezoid
from src.environments.mesh.mesh_refinement.mesh_refinement_visualization import scalar_per_element_plot
from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from util.types import *

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parabolic_inlet_profile(inlet_coordinates: DiscreteField, parabolic_velocity: float,
                            minimum_y_value: float, maximum_y_value: float,
                            phase: float = np.pi) -> np.array:
    """
    Define the parabolic velocity profile at the inlet. Higher values of the velocity will result in a more
    turbulent flow, and thus a longer time to reach steady state.
    Args:
        inlet_coordinates:  The inlet coordinates. A discrete field of shape ([x,y], inlet_facets, quadrature_points)
        parabolic_velocity: The maximum velocity at the inlet
        minimum_y_value:    The minimum y value of the inlet
        maximum_y_value:    The maximum y value of the inlet
        phase: The phase of the sinusoidal component of the inlet velocity profile. Should be 0 or pi, because the
          velocity profile on the boundaries should be 0.

    Returns: The plane Poiseuille parabolic inlet profile, i.e.,
    a discrete field of shape ([u,v], inlet_facets, quadrature_points) where u and v are the x and y components of the
    velocity respectively, and v = 0.
    The equation for this profile reads u = velocity * y * (1 - y) + sin(2 * pi * y)
    The first term is the parabolic profile, the second term is the sinusoidal component. The phase of the sinusoidal
    component is given by the phase argument.

    """
    # normalize inlet coordinates to adapt to inlets that are not in x=[0,1]

    x_inlet_coordinates = inlet_coordinates[0]
    y_inlet_coordinates = inlet_coordinates[1]
    y_inlet_coordinates = (y_inlet_coordinates - minimum_y_value) / (maximum_y_value - minimum_y_value)

    velocity_profile = parabolic_velocity * y_inlet_coordinates * (1. - y_inlet_coordinates) + \
                       np.sin(phase + 2 * np.pi * y_inlet_coordinates)
    return np.stack([velocity_profile, np.zeros_like(x_inlet_coordinates)])


def solve_velocity_and_pressure(velocity_basis: Basis, parabolic_velocity: float,
                                minimum_y_value: float, maximum_y_value: float,
                                phase: float = np.pi) -> Tuple[np.array, np.array]:
    """
    Solve the Navier-Stokes equations for the velocity and pressure fields

    Args:
        velocity_basis: The basis for the velocity field
        parabolic_velocity: The maximum velocity at the inlet
        minimum_y_value:    The minimum y value of the inlet
        maximum_y_value:    The maximum y value of the inlet
        phase: The phase of the sinusoidal component of the inlet velocity profile. Should be 0 or pi, because the
            velocity profile on the boundaries should be 0.

    Returns: The velocity and pressure fields as np.arrays of shape
    (velocity_basis.dofs,) and (pressure_basis.dofs,) respectively

    """
    mesh = velocity_basis.mesh
    element = velocity_basis.elem

    # Create the pressure basis, which is a scalar basis on the same mesh as the velocity basis, but with elements
    # of a lower order
    pressure_basis = velocity_basis.with_element(ElementTriP1())

    # Assemble the block matrix K
    A = asm(vector_laplace, velocity_basis)
    B = -asm(divergence, velocity_basis, pressure_basis)
    K = bmat([[A, B.T], [B, None]], 'csr')
    # # block matrix with the following structure
    # # | A  B^T |
    # # | B   0  |
    # # CSR stands for Compressed Sparse Row format

    # Define the inlet basis
    inlet_basis = FacetBasis(mesh, element, facets=mesh.boundaries['inlet'])

    # Solve the system of equations
    parabolic_ = partial(parabolic_inlet_profile, parabolic_velocity=parabolic_velocity, phase=phase,
                         minimum_y_value=minimum_y_value, maximum_y_value=maximum_y_value)
    velocity_inlet_basis = inlet_basis.project(parabolic_)

    uvp = np.hstack((
        velocity_inlet_basis,
        pressure_basis.zeros(),
    ))

    boundary_dofs = velocity_basis.get_dofs(['inlet', 'ceiling', 'floor'])  # boundary except for the outlet
    uvp = solve(*condense(K, x=uvp, D=boundary_dofs))
    # Separate the velocity and pressure components
    velocity, pressure = np.split(uvp, [A.shape[0]])
    return velocity, pressure


def get_stream_function(velocity_basis: Basis, velocity: np.array) -> np.array:
    """
    Calculate the vorticity field from the velocity field
    Args:
        velocity_basis: The basis for the velocity field
        velocity: The velocity field

    Returns: The stream function on the nodal points of the mesh

    """
    # Define the basis for the stream function
    stream_basis = velocity_basis.with_element(ElementTriP1())

    A = asm(laplace, stream_basis)
    vorticity = asm(rot, stream_basis, w=velocity_basis.interpolate(velocity))
    psi = solve(*condense(A, vorticity, D=stream_basis.get_dofs('floor')))
    # psi is the stream function, which is the solution to the Laplace equation of the vorticity field
    return psi[stream_basis.nodal_dofs].squeeze()


class StokesFlow(AbstractFiniteElementProblem):
    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Args:
            fem_config: A dictionary containing the configuration for the finite element method.
            random_state: A random state to use for reproducibility.

        """
        # displacement of the right boundary
        stokes_flow_config = fem_config.get("stokes_flow")

        self._fixed_velocity = stokes_flow_config.get("fixed_velocity")
        # define a range of available velocities from the range specified in the config
        self._parabolic_velocity_range = np.array([stokes_flow_config.get("lower_velocity"),
                                                   stokes_flow_config.get("upper_velocity")])
        self._parabolic_velocity = None

        # elements and meshes
        self.velocity_element_vector = ElementVector(ElementTriP2(), dim=2)
        # the element for the velocity needs to be an order of magnitude higher than the element for the pressure.
        # This is known as a Taylor-Hood element, and is necessary for the stability of the Navier-Stokes equations
        # See also: https://www.math.colostate.edu/~bangerth/videos/676/slides.33.25.pdf)

        # cache
        self._vertex_velocities = None

        self._upper_slope = None
        self._lower_slope = None
        self._lower_left_y = None
        self._upper_left_y = None

        super().__init__(fem_config=fem_config,
                         random_state=random_state)  # also calls reset() and thus _set_pde()

    def _set_pde(self) -> None:
        """
        Draw a new PDE instance from the available family of plate bending PDEs.

        """
        if self._fixed_velocity:
            self._parabolic_velocity = np.mean(self._parabolic_velocity_range)
        else:
            # log-uniform sampling of the velocity, since changes in the velocity are more significant at low velocities
            log_velocity_range = np.log(self._parabolic_velocity_range)
            self._parabolic_velocity = np.exp(self._random_state.uniform(low=log_velocity_range[0],
                                                                         high=log_velocity_range[1],
                                                                         size=1).item())
            # assert the domain and pre-compute some values for the boundary conditions
        assert isinstance(self._domain, Trapezoid), f"Fluid flow task currently only defined " \
                                                    f"for Trapezoid domain, given '{type(self._domain)}'"
        boundary_nodes = self._domain.boundary_nodes  # normalized corners of the trapezoid domain

        self._upper_slope = (boundary_nodes[2, 1] - boundary_nodes[1, 1]) / \
                            (boundary_nodes[2, 0] - boundary_nodes[1, 0])
        self._lower_slope = (boundary_nodes[3, 1] - boundary_nodes[0, 1]) / \
                            (boundary_nodes[3, 0] - boundary_nodes[0, 0])
        self._lower_left_y = boundary_nodes[0, 1]
        self._upper_left_y = boundary_nodes[1, 1]

    def add_boundary_conditions_and_create_basis(self, mesh: Mesh) -> Basis:
        boundaries = {
            "inlet": lambda x: x[0] == 0,
            "ceiling": lambda x: x[1] - self._upper_left_y >= self._upper_slope * x[0] - 1.0e-10,
            "floor": lambda x: x[1] - self._lower_left_y <= self._lower_slope * x[0] + 1.0e-10
        }
        # add small offset to the boundaries to avoid numerical issues
        mesh = mesh.with_boundaries(boundaries)

        # Define basis with integration order 2 for the velocity field
        basis = Basis(mesh, self.velocity_element_vector, intorder=2)
        return basis

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution for the fluid flow problem.
        Args:
            basis: The basis to use for the solution.
            cache: Whether to cache the solution for plotting purposes.

        Returns: An array (num_vertices, 2), where every entry corresponds to a vector of the norm of the velocity
        at the corresponding vertex, and the pressure at the corresponding vertex.

        """

        velocity, pressure = solve_velocity_and_pressure(velocity_basis=basis,
                                                         parabolic_velocity=self._parabolic_velocity,
                                                         minimum_y_value=self._lower_left_y,
                                                         maximum_y_value=self._upper_left_y)

        vertex_velocity = velocity[basis.nodal_dofs]  # velocity on the vertices, i.e., a solution per vertex
        if cache:
            self._vertex_velocities = vertex_velocity
        else:
            self._vertex_velocities = None

        velocity_norm = np.linalg.norm(vertex_velocity, axis=0)
        return velocity_norm[:, None]

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
        return None

    def global_features(self, basis: Basis, global_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_global_features,) containing the features for the entire mesh.
        Args:
            basis: The basis to use for the calculation
            global_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        """
        features = []
        if "velocity" in global_feature_names:
            features.append(self._parabolic_velocity)
        return np.array(features) if len(features) > 0 else None

    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["velocity_norm"]

    @property
    def inlet_line_segments(self):
        source_facets = self.initial_mesh.facets_satisfying(lambda x: x[0] == 0,  # inlet facets
                                                            boundaries_only=True)
        boundary_node_indices = self.initial_mesh.facets[:, source_facets]
        line_segments = self.initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        return line_segments

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots_from_basis(self, basis) -> Dict[str, go.Figure]:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            basis:

        """
        additional_plots = {}
        if self._vertex_velocities is not None:
            mesh = basis.mesh

            # velocity x
            mesh_x_velocity = np.mean(self._vertex_velocities[0][mesh.t], axis=0)
            x_velocity_plot = scalar_per_element_plot(mesh=mesh, scalar_per_element=mesh_x_velocity,
                                                      title="Velocity x")
            additional_plots["velocity_x"] = x_velocity_plot

            # velocity y
            mesh_y_velocity = np.mean(self._vertex_velocities[1][mesh.t], axis=0)
            y_velocity_plot = scalar_per_element_plot(mesh=mesh, scalar_per_element=mesh_y_velocity,
                                                      title="Velocity y")
            additional_plots["velocity_y"] = y_velocity_plot
        return additional_plots
