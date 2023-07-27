r"""Linear elasticity.
This example solves the linear elasticity problem using trilinear elements.  The
weak form of the linear elasticity problem is defined in
:func:`skfem.models.elasticity.linear_elasticity`.

combination of https://github.com/kinnala/scikit-fem/blob/7.0.1/docs/examples/ex04.py
and https://github.com/kinnala/scikit-fem/blob/7.0.1/docs/examples/ex11.py

"""

import os

import numpy as np
from plotly import graph_objects as go
from skfem import Mesh, Basis, asm, solve, condense, ElementTriP1, ElementVector
from skfem.helpers import sym_grad
from skfem.models.elasticity import linear_stress, linear_elasticity, lame_parameters

from src.environments.mesh.mesh_refinement.mesh_refinement_visualization import scalar_per_element_plot
from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from util.types import *

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LinearElasticity(AbstractFiniteElementProblem):
    def __init__(self, *,
                 fem_config: ConfigDict,
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Args:
            fem_config: A dictionary containing the configuration for the finite element method.
            random_state: A random state to use for reproducibility.

        """
        # (fixed) material parameters.
        # Since these parameters describe a certain material, it does not make sense to sample them from a distribution.
        youngs_modulus = 1.0  # 1000
        self._poisson_ratio = 0.3  # 0.3
        lame_lambda, lame_mu = lame_parameters(youngs_modulus, self._poisson_ratio)
        self.stiffness_matrix = linear_elasticity(lame_lambda, lame_mu)
        self.material_model = linear_stress(lame_lambda, lame_mu)  # Linear-elastic stress-strain relationship.

        # displacement of the right boundary
        linear_elasticity_config = fem_config.get("linear_elasticity")

        # whether to fix the displacement of the right boundary to a certain value
        self._fixed_displacement = linear_elasticity_config.get("fixed_displacement")
        # define a range of available displacement magnitudes from the range specified in the config
        self._boundary_displacement_magnitude_range = np.array(
            [linear_elasticity_config.get("lower_displacement_magnitude"),
             linear_elasticity_config.get("upper_displacement_magnitude")])

        self._boundary_x_displacement = None
        self._boundary_y_displacement = None

        # weights of the different solution dimensions in the error integration
        self._relative_stress_weight = linear_elasticity_config.get("relative_stress_weight")
        assert 0 <= self._relative_stress_weight <= 1, \
            f"Relative stress weight must be in [0, 1], given {self._relative_stress_weight}"

        # elements and meshes
        # Define first order triangles and define a 2d variable on it (u1 and u2)
        self.element_vector = ElementVector(ElementTriP1(), dim=2)  # define a 2d variable on the triangle (u^1, u^2)
        # solve for displacements in x and y, which are 2d vectors
        # this is the result of 2 coupled differential equations, one for each component of the displacement vector
        # We could also take the l2 norm of the vector if we want a scalar value for the displacement
        self.element_tensor = ElementVector(self.element_vector, dim=2)  # define a 2x2 tensor per element

        self._displacement = None  # save the displacement field for plotting utility

        super().__init__(fem_config=fem_config,
                         random_state=random_state)  # also calls reset() and thus _set_pde()

    def _set_pde(self) -> None:
        """
        Draw a new PDE instance from the available family of plate bending PDEs.

        """
        if self._fixed_displacement:
            displacement_direction = np.pi / 4  # value in [0, 2pi], default is pi/4 = 45 degrees
            displacement_magnitude = np.mean(self._boundary_displacement_magnitude_range)
        else:
            displacement_direction = self._random_state.uniform(low=0, high=np.pi,
                                                                size=1).item()
            # draw value in [0, pi] to ensure that the displacement stretches rather than compresses the mesh
            displacement_magnitude = self._random_state.uniform(low=self._boundary_displacement_magnitude_range[0],
                                                                high=self._boundary_displacement_magnitude_range[1],
                                                                size=1).item()
        self._boundary_x_displacement = displacement_magnitude * np.sin(displacement_direction)
        self._boundary_y_displacement = displacement_magnitude * np.cos(displacement_direction)

    def add_boundary_conditions_and_create_basis(self, mesh: Mesh) -> Basis:
        mesh = mesh.with_boundaries(  # Label boundaries
            {
                "right": lambda x: x[0] == 1,
                "left": lambda x: x[0] == 0,
            }
        )
        basis = Basis(mesh, self.element_vector)
        return basis

    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        """
        Calculates a solution for the linear elasticity equations on the given basis. The solution is
        calculated for every node/vertex of the underlying mesh, and the way it is calculated depends on the element
        used in the basis.
        For example, ElementTriP1() elements will draw 3 quadrature points for each element that lie in the middle of the
        edge between the barycenter of the element and its spanning nodes, and then linearly interpolate based on those
        elements.
        Args:
            cache: Whether to cache the solution for plotting purposes.

        Returns: An array (num_vertices, 2), where every entry corresponds to a vector of the norm of the stress
        at the corresponding vertex, and the magnitude of the displacement at the corresponding vertex.

        """
        # define boundary conditions
        # Get all degrees of freedom and set appropriate entry to prescribed BCs.
        displacement = basis.zeros()
        displacement[basis.get_dofs({"right"}).nodal['u^1']] = self._boundary_x_displacement
        displacement[basis.get_dofs({"right"}).nodal['u^2']] = self._boundary_y_displacement
        displacement[basis.get_dofs({"left"}).nodal['u^1']] = 0.0
        displacement[basis.get_dofs({"left"}).nodal['u^2']] = 0.0

        # Get remaining degrees of freedom
        included_degrees_of_freedom = basis.complement_dofs(basis.get_dofs({"left", "right"}))

        # Assemble lhs of the linear system
        K = asm(self.stiffness_matrix, basis)

        # Solve the problem
        displacement = solve(*condense(K, b=basis.zeros(), x=displacement, I=included_degrees_of_freedom))

        # Postprocess, compute stress
        interpolated_displacement = basis.interpolate(displacement)
        # we interpolate the displacement per node (linearly) to the quadrature points of the mesh
        # u.shape -> (2 x #vertices ,). ui.shape -> (2, #elements, #quadrature points).
        # I.e., this transforms a flattened [x,y] solution vector per node to a solution per element of shape
        #  ([x,y], #elements, #quadrature points)

        # this builds a 2x2 tensor basis per element
        sgb = basis.with_element(self.element_tensor)

        # sym_grad(ui) is the distortion tensor, same dimension as the stress tensor
        # the resulting stress tensor is of shape (2,2, #elements, #quadrature points)
        stress_tensor = self.material_model(
            sym_grad(interpolated_displacement))
        sigma = sgb.project(stress_tensor)
        # one stress value per entry of the tensor

        # calculate von-mises stress
        s = {
            (0, 0): sigma[sgb.nodal_dofs[0]],  # tension in x direction
            (0, 1): sigma[sgb.nodal_dofs[1]],  # shear
            (1, 0): sigma[sgb.nodal_dofs[2]],  # shear, symmetric to (0, 1)
            (1, 1): sigma[sgb.nodal_dofs[3]],  # tension in y direction
            (2, 2): self._poisson_ratio * (sigma[sgb.nodal_dofs[0]] + sigma[sgb.nodal_dofs[3]])
        }
        stress_norm = np.sqrt(.5 * ((s[0, 0] - s[1, 1]) ** 2 +
                                    (s[1, 1] - s[2, 2]) ** 2 +
                                    (s[2, 2] - s[0, 0]) ** 2 +
                                    6. * s[0, 1] ** 2))

        displacement_norm = np.linalg.norm(displacement[basis.nodal_dofs], axis=0)

        if cache:
            self._displacement = displacement
        else:
            self._displacement = None  # clear cache
        solution = np.hstack((stress_norm[:, None], displacement_norm[:, None]))
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
        if "x_displacement" in global_feature_names:
            features.append(self._boundary_x_displacement)
        if "y_displacement" in global_feature_names:
            features.append(self._boundary_y_displacement)
        return np.array(features) if len(features) > 0 else None

    @staticmethod
    def solution_dimension_names() -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return ["stress_norm", "displacement_norm"]

    @property
    def solution_dimension_weights(self) -> np.array:
        """
        Returns a list of weights for the solution dimensions. This is used to weight the solution dimensions
        when calculating the error.
        Returns: A list of weights for the solution dimensions that sums up to one

        """
        dimension_weights = np.array([self._relative_stress_weight, 1.0 - self._relative_stress_weight])
        return dimension_weights

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
        if self._displacement is not None:
            mesh = basis.mesh
            deformed_mesh = mesh.translated(self._displacement[basis.nodal_dofs])  # deformed mesh

            vertex_displacements = self._displacement[basis.nodal_dofs]
            # displacement norm
            displacement_norm_per_node = np.linalg.norm(vertex_displacements, axis=0)
            mesh_displacement_norm = np.mean(displacement_norm_per_node[deformed_mesh.t], axis=0)
            norm_displacement_plot = scalar_per_element_plot(mesh=deformed_mesh,
                                                             scalar_per_element=mesh_displacement_norm,
                                                             title="Displacement magnitude")
            additional_plots["displacement_norm"] = norm_displacement_plot

            # displacement x
            mesh_x_displacement = np.mean(vertex_displacements[0][deformed_mesh.t], axis=0)
            x_displacement_plot = scalar_per_element_plot(mesh=deformed_mesh, scalar_per_element=mesh_x_displacement,
                                                          title="Displacement x")
            additional_plots["displacement_x"] = x_displacement_plot

            # displacement y
            mesh_y_displacement = np.mean(vertex_displacements[1][deformed_mesh.t], axis=0)
            y_displacement_plot = scalar_per_element_plot(mesh=deformed_mesh, scalar_per_element=mesh_y_displacement,
                                                          title="Displacement y")
            additional_plots["displacement_y"] = y_displacement_plot
        return additional_plots
