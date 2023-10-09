import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from modules.swarm_environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1
from typing import Dict, Any, Union


class ConvexPolygon(AbstractDomain):
    """
    A class of convex polygonal meshes.
    """

    def __init__(self, domain_description_config: Dict[Union[str, int], Any], fixed_domain: bool,
                 random_state: np.random.RandomState):
        """
        Initializes the domain of the pde to solve with the given finite element method
        Args:
            domain_description_config: Config containing additional details about the domain. Depends on the domain
            fixed_domain: Whether to use a fixed target domain. If True, a deterministic domain will be used.
            Else, a random one will be drawn
            random_state: The RandomState to use to draw the domain
        """
        center = np.array([0.5, 0.5])
        radius = 0.4

        self.num_boundary_nodes = domain_description_config.get("num_boundary_nodes")
        self.maximum_distortion = domain_description_config.get("maximum_distortion", 0.1)

        point_angles = np.cumsum(np.repeat(2 * np.pi / self.num_boundary_nodes, self.num_boundary_nodes))
        x_positions = center[0] + radius * np.cos(point_angles)
        y_positions = center[1] + radius * np.sin(point_angles)
        self._mean_positions = np.vstack((x_positions, y_positions)).T

        super().__init__(domain_description_config=domain_description_config,
                         fixed_domain=fixed_domain,
                         random_state=random_state)

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Reset the domain and create a new initial mesh.
        This method is only called once at the start of iff self.fixed_domain = False.

        A new domain is always drawn from a distribution specified by the config.

        This method is called by the environment when the reset() method is called.

        Returns: The boundary mesh of the new domain, i.e., the simplest mesh that describes the geometry of the domain


        """
        boundary_nodes = self._mean_positions + self._random_state.uniform(low=-self.maximum_distortion,
                                                                           high=self.maximum_distortion,
                                                                           size=(self.num_boundary_nodes, 2))

        initial_mesh = ExtendedMeshTri1.init_convex_polygon(max_element_volume=self.max_initial_element_volume,
                                                            boundary_nodes=boundary_nodes,
                                                            initial_meshing_method=self.initial_meshing_method)
        return initial_mesh
