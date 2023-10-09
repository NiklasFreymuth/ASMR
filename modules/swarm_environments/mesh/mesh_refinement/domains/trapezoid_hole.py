from typing import Dict, Any, Union

import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from modules.swarm_environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1


class TrapezoidHole(AbstractDomain):
    """
    A quadrilateral geometry/domain. Specified by its 4 boundaries.
    Will always have its left boundary at x=0 and its right boundary at x=1
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
        super().__init__(domain_description_config=domain_description_config,
                         fixed_domain=fixed_domain,
                         random_state=random_state)

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Create an initial mesh

        Returns: The initial mesh on the sampled domain

        """
        boundary_nodes = np.array([[0.0, 0.0], [0.0, 1.0],
                                   [1.0, 1.0], [1.0, 0.0]])
        # used calculate the slope of the lower and upper boundary
        self._boundary_nodes = boundary_nodes

        # calculate a rhombus centered on each of the random points with length 0.1
        # the rhombus is defined by the 4 points (x0, y0-0.075), (x0 + 0.15, y0), (x0, y0 + 0.075), (x0 - 0.15, y0)
        if self.fixed_domain:
            hole_centers_x = np.array([0.3, 0.5, 0.7])

        else:
            hole_centers_x = self._random_state.uniform(low=0.3, high=0.7, size=(3,))
        hole_centers_y = np.array([0.2, 0.5, 0.8])
        # stack the x and y coordinates to get the centers of the holes
        hole_centers = np.stack((hole_centers_x, hole_centers_y), axis=1)

        holes = []
        for random_point in hole_centers:
            holes.append(np.array([[random_point[0], random_point[1] - 0.1],
                                   [random_point[0] + 0.2, random_point[1]],
                                   [random_point[0], random_point[1] + 0.1],
                                   [random_point[0] - 0.2, random_point[1]]]))

        initial_mesh = ExtendedMeshTri1.init_trapezoid_hole(holes=holes,
                                                            boundary_nodes=boundary_nodes,
                                                            max_element_volume=self.max_initial_element_volume,
                                                            initial_meshing_method=self.initial_meshing_method)
        return initial_mesh

    @property
    def boundary_nodes(self):
        return self._boundary_nodes
