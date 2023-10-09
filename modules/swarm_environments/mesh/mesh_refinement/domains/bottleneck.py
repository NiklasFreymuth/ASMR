from typing import Dict, Any, Union

import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from modules.swarm_environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1


class Bottleneck(AbstractDomain):
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
        self.maximum_distortion = domain_description_config.get("maximum_distortion")
        assert 0 <= self.maximum_distortion < 0.5, \
            f"Maximum distortion must be in [0, 0.5), given '{self.maximum_distortion}'"

        super().__init__(domain_description_config=domain_description_config,
                         fixed_domain=fixed_domain,
                         random_state=random_state)

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Create an initial mesh

        Returns: The initial mesh on the sampled domain

        """

        if self.fixed_domain:
            boundary_nodes = np.array([[0.0, 0.0 + self.maximum_distortion],
                                       [0.0, 1.0],
                                       [0.5, 0.6],
                                       [1.0, 1.0 - self.maximum_distortion],
                                       [1.0, 0.0],
                                       [0.5, 0.4]],
                                      )

        else:
            boundary_nodes = np.array([[0.0, 0.0], [0.0, 1.0],
                                       [1.0, 1.0], [1.0, 0.0]])

            # to get interesting trapezoids in [0,1]^2 from uniform distributions, we randomly fix 2 of the 4 corners
            # and distort only the other two. Additionally, the distortion is drawn from U(0, maximum_distortion) and
            # the sign is chosen according to the corner's position in the domain
            distortions = self._random_state.uniform(low=0,
                                                     high=self.maximum_distortion,
                                                     size=(2,))
            corner_indices = self._random_state.choice(4, size=2, replace=False)
            distortions *= -np.sign(boundary_nodes[corner_indices, 1] - 0.5)
            # distort
            boundary_nodes[corner_indices, 1] += distortions

            # "clip" the corners to the boundary, i.e., assure that there is a corner at y=0 and one at y=1.
            # this effectively means that if both corner indices are selected the "upper" (or equivalently "lower") 
            # part of the trapezoid, then only the larger of the two distortions is kept.
            boundary_nodes[np.argmax(boundary_nodes[:, 1]), 1] = 1.0
            boundary_nodes[np.argmin(boundary_nodes[:, 1]), 1] = 0.0

            bottleneck_positions = np.array([[0.5, 0.6], [0.5, 0.4]])
            bottleneck_offsets = self._random_state.uniform(low=-0.2,
                                                            high=0.2,
                                                            size=(2,))
            bottleneck_positions[:, 0] += bottleneck_offsets[0]
            bottleneck_positions[:, 1] += bottleneck_offsets[1]

            boundary_nodes = np.array((boundary_nodes[0],
                                       boundary_nodes[1],
                                       bottleneck_positions[0],
                                       boundary_nodes[2],
                                       boundary_nodes[3],
                                       bottleneck_positions[1],
                                       ))

        # used calculate the slope of the lower and upper boundary
        self._boundary_nodes = boundary_nodes

        initial_mesh = ExtendedMeshTri1.init_polygon(
            max_element_volume=self.max_initial_element_volume,
            boundary_nodes=boundary_nodes,
            initial_meshing_method=self.initial_meshing_method)
        print("@@ Mesh:", initial_mesh)
        return initial_mesh

    @property
    def boundary_nodes(self):
        return self._boundary_nodes
