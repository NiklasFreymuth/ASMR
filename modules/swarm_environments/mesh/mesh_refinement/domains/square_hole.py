import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from modules.swarm_environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1
from typing import Dict, Any, Union


class SquareHole(AbstractDomain):
    """
    A class of triangular meshes for quadratic geometries with a quadratic hole in them.
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
        self._mean_hole_position = np.array([0.5, 0.5])
        self._mean_hole_size = domain_description_config.get("mean_hole_size")
        self.maximum_size_distortion = domain_description_config.get("maximum_size_distortion")
        self.maximum_position_distortion = domain_description_config.get("maximum_position_distortion")

        # assertions to make sure that the domain is not violated
        minimum_hole_size = self._mean_hole_size - self.maximum_size_distortion
        assert minimum_hole_size > 1.0e-2, f"Can not produce holes with negative minimum size, " \
                                           f"given {minimum_hole_size}"
        maximum_hole_position = self._mean_hole_position[0] + self.maximum_position_distortion + \
                                (self._mean_hole_size + self.maximum_size_distortion) / 2
        assert maximum_hole_position < 1 - 1.0e-2, \
            f"Can not produce holes with maximum size of " \
            f"{maximum_hole_position} " \
            f"that are not contained in the domain [0,1]^2"

        self._hole_boundary = None

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
        if self.fixed_domain:
            hole_position = self._mean_hole_position - self._mean_hole_size / 2
            # Set hole position to lower left corner
            hole_size = np.array([self._mean_hole_size,
                                  self._mean_hole_size])
        else:
            hole_position = self._mean_hole_position + self._random_state.uniform(low=-self.maximum_position_distortion,
                                                                                  high=self.maximum_position_distortion,
                                                                                  size=2)
            hole_size = self._mean_hole_size + self._random_state.uniform(low=-self.maximum_size_distortion,
                                                                          high=self.maximum_size_distortion,
                                                                          size=2)

            hole_position = hole_position - hole_size / 2
            # Set hole position to lower left corner

        initial_mesh = ExtendedMeshTri1.init_square_hole(max_element_volume=self.max_initial_element_volume,
                                                         hole_position=hole_position,
                                                         hole_size=hole_size,
                                                         initial_meshing_method=self.initial_meshing_method)

        self._hole_boundary = np.concatenate((hole_position, hole_position + hole_size), axis=0)
        return initial_mesh

    @property
    def hole_boundary(self) -> np.array:
        return self._hole_boundary
