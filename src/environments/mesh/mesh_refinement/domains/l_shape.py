import numpy as np
from skfem import MeshTri1

from src.environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from src.environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1
from util.types import *


class LShape(AbstractDomain):
    """
    A class of triangular meshes for quadratic geometries with a quadratic hole in them.
    """

    def __init__(self, domain_description_config: ConfigDict, fixed_domain: bool,
                 random_state: np.random.RandomState):
        """
        Initializes the domain.
        Args:
            domain_description_config: Config containing additional details about the domain.
                Depends on the domain
            fixed_domain: Whether to use a fixed domain. If True, the same domain will be used
                throughout.
                If False, a family of geometries will be created and a new domain will be drawn
                from this family whenever the reset() method is called
            random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
                function class
        """
        self._mean_hole_position = np.array([0.5, 0.5])
        self.maximum_position_distortion = domain_description_config.get("maximum_position_distortion", 0.3)
        super().__init__(domain_description_config=domain_description_config, fixed_domain=fixed_domain,
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
            hole_position = self._mean_hole_position
        else:
            offset = self._random_state.uniform(low=-self.maximum_position_distortion,
                                                high=self.maximum_position_distortion,
                                                size=2)
            hole_position = self._mean_hole_position + np.clip(offset, -0.3, 0.45)
        initial_mesh = ExtendedMeshTri1.init_lshaped(max_element_volume=self.max_initial_element_volume,
                                                     hole_position=hole_position,
                                                     initial_meshing_method=self.initial_meshing_method)
        return initial_mesh
