import numpy as np
from skfem import MeshTri1

from src.environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from src.environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1
from util.types import *


class SimpleDomain(AbstractDomain):
    """
    An umbrella class for simple geometries that are mainly used for testing purposes.
    """

    def __init__(self, domain_description_config: ConfigDict, fixed_domain: bool, random_state: np.random.RandomState):
        """
        Initializes the domain of the pde to solve with the given finite element method
        Args:
            domain_description_config: Config containing additional details about the domain. Depends on the domain
            fixed_domain: Whether to use a fixed target domain. If True, a deterministic domain will be used.
            Else, a random one will be drawn
            random_state: The RandomState to use to draw the domain
        """
        domain_type = domain_description_config["domain_type"].upper()

        if domain_type == "SQ_SYMMETRIC":
            initial_mesh_function = ExtendedMeshTri1.init_sqsymmetric
        elif domain_type in ["SQUARE", "SYMMETRIC"]:
            initial_mesh_function = ExtendedMeshTri1.init_symmetric
        elif domain_type == "HEXAGON":
            initial_mesh_function = ExtendedMeshTri1.init_hexagon
        elif domain_type == "OCTAGON":
            initial_mesh_function = ExtendedMeshTri1.init_octagon
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")

        self._initial_mesh_function = partial(initial_mesh_function)
        super().__init__(domain_description_config=domain_description_config,
                         fixed_domain=fixed_domain,
                         random_state=random_state)

    def _get_initial_mesh(self) -> MeshTri1:
        """
        Reset the domain and create a new initial mesh. For SimpleDomain, this is a deterministic mesh, so we can
        just pull this deterministic mesh from the self._initial_mesh buffer/cache.
        Returns: The boundary mesh of the new domain, i.e., the simplest mesh that describes the geometry of the domain

        """
        return self._initial_mesh_function(max_element_volume=self.max_initial_element_volume,
                                           initial_meshing_method=self.initial_meshing_method)
