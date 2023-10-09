from typing import Dict, Any, Union

import numpy as np
from skfem import MeshTri1

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from modules.swarm_environments.mesh.mesh_refinement.domains.extended_mesh_tri1 import ExtendedMeshTri1


class BigDomain(AbstractDomain):
    """
    A class of triangular meshes for quadratic geometries with a quadratic hole in them.
    """

    def __init__(self, domain_description_config: Dict[Union[str, int], Any], fixed_domain: bool,
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
        self._domain_type = domain_description_config["domain_type"]
        boundary = domain_description_config["boundary"]
        assert boundary[0] == boundary[1] and boundary[2] == boundary[3], "The domain must be quadratic"
        self._scale = boundary[2] - boundary[0]

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
        return ExtendedMeshTri1.init_big_domain(max_element_volume=self.max_initial_element_volume,
                                                initial_meshing_method=self.initial_meshing_method,
                                                scale=self._scale,
                                                domain_type=self._domain_type)
