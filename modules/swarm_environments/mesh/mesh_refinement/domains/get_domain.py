import numpy as np

from modules.swarm_environments.mesh.mesh_refinement.domains.abstract_domain import AbstractDomain
from typing import Dict, Any, Union


def get_domain(*, domain_config: Dict[Union[str, int], Any], random_state: np.random.RandomState) -> AbstractDomain:
    """
    Builds and returns a domain class.
    Args:
        domain_config: Config containing additional details about the domain. Depends on the domain. Contains
            fixed_domain: Whether to use a fixed target domain or sample a random one
            domain_type: The type of domain to use.

        random_state: The RandomState to use to draw the domain in the __init__() call

    Returns: Some domain class that inherits from AbstractDomain.

    """
    fixed_domain = domain_config.get("fixed_domain")
    domain_type = domain_config.get("domain_type").lower()
    if domain_type == "convex_polygon":
        from modules.swarm_environments.mesh.mesh_refinement.domains.convex_polygon import ConvexPolygon
        domain = ConvexPolygon
    elif domain_type == "trapezoid":
        from modules.swarm_environments.mesh.mesh_refinement.domains.trapezoid import Trapezoid
        domain = Trapezoid
    elif domain_type == "bottleneck":
        from modules.swarm_environments.mesh.mesh_refinement.domains.bottleneck import Bottleneck
        domain = Bottleneck
    elif domain_type == "trapezoid_hole":
        from modules.swarm_environments.mesh.mesh_refinement.domains.trapezoid_hole import TrapezoidHole
        domain = TrapezoidHole
    elif domain_type in ["square_hole", "symmetric_hole"]:
        from modules.swarm_environments.mesh.mesh_refinement.domains.square_hole import SquareHole
        domain = SquareHole
    elif domain_type in ["lshape", "lshaped", "l_shaped", "l_shape", "l-shaped", "l-shape"]:
        from modules.swarm_environments.mesh.mesh_refinement.domains.l_shape import LShape
        domain = LShape
    elif domain_type in ["big_square", "small_spiral", "large_spiral"]:
        from modules.swarm_environments.mesh.mesh_refinement.domains.big_domain import BigDomain
        domain = BigDomain
    else:
        from modules.swarm_environments.mesh.mesh_refinement.domains.simple_domain import SimpleDomain
        domain = SimpleDomain
    domain_instance = domain(domain_description_config=domain_config,
                             fixed_domain=fixed_domain,
                             random_state=random_state)
    return domain_instance
