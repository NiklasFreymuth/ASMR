import numpy as np

from src.environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import AbstractFiniteElementProblem
from util.types import *


def get_finite_element_problem(*,
                               fem_config: ConfigDict,
                               random_state: np.random.RandomState) -> AbstractFiniteElementProblem:
    """
    Builds and returns a finite element problem class.
    Args:
        fem_config: Config containing additional details about the finite element method. Contains
            poisson: Config containing additional details about the poisson problem. Depends on the problem
            domain: Config containing additional details about the domain. Depends on the domain
        random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
            function class

    Returns: Some domain class that inherits from AbstractDomain.

    """
    pde = get_finite_element_problem_class(fem_config=fem_config)
    return pde(fem_config=fem_config, random_state=random_state)


def get_finite_element_problem_class(*, fem_config: ConfigDict) -> Type[AbstractFiniteElementProblem]:
    """
    Builds and returns a finite element problem class.
    Args:
        fem_config: Config containing additional details about the finite element method.


    Returns: Some domain class that inherits from AbstractDomain.

    """

    pde_type = fem_config.get("pde_type")
    pde_type = pde_type.lower()
    if pde_type == "laplace":
        from src.environments.mesh.mesh_refinement.problems.laplace import Laplace
        pde = Laplace
    elif pde_type == "poisson":
        from src.environments.mesh.mesh_refinement.problems.poisson import Poisson
        pde = Poisson
    elif pde_type == "stokes_flow":
        from src.environments.mesh.mesh_refinement.problems.stokes_flow import StokesFlow
        pde = StokesFlow
    elif pde_type == "linear_elasticity":
        from src.environments.mesh.mesh_refinement.problems.linear_elasticity import LinearElasticity
        pde = LinearElasticity
    elif pde_type == "heat_diffusion":
        from src.environments.mesh.mesh_refinement.problems.heat_diffusion import HeatDiffusion
        pde = HeatDiffusion
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")
    return pde
