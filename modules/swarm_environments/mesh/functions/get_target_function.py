from modules.swarm_environments.mesh.functions.abstract_target_function import AbstractTargetFunction
from typing import Dict, Any, Union
import numpy as np


def get_target_function(*, target_function_name: str, target_function_config: Dict[Union[str, int], Any],
                        boundary: np.array, fixed_target: bool,
                        random_state: np.random.RandomState) -> AbstractTargetFunction:
    """
    Builds and returns a density class.
    Args:
        target_function_name: Name of the target function to retrieve
        target_function_config: Config containing additional details about the target function. Depends on the
            target function
        boundary: 2d-rectangle that defines the boundary that this function should act in
        fixed_target: Whether to use a fixed target function. If True, the same target function will be used
            throughout. If False, a family of target functions will be created and a new target function will be drawn
            from this family whenever the reset() method is called
        random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
            function class

    Returns: Some density/target function that inherits from AbstractTargetFunction.

    """
    target_function_name = target_function_name.lower()
    if target_function_name == "gaussian":
        from modules.swarm_environments.mesh.functions.gaussian_density import GaussianDensity
        target_function = GaussianDensity
    elif target_function_name == "gmm":
        from modules.swarm_environments.mesh.functions.gmm_density import GMMDensity
        target_function = GMMDensity
    else:
        raise ValueError(f"Unknown target function name: {target_function_name}")
    target_function_instance = target_function(target_function_config=target_function_config, boundary=boundary,
                                               fixed_target=fixed_target, random_state=random_state)
    return target_function_instance
