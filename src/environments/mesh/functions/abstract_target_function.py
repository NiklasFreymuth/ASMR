import numpy as np
from plotly.basedatatypes import BaseTraceType

from util.types import *


class AbstractTargetFunction:
    """
    Represents an abstract 2d bounded family of target functions to train on.
    These can be seen as target functions that the agents in the MeshRefinement task should approximate.
    """

    def __init__(self, *, target_function_config: ConfigDict, boundary: np.array, fixed_target: bool,
                 random_state: np.random.RandomState):
        """

        Args:
            target_function_config: Config containing information on the (family) of functions to build.
            boundary: A rectangular boundary that the density must adhere to. Used for determining the mean of the
              gaussian used to draw the Gaussian Density
            fixed_target: Whether to represent a fixed_target function, or create/draw a new function with every reset
            random_state: Internally used random_state. Will be used to sample functions from pre-defined families,
                either once at the start if fixed_target, or for every reset() else.
        """
        self.target_function_config = target_function_config
        self.boundary = boundary
        self.fixed_target = fixed_target
        self.random_state = random_state

    def reset(self, valid_point_function: Optional[Callable[[np.array], np.array]] = None):
        """
        Reset the target function. For non-deterministic target functions (fixed_target=False), this means that
        a new function can be drawn out of the family of functions.
        Args:
            valid_point_function: A function that takes an array of points as input and outputs the subset of points
            that is valid wrt. some constraint. If given, the new function will respect these constraints

        Returns:

        """
        raise NotImplementedError("AbstractTargetFunction does not implement 'reset()'")

    def __call__(self, samples: np.array, include_gradient: bool = False) -> np.array:
        """
        Wrapper for evaluate()
        """
        return self.evaluate(samples=samples, include_gradient=include_gradient)

    def evaluate(self, samples: np.array, include_gradient: bool = False) -> np.array:
        """
        Evaluate the target function on the given samples
        Args:
            samples: Array of shape (..., 2) of 2d points to evaluate the function at
            include_gradient: Whether to also include gradients wrt. x and y for each output

        Returns: A scalar evaluation per point as an array of shape (...,) if no gradients should be included.
        An array of shape (..., 3) where the first entry is the value, and the second and third are the input gradients
        wrt. x and y otherwise

        """
        raise NotImplementedError("AbstractTargetFunction does not implement 'evaluate()'")

    def plot(self) -> List[BaseTraceType]:
        raise NotImplementedError("AbstractTargetFunction does not implement 'plot()'")
