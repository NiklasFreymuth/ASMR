from typing import Union

import numpy as np
from skfem import adaptive_theta


class RemeshingHeuristic:
    def __init__(self, theta: Union[float, int], area_scaling: bool = False):
        """

        Args:
            theta: If the error is within theta*max_error, the element is not refined.
                If int: The number of elements to refine.
                If float: The fraction of elements to refine.
            area_scaling: If true, the error is scaled by the area of the element.
        """
        self._theta = theta
        self._area_scaling = area_scaling

    def __call__(self, error_per_element: np.ndarray, element_areas: np.ndarray) -> np.ndarray:
        """

        Args:
            error_per_element: error per face. The error is the integrated
            L1 norm of the difference between the solution and the
            reference solution.

        Returns: actions to take. 1 means refine, -1 means do nothing

        """
        if self._area_scaling:
            error_per_element = error_per_element / element_areas
        if isinstance(self._theta, float):
            assert 0 <= self._theta <= 1
            elements_to_refine = adaptive_theta(error_per_element, theta=self._theta)
            actions = np.zeros_like(error_per_element)
            actions[elements_to_refine] = 1
        elif isinstance(self._theta, int):
            error_per_element[element_areas < 1.0e-6] = 0
            elements_to_refine = np.argsort(error_per_element)[::-1][:self._theta]
            actions = np.zeros_like(error_per_element)
            actions[elements_to_refine] = 1
        else:
            raise ValueError(f"Theta must be float or int, but is {type(self._theta)}")
        return actions

    def get_actions(self, error_per_element: np.ndarray, element_areas) -> np.ndarray:
        return self.__call__(error_per_element=error_per_element, element_areas=element_areas)
