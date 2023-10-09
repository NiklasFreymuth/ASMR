import abc

import numpy as np
import torch

from modules.swarm_environments.mesh.functions.abstract_target_function import AbstractTargetFunction
from modules.swarm_environments.util.torch_util import detach
from typing import Dict, Any, Union


class AbstractDensity(AbstractTargetFunction, abc.ABC):
    """
    Represents a 2d Gaussian Mixture Model with a given number of components.
    Each time the reset() function is called, a new number of components is chosen. The components are then randomly
     assigned weights, mean and covariance.
    """

    def __init__(self, *, target_function_config: Dict[Union[str, int], Any], boundary: np.array, fixed_target: bool,
                 random_state: np.random.RandomState):
        """

        Args:
            target_function_config: Config containing details about the distribution/density function. Must contain a
                "density_mode" parameter
            boundary: A rectangular boundary that the density must adhere to. Used for determining the mean of the
              gaussian used to draw the Gaussian Density
            random_state: Internally used random_state. Will be used to create densities, either
              once at the start if fixed_target, or for every reset() else.
        """
        super().__init__(target_function_config=target_function_config, boundary=boundary,
                         fixed_target=fixed_target, random_state=random_state)
        self._density_mode = target_function_config.get("density_mode", "density")
        self._boundary_center = np.array((np.mean(boundary[0::2]), np.mean(boundary[1::2])))
        self._distribution = None

    def evaluate(self, samples: np.array, include_gradient: bool = False) -> np.array:
        """

        Args:
            samples: Array of shape (#samples, 2)
            include_gradient: Whether to include a gradient in the returns or not. If True, the output is an array
              of shape (#samples, 3), where the last dimension is for the function evaluation, and the grdient wrt.
              x and y. If False, the output is one evaluation per sample, i.e., of shape (#samples, )

        Returns:

        """
        assert self._distribution is not None, "Need to specify a distribution before evaluating. " \
                                               "Try calling reset() first."
        input_samples = torch.tensor(samples)
        if include_gradient:
            input_samples.requires_grad = True
        log_probability = self._distribution.log_prob(input_samples)
        if self._density_mode == "log_density":
            value = log_probability
        elif self._density_mode == "density":
            value = torch.exp(log_probability)
        else:
            raise ValueError(f"Unknown density_mode '{self._density_mode}'")

        if include_gradient:
            value.backward(torch.ones(len(input_samples)))
            gradients = detach(input_samples.grad)
            values = detach(value)
            return np.concatenate((values[:, None], gradients), axis=-1)
        else:
            return detach(value)

    @property
    def distribution(self) -> np.array:
        return self._distribution
