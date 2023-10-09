import numpy as np
from plotly.basedatatypes import BaseTraceType

from modules.swarm_environments.mesh.functions.abstract_density import AbstractDensity
from modules.swarm_environments.mesh.functions.target_function_util import build_gaussian
from modules.swarm_environments.util.torch_util import detach
from typing import Dict, Any, List, Union, Callable, Optional
from modules.swarm_environments.util.visualization import plot_2d_covariance


class GaussianDensity(AbstractDensity):
    """
    Represents a simple 2d Gaussian. Each time the reset() function is called, a new mean and covariance for the
    Gaussian are chosen according to another Gaussian.
    """

    def __init__(self, *, target_function_config: Dict[Union[str, int], Any], boundary: np.array, fixed_target: bool,
                 random_state: np.random.RandomState
                 ):
        """

        Args:
            target_function_config: Config containing the covariances of the gaussians that choose the means and
            covariances of the target gaussian
            boundary: A rectangular boundary that the density must adhere to. Used for determining the mean of the
              gaussian used to draw the Gaussian Density
            random_state: Internally used random_state to generate Gaussians. Will be used to create GMMs, either
              once at the start if fixed_target, or for every reset() else.
        """
        super().__init__(target_function_config=target_function_config, boundary=boundary,
                         fixed_target=fixed_target, random_state=random_state)
        self._initial_covariance = np.matrix([[0.02, 0], [0.0, 0.01]])

        self._gaussian = None
        if self.fixed_target:
            self._distribution = build_gaussian(mean=self._boundary_center,
                                                initial_covariance=self._initial_covariance,
                                                rotation_angle=0)
        else:
            self._mean_covariance = target_function_config.get("mean_covariance")

    def reset(self, valid_point_function: Optional[Callable[[np.array], np.array]] = None):
        if not self.fixed_target:
            rotation_angle = self.random_state.random() * 2 * np.pi
            if self._mean_covariance > 0:
                if valid_point_function is None:
                    mean = self.random_state.multivariate_normal(mean=np.array((np.mean(self.boundary[0::2]),
                                                                                np.mean(self.boundary[1::2]))),
                                                                 cov=np.eye(2) * self._mean_covariance,
                                                                 size=1).squeeze()
                else:
                    mean = None
                    while mean is None:
                        candidate_mean = self.random_state.multivariate_normal(
                            mean=np.array((np.mean(self.boundary[0::2]),
                                           np.mean(self.boundary[1::2]))),
                            cov=np.eye(2) * self._mean_covariance,
                            size=1).squeeze()
                        valid_mean = valid_point_function(np.array([candidate_mean]))
                        if len(valid_mean) > 0:
                            mean = candidate_mean
            else:
                mean = self._boundary_center
            self._distribution = build_gaussian(mean=mean,
                                                initial_covariance=self._initial_covariance,
                                                rotation_angle=rotation_angle)

    def plot(self) -> List[BaseTraceType]:
        return plot_2d_covariance(mean=detach(self._distribution.mean),
                                  covariance_matrix=detach(self._distribution.covariance_matrix),
                                  line_color="grey")
