from typing import Dict, Any, List, Union, Callable, Optional

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.basedatatypes import BaseTraceType

from modules.swarm_environments.mesh.functions.abstract_density import AbstractDensity
from modules.swarm_environments.mesh.functions.target_function_util import build_gmm
from modules.swarm_environments.util.torch_util import detach
from modules.swarm_environments.util.visualization import plot_2d_covariance


class GMMDensity(AbstractDensity):
    """
    Represents a 2d Gaussian Mixture Model with a given number of components.
    Each time the reset() function is called, a new number of components is chosen. The components are then randomly
     assigned weights, mean and covariance.
    """

    def __init__(self, *, target_function_config: Dict[Union[str, int], Any], boundary: np.array, fixed_target: bool,
                 random_state: np.random.RandomState):
        """

        Args:
            target_function_config: Config containing the covariances of the gaussians that choose the means and
            covariances of the target gaussian
            boundary: A rectangular boundary that the density must adhere to. Used for determining the mean of the
              gaussian used to draw the Gaussian Density
            random_state: Internally used random_state to generate GMMs. Will be used to create GMMs, either
              once at the start if fixed_target, or for every reset() else. 
        """
        super().__init__(target_function_config=target_function_config, boundary=boundary, fixed_target=fixed_target,
                         random_state=random_state)

        self._num_components = target_function_config.get("num_components")
        self._gmm_sample_mode = target_function_config.get("gmm_sample_mode", "random")
        self._gmm_noise_scale = target_function_config.get("gmm_noise_scale", 1.0)
        self._gmm_weight_multiplier = target_function_config.get("gmm_weight_multiplier", 1.0)
        self._fixed_component_weights = target_function_config.get("fixed_component_weights", False)
        # how far to move from the (normalized) middle of the boundary, as a float [0, 0.5)
        mean_position_range = target_function_config.get("mean_position_range", 0.4)
        assert 0 <= mean_position_range <= 0.5, \
            f"mean_position_range must be in [0, 0.5], given '{mean_position_range}'"
        mean_position_range = np.array([0.5 - mean_position_range,
                                        0.5 + mean_position_range])

        self._boundary = boundary

        # bounds of the diagonal covariance values, as list [lower, upper]
        lower_covariance_bound = target_function_config.get("lower_covariance_bound", 0.0001)
        upper_covariance_bound = target_function_config.get("upper_covariance_bound", 0.001)
        assert 1.0e-6 < lower_covariance_bound <= upper_covariance_bound, f"Need positive covariance and a lower " \
                                                                          f"bound smaller than the upper bound, " \
                                                                          f"given '{lower_covariance_bound}' " \
                                                                          f"and '{upper_covariance_bound}'"
        covariance_range = np.array([lower_covariance_bound, upper_covariance_bound])

        if fixed_target:
            if self._fixed_component_weights:
                weights = np.ones(self._num_components) / self._num_components
            else:
                weights = np.arange(self._num_components) + 1
                weights = weights / np.sum(weights)

            mean_x = np.linspace(mean_position_range[0], mean_position_range[1], self._num_components)
            mean_y = np.linspace(mean_position_range[0], mean_position_range[1], self._num_components)

            means = np.vstack((mean_x, mean_y)).T

            means = self._scale_to_boundary(means)

            diagonal_covariances = np.exp(np.linspace(np.log(covariance_range[0]),
                                                      np.log(covariance_range[1]),
                                                      (self._num_components * 2)))
            diagonal_covariances = diagonal_covariances.reshape((self._num_components, 2))
            rotation_angles = np.linspace(0, 1 * np.pi, self._num_components, endpoint=False)
            self._distribution = build_gmm(weights=weights,
                                           means=means,
                                           diagonal_covariances=diagonal_covariances,
                                           rotation_angles=rotation_angles
                                           )
            self._means = means
            self._weights = weights

        else:
            self._random_state = random_state
            self._mean_position_range = mean_position_range
            self._covariance_range = covariance_range
            self._means = None
            self._weights = None

    def reset(self, valid_point_function: Optional[Callable[[np.array], np.array]] = None, *args, **kwargs) -> None:
        """
        Resets the GMM to a new random GMM. The means can be chosen such that they are valid wrt. some constraints or
         domain.
        Args:
            valid_point_function: Function that takes an array of means and returns the valid subset of these means.
            If None, all means are considered valid.

        Returns: None

        """
        if not self.fixed_target:
            # draw n random components with uniformly drawn (valid) mean, softmax weighting and random orientation

            if self._gmm_sample_mode == "random":
                if valid_point_function is not None:
                    found_means = []
                    while len(found_means) < self._num_components:
                        # do rejection sampling on domain until enough means are found
                        candidate_means = self._random_state.uniform(self._mean_position_range[0],
                                                                     self._mean_position_range[1],
                                                                     size=(self._num_components * 10, 2))
                        valid_means = valid_point_function(candidate_means)
                        found_means.extend(valid_means)

                    found_means = np.array(found_means)
                    means = found_means[:self._num_components]
                else:
                    means = self._random_state.uniform(low=self._mean_position_range[0],
                                                       high=self._mean_position_range[1],
                                                       size=(self._num_components, 2))
            elif self._gmm_sample_mode == "stratified":
                assert np.sqrt(self._num_components) % 1 == 0, "Need a square number of components for stratified " \
                                                               "sampling"
                mean_x = np.linspace(self._mean_position_range[0], self._mean_position_range[1],
                                     int(np.sqrt(self._num_components)))
                mean_y = np.linspace(self._mean_position_range[0], self._mean_position_range[1],
                                     int(np.sqrt(self._num_components)))

                mean_x, mean_y = np.meshgrid(mean_x, mean_y)
                mean_x = mean_x.flatten()
                mean_y = mean_y.flatten()
                means = np.vstack((mean_x, mean_y)).T

                # add noise to means
                means += self._random_state.normal(scale=0.1, size=means.shape)

            elif self._gmm_sample_mode in ["normal", "gaussian"]:
                means = self._random_state.normal(loc=0.5, scale=self._gmm_noise_scale, size=(self._num_components, 2))
            else:
                raise ValueError(f"Unknown GMM sampling mode '{self._gmm_sample_mode}'")

            if self._fixed_component_weights:
                weights = np.ones(self._num_components) / self._num_components
            else:
                weights = 1 + np.exp(self._random_state.normal(size=self._num_components))
                weights = weights / np.sum(weights)  # make weights sum to 1

            means = self._scale_to_boundary(means)

            diagonal_covariances = np.exp(self._random_state.uniform(low=np.log(self._covariance_range[0]),
                                                                     high=np.log(self._covariance_range[1]),
                                                                     size=(self._num_components, 2)))

            rotation_angles = self._random_state.random(self._num_components) * 2 * np.pi

            self._distribution = build_gmm(weights=weights,
                                           means=means,
                                           diagonal_covariances=diagonal_covariances,
                                           rotation_angles=rotation_angles,
                                           )

            self._means = means
            self._weights = weights

    def _scale_to_boundary(self, means):
        # normalize means wrt. boundary. The boundary is given as (lower_x, lower_y, upper_x, upper_y)
        means = means * (self._boundary[2:] - self._boundary[:2]) + self._boundary[:2]
        return means

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
        log_probability = log_probability + np.log(self._gmm_weight_multiplier)
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
    def weights(self) -> np.array:
        return self._weights

    @property
    def means(self) -> np.array:
        return self._means

    def plot(self) -> List[BaseTraceType]:
        traces = []

        full_distribution = self._distribution.component_distribution.base_dist
        from plotly.colors import sequential
        color_scale = sequential.Greys
        weights = detach(self._distribution.mixture_distribution.probs)

        if np.min(weights) == np.max(weights):  # all weights equal, can't normalize
            scaled_weights = np.ones(len(weights))
        else:  # different weights, can normalize
            scaled_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        for scaled_weight, mean, covariance_matrix in zip(scaled_weights,
                                                          full_distribution.mean,
                                                          full_distribution.covariance_matrix):
            color = 255 - int(scaled_weight * 255)
            line_color = f'rgb({color},{color},{color})'  # greyscale color depending on weight

            traces.extend(plot_2d_covariance(mean=detach(mean),
                                             covariance_matrix=detach(covariance_matrix),
                                             line_color=line_color))

        if len(weights) > 1:  # has multiple components
            colorbar_trace = go.Scatter(x=[None],
                                        y=[None],
                                        mode='markers',
                                        marker=dict(
                                            colorscale=color_scale,
                                            showscale=True,
                                            cmin=0,
                                            cmax=1,
                                            colorbar=dict(orientation="h", thickness=10, title='Component Weights',
                                                          tickvals=scaled_weights,
                                                          ticktext=[f"{weight:.3f}" for weight in weights],
                                                          outlinewidth=0, y=-0.02, yanchor="top")
                                        ),
                                        hoverinfo='none',
                                        showlegend=False
                                        )
            traces.append(colorbar_trace)
        return traces
