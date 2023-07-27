import numpy as np
import torch
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical, Independent


def build_gmm(weights: np.array, means: np.array,
              diagonal_covariances: np.array, rotation_angles: np.array) -> MixtureSameFamily:
    """
    Builds a 2d Gaussian Mixture Model from input weights, means, covariance diagonals and a rotation of the covariance
    Args:
        weights:
        means:
        diagonal_covariances: Diagonal/uncorrelated covariance matrices.
         Has shape (num_components, 2) and will be broadcast to an array of matrices
        rotation_angles: Angle to rotate the covariance matrix by. A rotation of 2pi results in the original matrix

    Returns:

    """
    diagonal_covariances = np.eye(2)[None, ...] * diagonal_covariances[:, None, :]  # broadcast to matrix
    # shape (num_components, 2, 2)

    rotation_matrices = np.array([[np.cos(rotation_angles), -np.sin(rotation_angles)],
                                  [np.sin(rotation_angles), np.cos(rotation_angles)]])
    # shape (2, 2, num_components)

    rotated_covariances = np.einsum("jki, ikl, mli -> ijm", rotation_matrices,
                                    diagonal_covariances, rotation_matrices)
    # generalization/vectorization of "ij, jk, lk -> il", i.e.,
    # rotated_covariance = rotation_matrix @ diagonal_covariance @ rotation_matrix.T

    weights = torch.tensor(weights)
    means = torch.tensor(means)
    rotated_covariances = torch.tensor(rotated_covariances)

    mix = Categorical(weights, validate_args=False)
    comp = Independent(base_distribution=MultivariateNormal(loc=means,
                                                            covariance_matrix=rotated_covariances),
                       reinterpreted_batch_ndims=0, validate_args=False)
    gmm = MixtureSameFamily(mix, comp, validate_args=False)
    return gmm
