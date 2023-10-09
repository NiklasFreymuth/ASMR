import numpy as np


def log_normalize(samples: np.array, axis: int = 0) -> np.array:
    """
    Normalizes the given samples in logspace, i.e. returns an offset of the samples such that
    sum(exp(samples))==1
    Args:
        samples: The samples to log_normalize
        axis: The axis along which to normalize. Defaults to 0, i.e., the first dimension

    Returns: Axis-wise Log-normalized samples

    """
    if not axis == 0:
        axes_enumeration = list(range(samples.ndim))
        axes_enumeration[0] = axis
        axes_enumeration[axis] = 0
        samples = samples.transpose(axes_enumeration)  # tranpose relevant axis to the front
        samples = samples - logsumexp(samples)
        samples = samples.transpose(axes_enumeration)  # undo transpose after normalization
    else:
        samples = samples - logsumexp(samples)
    return samples


def logsumexp(samples: np.array) -> np.array:
    """
    Uses the identity
    np.log(np.sum_i(np.exp(sample_i))) = np.log(np.sum_i(np.exp(sample_i-maximum)))+maximum
    to calculate a sum of e.g. functions in a numerically stable way
    Args:
        samples: Samples to perform logsumexp on

    Returns:
    """
    assert len(samples) > 0, "Must have at least 1 sample to logsumexp it. Given {}".format(samples)
    maximum_log_density = np.max(samples, axis=0)
    log_term = np.log(np.sum(np.exp(samples - maximum_log_density), axis=0))
    results = log_term + maximum_log_density
    return results
