from typing import Tuple

import torch
from torch import Tensor


class TorchRunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), dtype: torch.dtype = torch.float64):
        """
        Creates a running mean and std object

        Args:
            epsilon: small value to avoid division by zero
            shape: shape of the data
            dtype: data type of the data
        """
        self.mean = torch.zeros(*shape, dtype=dtype)
        self.var = torch.ones(*shape, dtype=dtype)
        self.count = epsilon

    def update(self, arr: Tensor) -> None:
        """
        Updates the running mean and std with a new batch of data

        Args:
            arr: new batch of data as a tensor
        """
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0)
        batch_var = torch.nan_to_num(batch_var, nan=0.0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: Tensor, batch_var: Tensor, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

