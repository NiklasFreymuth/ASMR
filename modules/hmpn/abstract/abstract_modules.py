import abc
from typing import Callable, List

import torch


class AbstractMetaModule(torch.nn.Module, abc.ABC):
    """
    An abstract class for the modules used in the GNN. They are used for updating node-, and edge features.
    """

    def __init__(self, scatter_reducers: List[Callable]):
        """
        Args:
            scatter_reducers: How to aggregate over the nodes/edges. Can for example be torch.scatter_mean()
        """
        super().__init__()

        self._scatter_reducers = scatter_reducers
        self._n_scatter_reducers = len(self._scatter_reducers)

        if self._n_scatter_reducers == 1:
            # saving on operations if only one reducer is used.
            self.multiscatter = self._no_multiscatter
        else:
            self.multiscatter = self._multiscatter

    def _no_multiscatter(self, features: torch.Tensor, indices: torch.Tensor, dim: int, dim_size: int):
        return self._scatter_reducers[0](features, indices, dim=dim, dim_size=dim_size)

    def _multiscatter(self, features: torch.Tensor, indices: torch.Tensor, dim: int, dim_size: int):
        """
        This function is used to aggregate over the nodes/edges/globals. Uses an internal list of scatter recude
        operations such as min/mean/sum/max and outputs a concatenation of the reduced results
        Args:
            features:
            indices:
            dim:
            dim_size:

        Returns:

        """
        # only invoked if more than one reducer is used.
        latent_dimension = features.shape[1]
        reduced = torch.zeros((dim_size, latent_dimension * self._n_scatter_reducers), device=features.device)
        for position, reducer in enumerate(self._scatter_reducers):
            reduced[:, features.shape[1] * position:features.shape[1] * (position + 1)] = \
                reducer(features, indices, dim=dim, dim_size=dim_size)

        # we have [[1,1.1,1.2,1.3,1.4],[2,2.1,2.2,2.3,2.4]]
        # and [[1,1.5,1.6,1.7,1.8],[2,2.5,2.6,2.7,2.8]]
        # and we want [[1,1.1,1.2,1.3,1.4,1,1.5,1.6,1.7,1.8],[2,2.1,2.2,2.3,2.4,2,2.5,2.6,2.7,2.8]]
        # this seems to be the correct thing to do:
        return reduced

    def extra_repr(self) -> str:
        """
        To be printed when using print(self) on either this class or some module implementing it
        Returns: A string containing information about the parameters of this module
        """
        return f"scatter reduce: {self._scatter_reducers}"
