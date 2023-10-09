import abc
from typing import Dict, Optional, Any

import torch.nn as nn
from torch_geometric.data.batch import Batch


class AbstractStack(nn.Module, abc.ABC):
    """
    Message Passing module that acts on both node and edge features used for observation graphs.
    Internally stacks multiple instances of MessagePassingStep.
    """

    def __init__(self,
                 latent_dimension: int,
                 stack_config: Dict[str, Any]):
        """
        Args:
            latent_dimension: Dimensionality of the latent space
            stack_config: Dictionary specifying the way that the gnn base should look like.
                num_steps: how many steps this stack should have
                residual_connections: Which kind of residual connections to use
        """
        super().__init__()
        self._num_steps: int = stack_config.get("num_steps")
        self._num_step_repeats: int = stack_config.get("num_step_repeats", 1)
        self._residual_connections: Optional[str] = stack_config.get("residual_connections")
        self._latent_dimension: int = latent_dimension
        self._message_passing_steps: nn.ModuleList = None

    @property
    def num_steps(self) -> int:
        """
        How many steps this stack is composed of.
        """
        return self._num_steps

    @property
    def latent_dimension(self) -> int:
        """
        Dimensionality of the features that are handled in this stack
        Returns:

        """
        return self._latent_dimension

    def forward(self, graph: Batch) -> None:
        """
        Computes the forward pass for this message passing stack.
        Updates node and edge features (new_node_features, new_edge_features)
        for each type as a tuple

        Args: graph of type torch_geometric.data.Batch containing homogeneous graphs

        Returns: None, in-place operation
        """
        for message_passing_step in self._message_passing_steps:
            for repeat in range(self._num_step_repeats):
                message_passing_step(graph=graph)

    def __repr__(self):
        if self._message_passing_steps:
            return f"{self.__class__.__name__}(\n" \
                   f" num_message_passing_steps={self.num_steps},\n" \
                   f" first_step={self._message_passing_steps[0]}\n"
        else:
            return f"{self.__class__.__name__}(\n" \
                   f" num_message_passing_steps={self.num_steps}\n"
