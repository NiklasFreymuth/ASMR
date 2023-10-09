from typing import Dict, Any, List, Callable

from torch import nn

from modules.hmpn.abstract.abstract_stack import AbstractStack
from modules.hmpn.homogeneous.homogeneous_step import HomogeneousStep


class HomogeneousStack(AbstractStack):
    """
    Message Passing module that acts on both node and edge features.
    Internally stacks multiple instances of MessagePassingSteps.
    This implementation is used for homogeneous observation graphs.
    """

    def __init__(self,
                 stack_config: Dict[str, Any],
                 latent_dimension: int,
                 scatter_reducers: List[Callable],
                 flip_edges_for_nodes: bool = False):
        """
        Args:
            stack_config: Dictionary specifying the way that the message passing network base should look like.
                num_steps: how many steps this stack should have
                residual_connections: which kind of residual connections to use. null/None for no connections,
                "outer" for residuals around each full message passing step, "inner" for residuals after each message
            latent_dimension: the latent dimension of all vectors used in this stack
            scatter_reducers: functions of torch_scatter: min,max,mean,std,etc, as a list of functions
        """
        super().__init__(latent_dimension=latent_dimension, stack_config=stack_config)
        self._message_passing_steps = nn.ModuleList([HomogeneousStep(stack_config=stack_config,
                                                                     latent_dimension=latent_dimension,
                                                                     scatter_reducers=scatter_reducers,
                                                                     flip_edges_for_nodes=flip_edges_for_nodes)
                                                     for _ in range(self._num_steps)])
