import numpy as np

from modules.swarm_environments.mesh.mesh_refinement.fem_problem_circular_queue import FEMProblemCircularQueue
from typing import Dict, Any, Union


class EvaluationFEMProblemCircularQueue(FEMProblemCircularQueue):
    def __init__(self, *,
                 fem_config: Dict[Union[str, int], Any],
                 random_state: np.random.RandomState = np.random.RandomState()):
        super().__init__(fem_config=fem_config, random_state=random_state)
        self._index_sampler = None
        num_pdes = fem_config.get("num_pdes")  # number of pdes to store. None, 0 or -1 means infinite
        self._current_index = 0
        self._max_index = num_pdes

    def next(self) -> None:
        """
        Draws the next finite element problem. This method is called at the beginning of each episode and draws a
        (potentially new) finite element problem from the buffer.
        Returns:

        """
        pde_idx = self._current_index % self._max_index
        self._current_index += 1

        return self._next_from_idx(pde_idx=pde_idx)

    @property
    def num_pdes(self):
        return self._max_index
