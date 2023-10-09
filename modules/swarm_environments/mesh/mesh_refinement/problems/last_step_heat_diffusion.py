r"""Heat Diffusion task with an outer boundary with boundary condition Omega_0=0,
and some heat source
"""

import os

import numpy as np
from skfem import Basis

from modules.swarm_environments.mesh.mesh_refinement.problems.heat_diffusion import HeatDiffusion
from typing import List

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LastStepHeatDiffusion(HeatDiffusion):
    def _calculate_solution(self, basis: Basis, cache: bool) -> np.array:
        # only return the solution of the last step
        return super()._calculate_solution(basis=basis, cache=cache)[:, -1]

    @staticmethod
    def solution_dimension_names() -> List[str]:
        return ["final_step"]

    @property
    def solution_dimension_weights(self) -> np.array:
        return np.array([1])
