r"""
Buffer of finite element problems. This class stores several finite element problems and allows to sample from them.
In particular, each FEM Problem consists of an original coarse mesh and basis, and a fine-grained mesh, basis,
and solution.
The meshes are skfem Mesh objects, each basis is a skfem Basis object, and the solution is a PDESolution object.
This class further gives interfaces to interact with the finite element problems, e.g., to calculate a reward or to plot
them.
"""
import os
from typing import Dict, Any, List, Union, Optional, Type

import numpy as np

from modules.swarm_environments.mesh.mesh_refinement.fem_problem_wrapper import FEMProblemWrapper
from modules.swarm_environments.mesh.mesh_refinement.problems.abstract_finite_element_problem import \
    AbstractFiniteElementProblem
from modules.swarm_environments.mesh.mesh_refinement.problems.get_finite_element_problem import get_finite_element_problem_class
from modules.swarm_environments.util.function import filter_included_fields
from modules.swarm_environments.util.index_sampler import IndexSampler

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FEMProblemCircularQueue:
    def __init__(self, *,
                 fem_config: Dict[Union[str, int], Any],
                 random_state: np.random.RandomState = np.random.RandomState()):
        """
        Initializes the AbstractFiniteElementProblem.

        Args:
            fem_config: Config containing additional details about the finite element method. Contains
                domain_config: Configuration describing the family of geometries for this problem
                plot_resolution: The resolution of the plots in x and y direction
                error_metric:
                 The metric to use for the error estimation. Can be either
                    "squared": the squared error is used
                    "mean": the mean error is used
            random_state: A random state to use for drawing random numbers
        """
        self._fem_config = fem_config
        self._random_state = random_state

        #################
        # problem state #
        #################

        # parameters for a buffer of pdes to avoid calculating the same pde multiple times
        num_pdes = fem_config.get("num_pdes")  # number of pdes to store. None, 0 or -1 means infinite
        self._use_buffer = num_pdes is not None and num_pdes > 0
        num_pdes = num_pdes if self._use_buffer else 1

        self._index_sampler = IndexSampler(num_pdes, random_state=self._random_state)
        self._fem_problems: List[Optional[FEMProblemWrapper]] = [None for _ in range(num_pdes)]

        # parameters for the partial differential equation
        self._fem_problem_class: Type[AbstractFiniteElementProblem] \
            = get_finite_element_problem_class(fem_config=fem_config)

        pde_config = fem_config.get(fem_config.get("pde_type"))
        self._pde_features = {
            "element_features": filter_included_fields(pde_config.get("element_features", {})),
        }

    def next(self) -> FEMProblemWrapper:
        """
        Draws the next finite element problem. This method is called at the beginning of each episode and draws a
        (potentially new) finite element problem from the buffer.
        Returns:

        """
        pde_idx = self._index_sampler.next()
        return self._next_from_idx(pde_idx=pde_idx)

    def _next_from_idx(self, pde_idx: int):
        if (not self._use_buffer) or self._fem_problems[pde_idx] is None:
            # draw a new fem_problem from the given distribution if we are not using a buffer or if the buffer entry
            # is empty
            new_seed = self._random_state.randint(0, 2 ** 31)
            new_problem = self._fem_problem_class(fem_config=self._fem_config,
                                                  random_state=np.random.RandomState(seed=new_seed))
            new_problem = FEMProblemWrapper(fem_config=self._fem_config,
                                            fem_problem=new_problem,
                                            pde_features=self._pde_features,
                                            )
            self._fem_problems[pde_idx] = new_problem
        self._fem_problems[pde_idx].reset()
        return self._fem_problems[pde_idx]

    ##############################
    #    Static Observations     #
    ##############################

    @property
    def fem_config(self):
        return self._fem_config

    @property
    def num_pde_element_features(self) -> int:
        return len(self._pde_features["element_features"])


    @property
    def solution_dimension_names(self) -> List[str]:
        """
        Returns a list of names of the solution dimensions. This is used to name the columns of the solution
        matrix.
        This also determines the solution dimension
        Returns: A list of names of the solution dimensions

        """
        return self._fem_problem_class.solution_dimension_names()

    @property
    def current_size(self):
        return sum([1 for _ in self._fem_problems if _ is not None])
