from typing import Callable, Dict, List

import pandas as pd
from modules.swarm_environments import MeshRefinement
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment

from util.types import ConfigDict


class RLAlgorithmEvaluator:
    def __init__(self, config: ConfigDict, policy_step_function: Callable,
                 environments: List[AbstractSwarmEnvironment]):
        self.policy_step_function = policy_step_function
        self.environments = environments
        self.config = config

    def __call__(self) -> Dict[str, pd.DataFrame]:
        return self.get_tables()

    def get_tables(self) -> Dict[str, pd.DataFrame]:
        if isinstance(self.environments[0], MeshRefinement):
            # do mesh refinement evaluations
            from modules.swarm_environments.mesh.mesh_refinement.evaluation.evaluate_mesh_refinement import \
                evaluate_mesh_refinement

            final_evaluation_dict = {}
            for idx, environment in enumerate(self.environments):
                num_pdes = 100
                final_evaluation_dict[f"final_{idx}"] = evaluate_mesh_refinement(
                    policy_step_function=self.policy_step_function,
                    environment=environment,
                    num_pdes=num_pdes)
            return final_evaluation_dict
        else:
            return {}
