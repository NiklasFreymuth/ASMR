"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from util.function import get_from_nested_dict
from util.types import *


def get_algorithm(config: ConfigDict,
                  environment: Optional[AbstractSwarmEnvironment] = None,
                  evaluation_environments: Optional[List[AbstractSwarmEnvironment]] = None,
                  seed: Optional[int] = None) -> AbstractIterativeAlgorithm:
    algorithm_name = get_from_nested_dict(config, list_of_keys=["algorithm", "name"], raise_error=True).lower()
    if algorithm_name == "ppo":
        # also includes the vdgn-ppo and argmax-ppo variants, as these are derived from the main SwarmPPO class
        from src.algorithms.rl.on_policy.swarm_ppo import SwarmPPO
        algorithm_class = SwarmPPO
    elif algorithm_name == "dqn":
        from src.algorithms.rl.off_policy.swarm_dqn import SwarmDQN
        algorithm_class = SwarmDQN

    # baseline algorithms
    elif algorithm_name == "sweep_ppo":
        from src.algorithms.baselines.sweep_ppo import SweepPPO
        algorithm_class = SweepPPO
    elif algorithm_name == "sweep_dqn":
        from src.algorithms.baselines.sweep_dqn import SweepDQN
        algorithm_class = SweepDQN
    elif algorithm_name == "vdgn":  # "dqn-vdgn"
        from src.algorithms.baselines.vdgn_dqn import VDGN
        algorithm_class = VDGN
    elif algorithm_name == "argmax_dqn":
        from src.algorithms.baselines.argmax_dqn import ArgmaxDQN
        algorithm_class = ArgmaxDQN
    else:
        # argmax ppo and vdgn ppo are derived from the main SwarmPPO class
        raise NotImplementedError("Implement your algorithms here!")

    return algorithm_class(config=config,
                           environment=environment,
                           evaluation_environments=evaluation_environments,
                           seed=seed)
