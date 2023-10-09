"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from util.function import get_from_nested_dict
from util.types import *


def get_algorithm(config: ConfigDict, seed: Optional[int] = None) -> AbstractIterativeAlgorithm:
    algorithm_name = get_from_nested_dict(config, list_of_keys=["algorithm", "name"], raise_error=True).lower()
    if algorithm_name == "ppo":
        from src.algorithms.rl.on_policy.swarm_ppo import SwarmPPO
        return SwarmPPO(config=config, seed=seed)
    elif algorithm_name == "dqn":
        from src.algorithms.rl.off_policy.swarm_dqn import SwarmDQN
        return SwarmDQN(config=config, seed=seed)

    # baseline algorithms
    elif algorithm_name == "sweep_ppo":
        from src.algorithms.baselines.sweep_ppo import SweepPPO
        return SweepPPO(config=config, seed=seed)
    elif algorithm_name == "sweep_dqn":
        from src.algorithms.baselines.sweep_dqn import SweepDQN
        return SweepDQN(config=config, seed=seed)
    elif algorithm_name == "vdgn_ppo":
        # vdgn ppo can be derived from the main ppo class.
        from src.algorithms.rl.on_policy.swarm_ppo import SwarmPPO
        return SwarmPPO(config=config, seed=seed)
    elif algorithm_name in ["vdgn", "vdgn_dqn"]:  # uses DQN as underlying RL algorithm
        from src.algorithms.baselines.vdgn_dqn import VDGN
        return VDGN(config=config, seed=seed)
    elif algorithm_name == "single_agent_ppo":
        from src.algorithms.baselines.single_agent_ppo import SingleAgentPPO
        return SingleAgentPPO(config=config, seed=seed)
    elif algorithm_name == "single_agent_dqn":
        from src.algorithms.baselines.single_agent_dqn import SingleAgentDQN
        return SingleAgentDQN(config=config, seed=seed)
    else:
        raise NotImplementedError("Implement your algorithms here!")
