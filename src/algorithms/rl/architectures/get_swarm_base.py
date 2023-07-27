from src.algorithms.baselines.architectures.sweep.get_sweep_base import get_sweep_base
from src.environments.abstract_swarm_environment import AbstractSwarmEnvironment
from src.modules.mpn.message_passing_base import MessagePassingBase
from util.types import *


def get_swarm_base(graph_env: AbstractSwarmEnvironment, network_config: ConfigDict,
                   device) -> MessagePassingBase:
    """

    """
    from src.modules.mpn.get_message_passing_base import get_message_passing_base
    latent_dimension = network_config.get("latent_dimension")
    base_config = network_config.get("base")
    type_of_base = network_config.get("type_of_base")

    params = dict(in_node_features=graph_env.num_node_features,
                  in_edge_features=graph_env.num_edge_features,
                  in_global_features=graph_env.num_global_features,
                  latent_dimension=latent_dimension,
                  base_config=base_config,
                  agent_node_type=graph_env.agent_node_type,
                  device=device)

    # check for different baselines
    if type_of_base == "sweep":
        del params['in_global_features']  # local base uses no global or edge features
        del params['in_edge_features']
        del params['agent_node_type']
        return get_sweep_base(**params)
    else:  # get a full message passing network
        return get_message_passing_base(**params)
