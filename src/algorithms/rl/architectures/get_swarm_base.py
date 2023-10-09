from src.algorithms.baselines.architectures.sweep.get_sweep_base import get_sweep_base
from modules.swarm_environments.abstract_swarm_environment import AbstractSwarmEnvironment
from modules.hmpn import AbstractMessagePassingBase, get_hmpn
from util.types import *


def get_swarm_base(graph_env: AbstractSwarmEnvironment, network_config: ConfigDict,
                   device) -> AbstractMessagePassingBase:
    """

    Args:
        graph_env:
        network_config:
        device:

    Returns:

    """
    latent_dimension = network_config.get("latent_dimension")
    base_config = network_config.get("base")
    type_of_base = network_config.get("type_of_base")

    params = dict(in_node_features=graph_env.num_node_features,
                  in_edge_features=graph_env.num_edge_features,
                  latent_dimension=latent_dimension,
                  base_config=base_config,
                  node_name=graph_env.agent_node_type,
                  device=device)

    # check for different baselines
    if type_of_base == "vdgn_gat":
        from src.algorithms.baselines.architectures.vdgn_gat import VDGNGAT
        return VDGNGAT(**params)
    elif type_of_base == "sweep":
        del params["in_edge_features"]
        return get_sweep_base(**params)
    else:  # get a full message passing network
        return get_hmpn(**params)
