import numpy as np


def log_lr(ax, num_agents, error_metrics):
    """
    Method to plot the log-log quadratic regression for the pareto plot.
    Args:
        ax: Matplotlib figure axis
        num_agents: 1d numpy array containing the x-values of the pareto plot
        error_metrics: 1d numpy array containing the y-values of the pareto plot

    Returns:

    """
    # sort
    agent_idx = np.argsort(num_agents)
    num_agents = num_agents[agent_idx]
    error_metrics = error_metrics[agent_idx]

    # log values
    log_num_agents = np.log(num_agents)
    log_error_metrics = np.log(error_metrics)

    # log-log-quadratic regression
    model = np.poly1d(np.polyfit(log_num_agents, log_error_metrics, 2))

    # plot points
    ax.scatter(num_agents, error_metrics)
    # plot line
    ax.plot(num_agents, np.exp(model(log_num_agents)))

