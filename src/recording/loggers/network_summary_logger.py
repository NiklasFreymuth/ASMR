from util.types import *
from src.recording.loggers.abstract_logger import AbstractLogger
from src.algorithms.rl.abstract_rl_algorithm import AbstractRLAlgorithm


class NetworkSummaryLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        if iteration == 0:
            assert isinstance(self._algorithm, AbstractRLAlgorithm), f"Task must have a neural network, " \
                                                                     f"provided {type(self._algorithm)} " \
                                                                     f"instead"
            self._writer.info(self._algorithm.policy)
            total_network_parameters = sum(p.numel() for p in self._algorithm.policy.parameters())
            trainable_parameters = sum(p.numel() for p in self._algorithm.policy.parameters() if p.requires_grad)
            self._writer.info(f"Total parameters: {total_network_parameters}")
            self._writer.info(f"Trainable parameters: {trainable_parameters}")

    def finalize(self) -> None:
        pass
