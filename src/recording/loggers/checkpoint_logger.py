import os

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.abstract_logger import AbstractLogger
from util.function import get_from_nested_dict
from util.types import *


class CheckpointLogger(AbstractLogger):
    """
    Creates checkpoints of the algorithm at a given frequency (in iterations)
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        self._is_initial_save = True
        self.checkpoint_directory = os.path.join(self._recording_directory, "checkpoints")
        self._checkpoint_frequency: int = get_from_nested_dict(dictionary=config,
                                                               list_of_keys=["recording", "checkpoint_frequency"],
                                                               raise_error=True)

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        """
        Calls the internal save_checkpoint() method of the algorithm with the current iteration
        Args:
            recorded_values: A dictionary of previously recorded values
            iteration: The current iteration of the algorithm
        Returns:

        """
        if self._checkpoint_frequency > 0 and iteration % self._checkpoint_frequency == 0:
            self._writer.info(msg="Checkpointing algorithm")
            self._algorithm.save_checkpoint(directory=self.checkpoint_directory,
                                            iteration=iteration,
                                            is_initial_save=self._is_initial_save)
            self._is_initial_save = False

    def finalize(self) -> None:
        """
        Makes a final checkpoint of the algorithm
        Returns:

        """
        self._algorithm.save_checkpoint(directory=self.checkpoint_directory, is_final_save=True)
