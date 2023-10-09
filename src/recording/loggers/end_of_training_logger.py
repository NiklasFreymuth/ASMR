import os

import numpy as np

from src.recording.loggers.abstract_logger import AbstractLogger
from util.types import *


class EndOfTrainingLogger(AbstractLogger):
    """
    Records the logging at the end of training. This can e.g., be the evaluation of the algorithm
    on an expensive test set
    """

    def log_iteration(self, recorded_values: ValueDict, iteration: int) -> None:
        pass

    def finalize(self, final_values: ValueDict) -> None:
        """
        Saves the final values of the algorithm to a .npz file
        Returns:

        """
        np.savez_compressed(os.path.join(self._recording_directory, "final_values.npz"), **final_values)
