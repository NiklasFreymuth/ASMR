import os
from timeit import default_timer as timer

import numpy as np

import util.keys as Keys
from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.abstract_logger import AbstractLogger
from src.recording.loggers.logger_util.logger_util import save_to_yaml
from util.function import add_to_dictionary, prefix_keys
from util.types import *


def _get_memory_usage(scope: str) -> float:
    import resource
    if scope == "self":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    elif scope == "children":
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    elif scope == "total":
        import psutil
        total_memory = psutil.virtual_memory()[2]
        return total_memory
    else:
        raise ValueError(f"Scope must be either 'self' or 'total', given {scope}")



class ScalarsLogger(AbstractLogger):
    """
    A basic logger for scalar metrics
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        self._previous_duration = timer()  # start time
        self._cumulative_duration = 0
        self._all_scalars = {}

    def log_iteration(self, recorded_values: ValueDict,
                      iteration: int) -> None:
        assert Keys.SCALARS in recorded_values.keys()

        # adapt in-place to feed this forward to subsequent loggers
        recorded_values[Keys.SCALARS] = recorded_values.get(Keys.SCALARS) | self._get_default_scalars()
        scalars = recorded_values.get(Keys.SCALARS)

        self._write(scalars=scalars)
        save_to_yaml(dictionary=scalars,
                     save_name=self.processed_name,
                     recording_directory=self._recording_directory)
        self._all_scalars = add_to_dictionary(dictionary=self._all_scalars, new_scalars=scalars)

    def finalize(self, final_values: ValueDict) -> None:
        """
        Finalizes the recording, e.g., by saving certain things to disk or by postpressing the results in one way or
        another.
        Returns:

        """
        save_to_yaml(dictionary=self._all_scalars,
                     save_name=self.processed_name,
                     recording_directory=self._recording_directory)

    def _write(self, scalars: Dict[str, Any]):
        for key, value in scalars.items():
            if isinstance(value, (float, np.float32)) or (isinstance(value, np.ndarray) and value.ndim == 0):
                self._writer.info(msg=key.title().replace("_", " ") + ": {:.3f}".format(value))
            else:
                self._writer.info(msg=key.title().replace("_", " ") + ": " + str(value))

    def _get_default_scalars(self):
        duration = self._get_duration()
        self._cumulative_duration += duration
        default_scalars = {"iteration_runtime (seconds)": duration,
                           "cumulative_runtime (seconds)": self._cumulative_duration}
        if os.name == "posix":  # record memory usage per iteration. Only works on linux
            default_scalars["self_memory_usage"] = _get_memory_usage(scope="self")
            default_scalars["child_memory_usage"] = _get_memory_usage(scope="children")
            default_scalars["total_memory_usage"] = _get_memory_usage(scope="total")

        default_scalars = prefix_keys(dictionary=default_scalars, prefix="default")
        return default_scalars

    def _get_duration(self) -> float:
        current_duration = timer()
        duration = current_duration - self._previous_duration
        self._previous_duration = current_duration
        return duration
