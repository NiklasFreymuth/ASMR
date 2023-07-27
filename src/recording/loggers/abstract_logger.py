import os
from abc import ABC, abstractmethod

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.logger_util.get_logging_writer import get_logging_writer
from src.recording.loggers.logger_util.logger_util import process_logger_name
from util.function import get_from_nested_dict
from util.types import *


class AbstractLogger(ABC):
    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm):
        self._config = config
        self._algorithm = algorithm
        self._recording_directory: str = get_from_nested_dict(dictionary=config,
                                                              list_of_keys=["_recording_structure", "_recording_dir"],
                                                              default_return="reports/example/",)
        os.makedirs(self._recording_directory, exist_ok=True)
        self._writer = get_logging_writer(writer_name=self.processed_name,
                                          recording_directory=self._recording_directory)

    @abstractmethod
    def log_iteration(self, recorded_values: ValueDict,
                      iteration: int) -> None:
        """
        Log the current training iteration of the algorithm instance.
        Args:
            recorded_values: Metrics and other information that was computed by previous loggers
            iteration: The current algorithm iteration. Is provided for internal consistency, since we may not want to
              record every algorithm iteration

        Returns:

        """
        raise NotImplementedError

    def finalize(self) -> None:
        """
        Finalizes the recording, e.g., by saving certain things to disk or by postpressing the results in one way or
        another.
        Returns:

        """
        raise NotImplementedError

    def remove_writer(self) -> None:
        self._writer.handlers = []
        del self._writer

    @property
    def processed_name(self) -> str:
        return process_logger_name(self.__class__.__name__)
