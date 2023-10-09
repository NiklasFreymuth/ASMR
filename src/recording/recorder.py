from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.get_loggers import get_loggers
from src.recording.loggers.abstract_logger import AbstractLogger
from src.recording.loggers.logger_util.get_logging_writer import get_logging_writer
from util import keys
from util.types import *


class Recorder:
    """
    Records the algorithm whenever called by computing common recording values and then delegating the
    recording itself to different recorders.
    """

    def __init__(self, config: ConfigDict,
                 algorithm: AbstractIterativeAlgorithm):
        """
        Initialize the recorder, which itself is an assortment of loggers
        Args:
            config:
            algorithm:
        """
        self._loggers: List[AbstractLogger] = get_loggers(config=config, algorithm=algorithm)
        self._algorithm = algorithm
        self._writer = get_logging_writer(self.__class__.__name__,
                                          recording_directory=config.get("_recording_structure").get("_recording_dir"))
        self._recording_dict = config.get("recording", {})

    def record_iteration(self, iteration: int, recorded_values: ValueDict) -> ValueDict:
        self._writer.info("Recording iteration {}".format(iteration))

        for logger in self._loggers:
            try:
                logger.log_iteration(recorded_values=recorded_values,
                                     iteration=iteration)
            except Exception as e:
                self._writer.error("Error with logger '{}': {}".format(logger.__class__.__name__, repr(e)))
        self._writer.info("Finished recording iteration {}\n".format(iteration))
        scalars = recorded_values.get(keys.SCALARS, {})
        return scalars

    def finalize(self) -> None:
        """
        Finalizes the recording process.

        This method does the following:
        1. Notifies that the experiment has finished.
        2. Retrieves the final values from the algorithm (if specified).
        3. Requests each logger to finalize, potentially saving data or post-processing results.
        4. Removes any writer associated with the logger.
        5. Clears the writer's handlers and deletes the writer from the instance.

        Exceptions are handled gracefully, logging any errors encountered during the process.
        """
        self._log_experiment_completion()
        final_values = self._retrieve_final_values()
        self._finalize_loggers(final_values)
        self._cleanup_writer()

    def _log_experiment_completion(self) -> None:
        """Logs the completion of the experiment."""
        self._writer.info("Finished experiment! Finalizing recording")

    def _retrieve_final_values(self) -> dict:
        """
        Retrieves final values from the algorithm based on the recording settings.

        Returns:
            dict: Final values from the algorithm or an empty dictionary if not applicable or in case of an error.
        """
        try:
            if self._recording_dict.get("end_of_training_evaluation"):
                return self._algorithm.get_final_values()
        except Exception as e:
            self._writer.error(f"Error getting final values from algorithm: {e}. Continuing with empty dict.")
        return {}

    def _finalize_loggers(self, final_values: dict) -> None:
        """Finalizes each logger in the instance."""
        for logger in self._loggers:
            try:
                logger.finalize(final_values=final_values)
                logger.remove_writer()
            except Exception as e:
                self._writer.error(f"Error with logger '{logger.__class__.__name__}': {repr(e)}")

    def _cleanup_writer(self) -> None:
        """Cleans up the writer by removing its handlers and deleting it from the instance."""
        self._writer.info("Finalized recording.")
        self._writer.handlers = []
        del self._writer
