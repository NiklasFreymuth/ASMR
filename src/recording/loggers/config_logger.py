import datetime
from pprint import pformat

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.abstract_logger import AbstractLogger
from src.recording.loggers.logger_util.logger_util import save_to_yaml
from util.types import *


class ConfigLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm):
        super().__init__(config=config, algorithm=algorithm)
        save_to_yaml(dictionary=config, save_name=self.processed_name,
                     recording_directory=config.get("_recording_structure").get("_recording_dir"))
        self._writer.info(f"Start time: {datetime.datetime.now()}")
        self._writer.info("\n" + pformat(object=config, indent=2))

    def log_iteration(self, recorded_values: ValueDict,
                      iteration: int) -> None:
        pass

    def finalize(self, final_values: ValueDict) -> None:
        pass
