from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.recorder import Recorder
from util.types import *


class IterativeExperiment(experiment.AbstractIterativeExperiment):
    def __init__(self):
        super(IterativeExperiment, self).__init__()

        self._algorithm: Optional[AbstractIterativeAlgorithm] = None
        self._recorder: Optional[Recorder] = None
        self._config: Optional[ConfigDict] = None

    def initialize(self, config: ConfigDict, rep: int, logger: cw_logging.LoggerArray) -> None:
        from src.algorithms.get_algorithm import get_algorithm
        from util.initialize_config import initialize_config
        import copy
        import numpy as np
        import torch
        self._config = initialize_config(config=copy.deepcopy(config), repetition=rep)

        # initialize random seeds
        numpy_seed = self._config.get("random_seeds").get("numpy")
        pytorch_seed = self._config.get("random_seeds").get("pytorch")
        if numpy_seed is not None:
            np.random.seed(seed=numpy_seed)
        if pytorch_seed is not None:
            torch.manual_seed(seed=pytorch_seed)

        # initialize the actual algorithm and recording
        self._algorithm = get_algorithm(config=self._config, seed=numpy_seed)
        self._recorder = Recorder(config=self._config, algorithm=self._algorithm)

    def iterate(self, config: ConfigDict, rep: int, n: int) -> ValueDict:
        recorded_values = self._algorithm.fit_and_evaluate()
        scalars = self._recorder.record_iteration(iteration=n, recorded_values=recorded_values)
        return scalars

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if self._recorder is not None:
            try:
                self._recorder.finalize()
            except Exception as e:
                print("Failed finalizing recorder: {}".format(e))


if __name__ == '__main__':
    cw = cluster_work.ClusterWork(IterativeExperiment)
    cw.run()
