import abc
import pathlib

from util.types import *


class AbstractEnvironmentNormalizer(abc.ABC):

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, observations: InputBatch) -> InputBatch:
        raise NotImplementedError

    def update_and_normalize(self, observations: InputBatch, **kwargs) -> Union[Tuple, InputBatch]:
        raise NotImplementedError

    def update_observations(self, observations: InputBatch):
        raise NotImplementedError

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        raise NotImplementedError

    def save(self, destination_path: pathlib.Path) -> None:
        """
        Saves the current normalizers to a checkpoint file.

        Args:
            destination_path: the path to checkpoint to
        Returns:

        """
        import pickle as pkl
        with destination_path.open("wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def load(checkpoint_path: pathlib.Path) -> "AbstractEnvironmentNormalizer":
        """
        Loads existing normalizers from a checkpoint.
        Args:
            checkpoint_path: The checkpoints directory of a previous experiment.

        Returns: A new normalizer object with the loaded normalization parameters.

        """
        import pickle as pkl
        import pathlib

        checkpoint_path = pathlib.Path(checkpoint_path)
        with checkpoint_path.open('rb') as f:  # load the file, create a new normalizer object and return it
            return pkl.load(f)
