from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer
from util.types import *


class DummySwarmEnvironmentNormalizer(AbstractEnvironmentNormalizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observations: InputBatch,
              **kwargs) -> Union[InputBatch, Tuple]:
        if len(kwargs) > 0:
            return observations, *kwargs.values()
        else:
            return observations

    def update_and_normalize(self, observations: InputBatch,
                             **kwargs) -> Union[InputBatch, Tuple]:
        if len(kwargs) > 0:
            return observations, *kwargs.values()
        else:
            return observations

    def update_observations(self, observations: InputBatch) -> None:
        pass

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        return observations
