from dataclasses import dataclass
from typing import Optional

from src.modules.abstract_architecture import AbstractArchitecture
from src.algorithms.rl.normalizers.abstract_environment_normalizer import AbstractEnvironmentNormalizer


@dataclass
class SwarmRLCheckpoint:
    architecture: AbstractArchitecture
    normalizer: Optional[AbstractEnvironmentNormalizer]
