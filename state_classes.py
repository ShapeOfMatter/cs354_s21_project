from dataclasses import dataclass
from dataclasses_json import dataclass_json
from filelock import BaseFileLock, FileLock, SoftFileLock
from typing import Tuple

@dataclass_json
@dataclass(frozen=True)
class TrainingProfile:
    name: str

@dataclass_json
@dataclass(frozen=True)
class AdamTrainingProfile(TrainingProfile):
    learning_rate: float = 1e-03
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-08
    weight_decay: float = 0
    ams_gradient_variant: bool = False

@dataclass_json
@dataclass(frozen=True)
class Settings:
    lock_file: str
    lock_timeout: int
    state_file: str
    source_csvs: Tuple[str]
    sub_graph_choices: Tuple[int]
    deterministic_random_seed: int
    model_name_format: str
    training_profile: AdamTrainingProfile
    max_batch_size: int
    total_lifetime: int

    @staticmethod
    def load(filename: str) -> "Settings":
        with open(filename, 'r') as f:
            return Settings.from_json(f.read())

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(self.to_json(indent=4))

