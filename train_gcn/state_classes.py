from dataclasses import dataclass
from dataclasses_json import dataclass_json
from filelock import BaseFileLock, FileLock, SoftFileLock
from os.path import isfile
from typing import List, Tuple

def read_whole_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()

def write_whole_file(filename: str, contents: str) -> None:
    with open(filename, 'w') as f:
        f.write(contents)

class _Settings:
    '''Mostly exists to keep mypy happy because it doesn't understand dataclasses_json.'''
    @staticmethod
    def from_json(j: str):
        raise Exception("Not implemented.")
    
    def to_json(self, *, indent: int=0):
        raise Exception("Not implemented.")

    @classmethod
    def load(cls, filename: str) -> 'Settings':
        return cls.from_json(read_whole_file(filename))

    def save(self, filename: str) -> None:
        write_whole_file(filename, self.to_json(indent=2))

@dataclass_json
@dataclass(frozen=True)
class TrainingProfile(_Settings):
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
class Settings(_Settings):
    lock_file: str
    lock_timeout: int
    source_csvs: Tuple[str]
    sub_graph_choices: Tuple[int]
    deterministic_random_seed: int
    model_filename: str
    training_profile: AdamTrainingProfile
    max_batch_size: int
    total_lifetime: int
    num_samples: int
    sample_size: int
    epochs: int



