from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from voice_vector.utility import dataclass_utility
from voice_vector.utility.git_utility import get_commit_id, get_branch_name


@dataclass
class DatasetConfig:
    input_glob: str
    vowel_glob: str
    speaker_dict_path: Path
    num_test: int
    num_times_test: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    hidden_size: int
    layer_num: int
    feature_size: Optional[int] = None
    in_size: int = 80
    out_size: int = 100


@dataclass
class ModelConfig:
    pass


@dataclass
class TrainConfig:
    batchsize: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    num_processes: Optional[int] = None
    optimizer: Dict[str, Any] = field(default_factory=dict(
        name='Adam',
    ))


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags['git-commit-id'] = get_commit_id()
        self.project.tags['git-branch-name'] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
