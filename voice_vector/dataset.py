import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data.dataset import Dataset

from voice_vector.config import DatasetConfig
from voice_vector.utility.dataset_utility import default_convert


@dataclass
class DatasetInputData:
    input_path: Path
    vowel_path: Path
    speaker_num: int


class InputTargetDataset(Dataset):
    def __init__(
            self,
            datas: Sequence[DatasetInputData],
    ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        input = SamplingData.load(data.input_path).array
        vowel = numpy.squeeze(SamplingData.load(data.vowel_path).array)
        speaker_num = data.speaker_num

        assert len(vowel) <= len(input), f'{data.input_path.stem} cannot be processed.'

        input_vowel = input[:len(vowel)][vowel]
        i = numpy.random.randint(len(input_vowel))

        return default_convert(dict(
            input=input_vowel[i],
            target=speaker_num,
        ))


def create_dataset(config: DatasetConfig):
    input_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.input_glob))}
    assert len(input_paths) > 0

    filenames = list(sorted(input_paths.keys()))

    vowel_paths = {Path(p).stem: Path(p) for p in glob.glob(str(config.vowel_glob))}
    assert set(vowel_paths.keys()) == set(filenames)

    filename_each_speaker: Dict[str, List[str]] = json.load(config.speaker_dict_path.open())
    speaker_nums = {
        fn: speaker_num
        for speaker_num, (_, fns) in enumerate(filename_each_speaker.items())
        for fn in fns
    }
    assert set(filenames).issubset(set(speaker_nums.keys()))

    inputs = [
        DatasetInputData(
            input_path=input_paths[filename],
            vowel_path=vowel_paths[filename],
            speaker_num=speaker_nums[filename],
        )
        for filename in filenames
    ]

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(inputs)

    tests, trains = inputs[:config.num_test], inputs[config.num_test:]
    train_tests = trains[:config.num_test]

    def dataset_wrapper(datas):
        return InputTargetDataset(
            datas=datas,
        )

    return {
        'train': dataset_wrapper(trains),
        'test': dataset_wrapper(tests),
        'train_test': dataset_wrapper(train_tests),
    }
