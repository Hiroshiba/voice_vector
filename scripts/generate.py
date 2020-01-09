import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy
import yaml
from tqdm import tqdm

from utility.save_arguments import save_arguments
from voice_vector.config import Config
from voice_vector.dataset import create_dataset
from voice_vector.generator import Generator


def _extract_number(f):
    s = re.findall(r'\d+', str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
        model_dir: Path,
        iteration: int = None,
        prefix: str = 'predictor_',
):
    if iteration is None:
        paths = model_dir.glob(prefix + '*.npz')
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + '{}.npz'.format(iteration))
        assert model_path.exists()
    return model_path


def generate(
        model_dir: Path,
        model_iteration: Optional[int],
        model_config: Optional[Path],
        output_dir: Path,
        data_par_speaker: int,
        use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / 'config.yaml'

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / 'arguments.yaml', generate, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    dataset = create_dataset(config.dataset)['train']
    features_dict = defaultdict(list)
    for data in tqdm(dataset, desc='generate'):
        speaker_num = data['target']
        if len(features_dict[speaker_num]) >= data_par_speaker:
            continue

        feature = generator.generate(data['input'])
        features_dict[speaker_num].append(feature)

    for speaker_num, features in features_dict.items():
        for i, feature in enumerate(features):
            path = output_dir / f'{speaker_num}-{i}.npy'
            numpy.save(path, feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=Path)
    parser.add_argument('--model_iteration', type=int)
    parser.add_argument('--model_config', type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--data_par_speaker', type=int, default=5)
    parser.add_argument('--use_gpu', action='store_true')
    generate(**vars(parser.parse_args()))
