import argparse
import glob
import multiprocessing
from functools import partial
from pathlib import Path

import numpy
import yaml
from acoustic_feature_extractor.data.phoneme import PhonemeType, phoneme_type_to_class
from tqdm import tqdm

from utility.save_arguments import save_arguments


def process(
        path: Path,
        output_directory: Path,
        phoneme_type: PhonemeType,
        rate: int,
):
    phoneme_class = phoneme_type_to_class[phoneme_type]
    phonemes = phoneme_class.load_julius_list(path)

    length = int(round(phonemes[-1].end * rate)) + 1
    array = numpy.zeros((length,), dtype=numpy.bool)

    for p in phonemes:
        s = int(round(p.start * rate))
        e = int(round(p.end * rate))
        array[s:e + 1] = True

    out = output_directory / (path.stem + '.npy')
    numpy.save(str(out), dict(array=array, rate=rate))


def extract_vowel(
        input_glob: str,
        output_directory: Path,
        phoneme_type: PhonemeType,
        rate: int,
):
    output_directory.mkdir(exist_ok=True)
    yaml.SafeDumper.add_representer(
        PhonemeType,
        lambda dumper, data: dumper.represent_scalar('!PhonemeType', str(data)),
    )
    save_arguments(output_directory / 'arguments.yaml', extract_vowel, locals())

    paths = [Path(p) for p in glob.glob(input_glob)]
    _process = partial(
        process,
        output_directory=output_directory,
        phoneme_type=phoneme_type,
        rate=rate,
    )

    pool = multiprocessing.Pool()
    list(tqdm(pool.imap_unordered(_process, paths), total=len(paths)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_glob', type=str, required=True)
    parser.add_argument('--output_directory', type=Path, required=True)
    parser.add_argument('--phoneme_type', type=PhonemeType, default=PhonemeType.seg_kit)
    parser.add_argument('--rate', type=int, default=100)
    extract_vowel(**vars(parser.parse_args()))
