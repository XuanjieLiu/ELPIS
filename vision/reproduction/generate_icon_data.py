"""Freeze a seeded sample from the existing online icon generator for inspection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A new output directory.')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--split', choices=['seen', 'unseen'], required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run icon generation inside a SLURM srun job step.')
    if args.samples < 1:
        parser.error('--samples must be positive')
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    import PIL
    from LM_align.synthData.SynthCommon_win import gen_recurrent_data, OBJ_LIST, OBJ_LIST_2
    font = ROOT / 'LM_align/synthData/fonts/seguiemj.ttf'
    if not font.is_file():
        raise FileNotFoundError(f'Required font missing; do not substitute a fallback font: {font}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    objects = OBJ_LIST if args.split == 'seen' else OBJ_LIST_2
    images = gen_recurrent_data(args.samples, obj_list=objects)
    hashes = []
    for index, (a, b, c, label) in enumerate(images):
        sample = output / f'{index:06d}'
        sample.mkdir()
        for role, image in [('a', a), ('b', b), ('c', c)]:
            image.save(sample / f'image_{role}.png')
        (sample / 'label.json').write_text(json.dumps(label, ensure_ascii=False, sort_keys=True) + '\n')
        for path in sorted(sample.iterdir()):
            hashes.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output).as_posix()}\n')
    metadata = {
        'seed': args.seed, 'samples': args.samples, 'split': args.split,
        'objects': objects, 'canvas': [224, 224], 'python': sys.version.split()[0],
        'pillow': PIL.__version__, 'numpy': np.__version__,
        'font_sha256': hashlib.sha256(font.read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256((ROOT / 'LM_align/synthData/SynthCommon_win.py').read_bytes()).hexdigest(),
        'purpose': 'New seeded inspection/evaluation fixture, not a recovery of historical online samples.',
    }
    (output / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'files.sha256').write_text(''.join(hashes))
    print(json.dumps({'output': str(output), **metadata}, indent=2))


if __name__ == '__main__':
    main()
