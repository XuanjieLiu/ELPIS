# ELPIS

Code and experiment configurations for visual numerical reasoning and music
representation learning.

| Directory | Contents | Starting point |
| --- | --- | --- |
| [`vision/`](vision/) | Visual addition, subtraction transfer and icon alignment | [Vision guide](vision/README.md) |
| [`music/`](music/) | Music training, evaluation, data preparation and experiment configurations | `run_training.py`, `run_evaluation.py` |

## Visual experiments

Start with the [reproduction guide](vision/REPRODUCING.md) for environment setup,
training and evaluation. The [experiment table](vision/EXPERIMENTS.md) maps the
42 visual experiment configurations to the paper figures.
The [data guide](vision/DATA.md) covers the image archive, checksums, fixed splits
and optional data generation.

Run a one-step training check from `vision/`:

```bash
cd vision
python VQ/batch_train.py dots_elpis --smoke-test
```

Full training and evaluation commands are documented in the reproduction guide.

## Music experiments

Run music commands with `music/` as the working directory. Training starts at
[`run_training.py`](music/run_training.py), with an explicit configuration such as
[`cfg_isymm.yaml`](music/cfg_isymm.yaml). Evaluation starts at
[`run_evaluation.py`](music/run_evaluation.py) and requires a trained checkpoint.
Additional configurations and instructions are under [`experiments/`](music/experiments/).

Prepare the music data and dependencies, then set the data paths in the selected
configuration. WAV datasets are not included in the repository.
