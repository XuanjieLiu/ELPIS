# ELPIS: music experiments

This directory contains music models, data-preparation code, experiment
configurations and evaluation scripts. Run music commands with `ELPIS/music`
as the working directory.

- [Transition experiments](experiments/transition/README.md): training, evaluation
  and plotting scripts.
- [Inducement experiments](experiments/inducement/README.md): training, evaluation
  and analysis scripts.
- [Additional experiments](experiments/other_runs/README.md): additional launch
  scripts and results.

Training starts at [`run_training.py`](run_training.py) with a configuration
selected through `--config`. Evaluation starts at
[`run_evaluation.py`](run_evaluation.py) with a trained checkpoint selected through
`--active_checkpoint`. Prepare the data and dependencies and set the data paths
in the selected configuration before running. WAV datasets are not included.
