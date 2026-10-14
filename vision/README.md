# ELPIS: visual numerical reasoning

This directory contains the visual-addition, subtraction-transfer and
icon-alignment experiments. Configurations use short experiment names such as
`dots_elpis`, `dots_recon_only` and `icon_alignment`.

- [Experiments and paper figures](EXPERIMENTS.md): all 42 VQ groups, icon alignment,
  datasets and training schedules.
- [Reproduction guide](REPRODUCING.md): setup, training, evaluation, outputs and checks.
- [Data release and generation](DATA.md): the original image archive, checksums,
  exact split membership, optional fixed-image regeneration and seeded icon exports.
- [Configuration reference](CONFIGURATION.md): model, loss and training parameters.

## Quick start

Run the following commands from `ELPIS/vision` after environment setup.
A CUDA GPU is required for the smoke check.

```bash
# One optimizer update, with automatic removal of temporary outputs.
python VQ/batch_train.py dots_elpis --smoke-test

# Full workflow for one experiment group; wait for each stage to finish.
python VQ/batch_train.py dots_elpis
python VQ/batch_other_task_eval.py dots_elpis
python VQ/eval_pipeline.py dots_elpis
```

The last three commands depend on preceding outputs; run them in order.
Full training runs all 20 repetitions. See the reproduction guide for setup and outputs.

For the Fig. 2B training-ratio sweep, see [EXPERIMENTS.md](EXPERIMENTS.md#fig-2b-training-ratio-sweep).

## Repository contents

| Path | Contents |
| --- | --- |
| `VQ/exp/<experiment>/` | Addition and subtraction configurations |
| `VQ/batch_train.py` | Addition training and one-step smoke option |
| `VQ/batch_other_task_eval.py` | Subtraction-head training |
| `VQ/statistic_batch.py` | Fig. 2B statistics from training-time evaluation records |
| `VQ/eval_pipeline.py` | Addition, matching, orderliness, interpolation and subtraction summaries |
| `LM_align/exp/icon_alignment/` | Icon-adaptation configuration |
| `LM_align/batch_train.py` | Icon-adaptation training |
| `LM_align/statistic_batch.py` | Aggregation of existing training/seen/OOD records |
| `dataset/` | The 14 bundled numerical image datasets |
| `LM_align/checkpoints/` | The fixed VQ model used for adaptation |
| `reproduction/` | Experiment lists, data manifests and data-generation tools |

Dataset paths are relative to this directory. See the reproduction guide for
training, checkpoint selection and evaluation outputs.
