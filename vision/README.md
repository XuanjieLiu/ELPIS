# ELPIS: visual numerical reasoning

This directory contains the retained visual-addition, subtraction-transfer and
icon-alignment experiments. Configurations use short experiment names such as
`dots_elpis`, `dots_recon_only` and `icon_alignment`.

- [Experiments and paper figures](EXPERIMENTS.md): all 24 VQ groups, icon alignment,
  schedules and coverage boundaries.
- [Reproduction guide](REPRODUCING.md): setup, training, evaluation, outputs and checks.
- [Data release and generation](DATA.md): the original image archive, checksums,
  exact split membership, optional fixed-image regeneration and seeded icon exports.
- [Configuration naming](CONFIGURATION.md): renamed fields and compatibility.
- [Validation record](reproduction/validation.md): what has actually been tested.

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
Full training runs all 20 repetitions. A smoke check establishes executability,
not final paper accuracy. See the reproduction guide for setup and outputs.

## Repository contents

| Path | Contents |
| --- | --- |
| `VQ/exp/<experiment>/` | Addition and subtraction configurations |
| `VQ/batch_train.py` | Addition training and one-step smoke option |
| `VQ/batch_other_task_eval.py` | Subtraction-head training |
| `VQ/eval_pipeline.py` | Addition, matching, orderliness, interpolation and subtraction summaries |
| `LM_align/exp/icon_alignment/` | Icon-adaptation configuration |
| `LM_align/batch_train.py` | Icon-adaptation training |
| `LM_align/statistic_batch.py` | Aggregation of existing training/seen/OOD records |
| `dataset/` | The 14 bundled numerical image datasets |
| `LM_align/checkpoints/` | The fixed VQ model used for adaptation |
| `reproduction/` | Run lists, alias maps, asset lists and validation scope |

Dataset paths are repository-relative. S3Plus is not needed at runtime. Historical
training outputs are not bundled in `VQ/exp`; see the guide for the inputs needed
by evaluation. The naming changes preserve configuration values, model state-dict
keys and result-record formats. Experiment protocol changes and inactive-branch
removal are separate work.
