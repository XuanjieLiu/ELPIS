# Reproducing the numerical experiments

Commands in this guide are submitted from `ELPIS/vision`. On this cluster, every
project Python command must run in a SLURM `srun` step. This includes imports,
unit tests, syntax checks, dataset generation, training and evaluation. An allocation
without an inner `srun` does not guarantee execution on a compute node.

## 1. Environment

Validation used Python 3.10.14, PyTorch 2.4.0+cu121 and torchvision 0.19.0+cu121.
[requirements.txt](requirements.txt) records the observed non-PyTorch dependency
versions. Fresh installation from these instructions has not yet been validated.

To create a separate environment on this cluster, submit the installation inside
a compute step as well. The following assumes conda is installed under `~/miniconda3`:

```bash
export ELPIS_VISION_ROOT="$PWD"
salloc -p ws-ia -N 1 --cpus-per-task=4 \
  --gres=gpu:nvidia-rtx-5000-ada-generation:1 --mem=32G \
  srun bash -lc '
    set -euo pipefail
    source ~/miniconda3/etc/profile.d/conda.sh
    conda create -y -n elpis python=3.10.14
    conda activate elpis
    cd "$ELPIS_VISION_ROOT"
    python -m pip install torch==2.4.0 torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements.txt
  '
export ELPIS_CONDA_ENV=elpis
```

The existing validation environment is `xuanjie`, which is the submission wrapper's
default. Override `ELPIS_CONDA_SH` if conda's activation script is elsewhere.
The wrapper sets `MPLBACKEND=Agg` and disables Python bytecode output. W&B is disabled
by default; online logging is optional and uses externally configured credentials.

## 2. Data and assets

The [dataset manifest](reproduction/datasets.txt) lists the 14 included directories.
Paths are derived from each configuration's location. No access to S3Plus is required.
For the downloadable `vision_data.zip`, integrity verification and optional
regeneration, follow [DATA.md](DATA.md). The packaged original files are the default
reproduction input; generation always writes to a new directory.

- Fixed blue-dot addition uses 68 training pairs and 101+62 held-out pairs. The
  full-data ceiling trains on all 231 pairs; its interpolation queries retain the
  historical fixed-training-set definition.
- Tally and multi-style datasets are included. Gaussian blur is applied online.
- Blue-dot subtraction uses 152 training and 79 held-out pairs.
- Icon images are generated online with the bundled emoji font.

[The alignment asset manifest](reproduction/lm_align_assets.txt) lists DINO weights,
the font and `dots_elpis/1/checkpoint_48000.pt`. Alignment first checks for that VQ
checkpoint under `VQ/exp/dots_elpis/1/`, then falls back to the bundled asset under
`LM_align/checkpoints/vqsps/`. This is a fixed reference model, not an automatic
selection from newly trained addition runs.

The clean `VQ/exp` directories contain configurations only. Complete trained VQ
groups and subtraction logs are not bundled. Either train them or place archived
checkpoints and records under the mapped canonical directory, preserving repetition
numbers and filenames. The [alias registry](reproduction/experiment_aliases.json)
provides exact historical mappings.

## 3. Fast execution checks

A check of one addition minibatch requires no trained checkpoint:

```bash
sbatch common_gpu_py.sbatch VQ/batch_train.py dots_elpis --smoke-test
```

It runs the original loss, backward pass and Adam update, checks finite values and
changed parameters, and removes its temporary outputs. To check all eight dataset
representatives:

```bash
bash reproduction/smoke_train.sh
```

A subtraction check needs addition repetition 1's selected checkpoint and training
record:

```bash
sbatch common_gpu_py.sbatch VQ/batch_other_task_eval.py dots_elpis --smoke-test
```

It performs one training and one evaluation minibatch in a temporary directory;
the supplied pretrained inputs are retained. These checks do not reproduce final
paper scores, convergence or the full repeated-run statistics.

## 4. Addition → subtraction → evaluation

For one experiment group, use SLURM dependencies so downstream stages only start
after successful completion:

```bash
train_job=$(sbatch --parsable common_gpu_py.sbatch VQ/batch_train.py dots_elpis)
minus_job=$(sbatch --parsable --dependency="afterok:$train_job" \
  common_gpu_py.sbatch VQ/batch_other_task_eval.py dots_elpis)
sbatch --dependency="afterok:$minus_job" \
  common_gpu_py.sbatch VQ/eval_pipeline.py dots_elpis
```

Each training command covers all 20 addition repetitions. Subtraction-head counts
and schedules differ by family; see [EXPERIMENTS.md](EXPERIMENTS.md). To submit all
retained groups, use `reproduction/runs.txt` as the list and apply the same three-stage
sequence to each group. Use your site's job limits when deciding concurrency.

You can pass multiple short names to the training, subtraction and evaluation
entry points. VQ evaluation without names evaluates all 24 retained groups:

```bash
sbatch common_gpu_py.sbatch VQ/eval_pipeline.py
```

That command requires all configured inputs. A missing experiment's records or
checkpoint cause failure rather than being silently excluded from the statistics.

### Inputs, checkpoint selection and outputs

For `dots_elpis`, the main files are:

```text
VQ/exp/dots_elpis/
  train_config.py
  other_tasks_config.py
  1/ ... 20/
    Train_record.txt
    Eval_record.txt
    checkpoint_<epoch>.pt
    curr_model.pt
    1/ ... <num_heads_per_model>/
      minus_16_1.1_train_record.txt
      minus_16_1.1_eval_record.txt
      minus_16_1.1_minus_model.pt
  PIPELINE_EVAL/
    all_results.json
    all_results_summary.json
    all_results_details.json
    orderliness/
```

`all_results.json` contains per-repetition values; the summary includes mean, standard
deviation, extrema and count. The details file records selected checkpoint filenames.
Additional training plots and records are written under each repetition.

The pipeline chooses checkpoints using
`eval_config.optimal_checkpoint_finding_config` in `train_config.py`. Subtraction
uses its own selection config, or the historical defaults. A repetition-local
`specific_checkpoint.txt` overrides selection and contains an integer epoch number.
Thus checkpoints may differ between pipeline and subtraction unless deliberately
specified. Keep `Train_record.txt` when distributing selected checkpoints.

Output metric names retain the historical schema:

| Output key pattern | Actual computation |
| --- | --- |
| `eval_set_one2n_accu` | Majority-vote code-to-label addition accuracy |
| `eval_set_one2one_accu` | Hungarian code-to-label addition accuracy |
| `eval_set_emb_select_accu` | Nearest sampled candidate-code accuracy |
| `orderliness` | Label-ordered nearest-neighbour score |
| `train_interpolate_modedict` | Interpolation with majority-vote labels |
| `train_interpolate_selection` | Interpolation with sampled candidate matching |
| `substraction_one` | Subtraction head 1 across the 20 addition repetitions |

Prefixes come from the config's evaluation `name` values. Keys containing `cycle`
use decode–encode predictions. Keep these variants distinct when comparing results.
This naming release preserves the historical evaluator; it does not change HM/DM
or sampling definitions to resolve outstanding manuscript-protocol differences.

### Resources, runtime and resuming

The wrapper requests one RTX 5000 Ada, four CPU cores, 32 GB RAM and a 20-hour walltime.
Full training runs 20 repetitions sequentially; total runtime has not been measured
for this release and completion within one allocation is not guaranteed. Inspect
`slurm-<jobid>.out` and `.err`, and query jobs with `squeue -u "$USER"`.

Re-running a training command uses the existing records/current model to resume.
Addition and subtraction recreate their Adam optimizer, so this is not an exact
optimizer/RNG-state continuation. Keep this limitation in mind when comparing
interrupted and uninterrupted runs. Override resource directives with `sbatch`
options as needed; do not run the project on the login node to work around limits.

## 5. Icon adaptation

```bash
align_job=$(sbatch --parsable common_gpu_py.sbatch \
  LM_align/batch_train.py icon_alignment)
sbatch --dependency="afterok:$align_job" common_gpu_py.sbatch \
  LM_align/statistic_batch.py icon_alignment
```

Training produces, for each repetition under `LM_align/exp/icon_alignment/`,
`Train_record.txt`, `Val_record.txt`, `Val_ood_record.txt` and model checkpoints.
It uses 20 repetitions and 30 epochs with online training/seen/unseen icon images.

`statistic_batch.py` reads these three records at the epoch with the highest training
accuracy (excluding epoch 0). It writes `all_results.json`,
`all_results_summary.json` and `all_results_details.json` at the group root.
It does not load a checkpoint or perform fresh inference. It currently summarizes
all 20 repetitions, without the manuscript's converged-run filtering.

## 6. Expected results and verification scope

Successful smoke checks print `SMOKE PASS` and leave no training artifacts.
Previous full evaluation of `dots_full_data` matched all 360 historical values
(18 metrics × 20 repetitions) exactly. The [validation record](reproduction/validation.md)
summarizes that check and the naming-refactor checks.

These execution/regression checks are distinct from reproducing the paper's final
numbers by retraining. New training trajectories are stochastic and original
initialization seeds were not recorded in the retained configs. Final paper plots,
the training-ratio sweep and random baselines still need the work identified in
[the experiment coverage table](EXPERIMENTS.md). Do not infer an exact expected score
from an unrelated metric key or from a one-step run.
