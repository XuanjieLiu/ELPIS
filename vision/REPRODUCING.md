# Reproducing the numerical experiments

Run commands from `ELPIS/vision` in an environment with the required dependencies.
Use a CUDA GPU for training and the smoke checks.

## 1. Environment

The reference environment uses Python 3.10.14, PyTorch 2.4.0+cu121 and
torchvision 0.19.0+cu121. Additional dependencies are listed in
[requirements.txt](requirements.txt).

Create a conda environment and install the dependencies:

```bash
conda create -y -n elpis python=3.10.14
conda activate elpis
python -m pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export MPLBACKEND=Agg
```

W&B is disabled by default; online logging is optional and uses externally
configured credentials.

## 2. Data and assets

The [dataset manifest](reproduction/datasets.txt) lists the 14 included directories.
Paths are derived from each configuration's location.
For the downloadable `vision_data.zip`, integrity verification and optional
regeneration, follow [DATA.md](DATA.md). The packaged original files are the default
reproduction input; generation always writes to a new directory.

- Fixed blue-dot addition uses 68 training pairs and 101+62 held-out pairs. The
  full-data ceiling trains on all 231 pairs. Interpolation queries use the fixed
  68-pair training subset.
- Tally and multi-style datasets are included. Gaussian blur is applied online.
- Blue-dot subtraction uses 152 training and 79 held-out pairs.
- Icon images are generated online with the bundled emoji font.

On first use, the DINO loader downloads the official ViT-S/8 weights and caches
them under the PyTorch Hub cache (`TORCH_HOME` controls its location). Subsequent
runs reuse the cached file. For offline use, prepopulate the cache on a connected
machine or call `load_dino_vit_s8(checkpoint_path=...)` with a local checkpoint.
The official download URL and pinned SHA-256 prefix are in
[`loader_dino.py`](LM_align/loader_dino.py).

For icon alignment, choose a best-performing checkpoint from any repetition of
a trained addition experiment. In
[`LM_align/exp/icon_alignment/config.py`](LM_align/exp/icon_alignment/config.py),
set `VQSPS.EXP_NAME` to the experiment name and `VQSPS.CHECK_POINT_NAME` to
`<repetition>/checkpoint_<epoch>.pt`. For example:

```python
'VQSPS': {
    'EXP_NAME': 'dots_elpis',
    'CHECK_POINT_NAME': '1/checkpoint_48000.pt',
},
```

The example reads `VQ/exp/dots_elpis/1/checkpoint_48000.pt`. A pretrained example
and the icon font are included in the [asset manifest](reproduction/lm_align_assets.txt).

The `VQ/exp` directories contain experiment configurations. Training creates the
checkpoints and records used by evaluation. To evaluate existing trained models,
place their checkpoints and records under `VQ/exp/<experiment>/<repetition>/`
using the filenames shown below.

## 3. Fast execution checks

A check of one addition minibatch requires no trained checkpoint:

```bash
python VQ/batch_train.py dots_elpis --smoke-test
```

It runs one loss calculation, backward pass and Adam update, checks finite values and
changed parameters, and removes its temporary outputs. To check all eight dataset
representatives:

```bash
bash reproduction/smoke_train.sh
```

A subtraction check needs addition repetition 1's selected checkpoint and training
record:

```bash
python VQ/batch_other_task_eval.py dots_elpis --smoke-test
```

It performs one training and one evaluation minibatch in a temporary directory;
the supplied pretrained inputs are retained.

## 4. Addition → subtraction → evaluation

For one experiment group, run each stage after the preceding stage completes:

```bash
python VQ/batch_train.py dots_elpis
python VQ/batch_other_task_eval.py dots_elpis
python VQ/eval_pipeline.py dots_elpis
```

Each training command covers all 20 addition repetitions. Subtraction-head counts
and schedules differ by family; see [EXPERIMENTS.md](EXPERIMENTS.md). For the 24 fixed-split groups, use
`reproduction/pipeline_runs.txt` and apply this three-stage sequence to each group.
The complete list of 42 groups is in `reproduction/runs.txt`. Choose concurrency to fit your available resources.

You can pass multiple short names to the training, subtraction and evaluation
entry points. VQ evaluation without names evaluates the 24 fixed-split groups:

```bash
python VQ/eval_pipeline.py
```

Prepare each group's trained checkpoints and subtraction records before running
the full pipeline.

### Fig. 2B: training-ratio sweep

Train the 18 groups listed in `reproduction/fig2b_runs.txt`:

```bash
while IFS= read -r run; do
    python VQ/batch_train.py "$run"
done < reproduction/fig2b_runs.txt
python VQ/statistic_batch.py
```

For a single group:

```bash
python VQ/batch_train.py dots_elpis_train_10pct
python VQ/statistic_batch.py dots_elpis_train_10pct
```

Each repetition trains on a random subset of the 231 addition triplets and
records held-out metrics in `plus_eval.txt`. Statistics use those records at
the selected checkpoint and write `statistic_<metric>.txt` under each group.
See [the experiment table](EXPERIMENTS.md#fig-2b-training-ratio-sweep) for the ratios.

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
uses its own selection config, or the defaults in `VQ/common_func.py`. A repetition-local
`specific_checkpoint.txt` overrides selection and contains an integer epoch number.
Thus checkpoints may differ between pipeline and subtraction unless deliberately
specified. Keep `Train_record.txt` when distributing selected checkpoints.

Output metric names and definitions:

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

### Runtime and resuming

Full training runs 20 repetitions sequentially.

Re-running a training command uses the existing records/current model to resume.
Addition and subtraction recreate their Adam optimizer when resuming.

## 5. Icon adaptation

```bash
python LM_align/batch_train.py icon_alignment
python LM_align/statistic_batch.py icon_alignment
```

Training produces, for each repetition under `LM_align/exp/icon_alignment/`,
`Train_record.txt`, `Val_record.txt`, `Val_ood_record.txt` and model checkpoints.
It uses 20 repetitions and 30 epochs with online training/seen/unseen icon images.

`statistic_batch.py` reads these three records at the epoch with the highest training
accuracy (excluding epoch 0). It writes `all_results.json`,
`all_results_summary.json` and `all_results_details.json` at the group root.
It summarizes all 20 repetitions from their recorded metrics.

## 6. Checking execution

Successful smoke checks print `SMOKE PASS` and leave no training artifacts.
Full evaluation writes per-repetition metrics, aggregate statistics and selected
checkpoint filenames to the experiment's `PIPELINE_EVAL/` directory.
