# Experiments and paper correspondence

All paths below are relative to `vision/`. Each name denotes an experiment group;
numbered directories `1/`–`20/` denote independent repetitions.

## VQ experiments

Each row links to its training configuration. The same directory contains
`other_tasks_config.py` for subtraction. There are 42 VQ groups, each with 20 repetitions.

| Experiment | Dataset | Model / condition | Paper correspondence |
| --- | --- | --- | --- |
| [eastern_tally_elpis](VQ/exp/eastern_tally_elpis/train_config.py) | Eastern tally marks | ELPIS (Order A + B) | Figs. S2–S3 |
| [eastern_tally_no_symmetry](VQ/exp/eastern_tally_no_symmetry/train_config.py) | Eastern tally marks | Without symmetry | Figs. S2–S3 |
| [eastern_tally_full_data](VQ/exp/eastern_tally_full_data/train_config.py) | Eastern tally marks | Full-data ceiling | Figs. S2–S3 |
| [eastern_tally_recon_only](VQ/exp/eastern_tally_recon_only/train_config.py) | Eastern tally marks | Reconstruction only | Figs. S2–S3 |
| [western_tally_elpis](VQ/exp/western_tally_elpis/train_config.py) | Western tally marks | ELPIS (Order A + B) | Figs. S4–S5 |
| [western_tally_no_symmetry](VQ/exp/western_tally_no_symmetry/train_config.py) | Western tally marks | Without symmetry | Figs. S4–S5 |
| [western_tally_full_data](VQ/exp/western_tally_full_data/train_config.py) | Western tally marks | Full-data ceiling | Figs. S4–S5 |
| [western_tally_recon_only](VQ/exp/western_tally_recon_only/train_config.py) | Western tally marks | Reconstruction only | Figs. S4–S5 |
| [multistyle_elpis](VQ/exp/multistyle_elpis/train_config.py) | 16 shape–colour styles | ELPIS (Order A + B) | Figs. S8–S9 |
| [multistyle_no_symmetry](VQ/exp/multistyle_no_symmetry/train_config.py) | 16 shape–colour styles | Without symmetry | Figs. S8–S9 |
| [dots_elpis](VQ/exp/dots_elpis/train_config.py) | Fixed blue dots | ELPIS (Order A + B) | Figs. 2A, 3, S1; A+B reference in S10 |
| [dots_fullsymm_assoc](VQ/exp/dots_fullsymm_assoc/train_config.py) | Fixed blue dots | FullsymmAssoc | Fig. S10 |
| [dots_no_symmetry](VQ/exp/dots_no_symmetry/train_config.py) | Fixed blue dots | Without symmetry | Figs. 2A, 3, S1 |
| [dots_mixed_order](VQ/exp/dots_mixed_order/train_config.py) | Fixed blue dots | Mixed order | Fig. S10 |
| [dots_order_a](VQ/exp/dots_order_a/train_config.py) | Fixed blue dots | Order A | Fig. S10 |
| [dots_order_b](VQ/exp/dots_order_b/train_config.py) | Fixed blue dots | Order B | Fig. S10 |
| [multistyle_full_data](VQ/exp/multistyle_full_data/train_config.py) | 16 shape–colour styles | Full-data ceiling | Figs. S8–S9 |
| [dots_full_data](VQ/exp/dots_full_data/train_config.py) | Fixed blue dots | Full-data ceiling | Figs. 2A, 3, S1 |
| [dots_recon_only](VQ/exp/dots_recon_only/train_config.py) | Fixed blue dots | Reconstruction only | Figs. 2A, 3, S1 |
| [multistyle_recon_only](VQ/exp/multistyle_recon_only/train_config.py) | 16 shape–colour styles | Reconstruction only | Figs. S8–S9 |
| [blurred_dots_elpis](VQ/exp/blurred_dots_elpis/train_config.py) | Online-blurred blue dots | ELPIS (Order A + B) | Figs. S6–S7 |
| [blurred_dots_full_data](VQ/exp/blurred_dots_full_data/train_config.py) | Online-blurred blue dots | Full-data ceiling | Figs. S6–S7 |
| [blurred_dots_no_symmetry](VQ/exp/blurred_dots_no_symmetry/train_config.py) | Online-blurred blue dots | Without symmetry | Figs. S6–S7 |
| [blurred_dots_recon_only](VQ/exp/blurred_dots_recon_only/train_config.py) | Online-blurred blue dots | Reconstruction only | Figs. S6–S7 |

## Fig. 2B: training-ratio sweep

The 18 configurations compare ELPIS and the no-symmetry baseline at training
fractions of 10% through 90%. Each repetition randomly partitions the 231 fixed
blue-dot addition triplets; training uses `int(231 * train_data_ratio)` triplets
and evaluation uses the remainder.

| Training fraction | Without symmetry | ELPIS |
| --- | --- | --- |
| 10% | [dots_no_symmetry_train_10pct](VQ/exp/dots_no_symmetry_train_10pct/train_config.py) | [dots_elpis_train_10pct](VQ/exp/dots_elpis_train_10pct/train_config.py) |
| 20% | [dots_no_symmetry_train_20pct](VQ/exp/dots_no_symmetry_train_20pct/train_config.py) | [dots_elpis_train_20pct](VQ/exp/dots_elpis_train_20pct/train_config.py) |
| 30% | [dots_no_symmetry_train_30pct](VQ/exp/dots_no_symmetry_train_30pct/train_config.py) | [dots_elpis_train_30pct](VQ/exp/dots_elpis_train_30pct/train_config.py) |
| 40% | [dots_no_symmetry_train_40pct](VQ/exp/dots_no_symmetry_train_40pct/train_config.py) | [dots_elpis_train_40pct](VQ/exp/dots_elpis_train_40pct/train_config.py) |
| 50% | [dots_no_symmetry_train_50pct](VQ/exp/dots_no_symmetry_train_50pct/train_config.py) | [dots_elpis_train_50pct](VQ/exp/dots_elpis_train_50pct/train_config.py) |
| 60% | [dots_no_symmetry_train_60pct](VQ/exp/dots_no_symmetry_train_60pct/train_config.py) | [dots_elpis_train_60pct](VQ/exp/dots_elpis_train_60pct/train_config.py) |
| 70% | [dots_no_symmetry_train_70pct](VQ/exp/dots_no_symmetry_train_70pct/train_config.py) | [dots_elpis_train_70pct](VQ/exp/dots_elpis_train_70pct/train_config.py) |
| 80% | [dots_no_symmetry_train_80pct](VQ/exp/dots_no_symmetry_train_80pct/train_config.py) | [dots_elpis_train_80pct](VQ/exp/dots_elpis_train_80pct/train_config.py) |
| 90% | [dots_no_symmetry_train_90pct](VQ/exp/dots_no_symmetry_train_90pct/train_config.py) | [dots_elpis_train_90pct](VQ/exp/dots_elpis_train_90pct/train_config.py) |

Train with `python VQ/batch_train.py <experiment>`. Training records evaluation
on the held-out split in each repetition's `plus_eval.txt`. After training, run
`python VQ/statistic_batch.py` to summarize all 18 groups, or pass experiment names
to summarize selected groups. The results are `statistic_<metric>.txt` files with
means, standard deviations and per-repetition values.

## Training schedules

`num_epochs` is the exclusive upper bound of the zero-based training loop.
For example, 50001 preserves epochs 0 through 50000.

| Family | Addition epochs / batch | Subtraction epochs / heads per addition model |
| --- | --- | --- |
| Fixed dots, including symmetry variants | 50001 / 128 | 20001 / 20, except mixed order: 1 head |
| Fig. 2B training-ratio sweep | 50001 / 128 | 20001 / 20 |
| Fixed dots, reconstruction only | 50001 / 128 | 10001 / 20 |
| Eastern and Western tally marks | 50001 / 128 | 20001 / 20 |
| Multi-style, ELPIS / no symmetry / reconstruction only | 10001 / 256 | 1001 / 1 |
| Multi-style, full-data ceiling | 5001 / 256 | 1001 / 1 |
| Online blur, all four models | 4001 / 128 | 801 / 10 |

All additions use learning rate 0.0001; subtraction uses 0.001. Read the configs
for checkpoint and logging intervals. The evaluation pipeline summarizes
subtraction head 1 across the 20 addition repetitions.

## Icon alignment

[icon_alignment](LM_align/exp/icon_alignment/config.py) corresponds to main-text
Fig. 4 and Supplementary Table S2 / Fig. S11. It trains 20 projectors for 30 epochs,
using frozen DINO features and a trained addition model. Choose a best-performing
checkpoint from any repetition of the selected addition experiment and set
`VQSPS.EXP_NAME` and `VQSPS.CHECK_POINT_NAME` in the alignment configuration.
Training, seen-icon evaluation and unseen-icon evaluation images are synthesized online.

`LM_align/statistic_batch.py` selects the best training-accuracy epoch
(excluding epoch 0) and summarizes the 20 repetitions.
