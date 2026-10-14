# Configuration reference

Each VQ experiment has a `train_config.py` for addition and an
`other_tasks_config.py` for subtraction under `VQ/exp/<experiment>/`.
[EXPERIMENTS.md](EXPERIMENTS.md) lists the 42 VQ groups and their paper correspondence.
Icon alignment is configured in
[`LM_align/exp/icon_alignment/config.py`](LM_align/exp/icon_alignment/config.py).

## Model and loss parameters

| Parameter | Meaning |
| --- | --- |
| `codebook_size` | Entries in the shared codebook |
| `num_content_groups` | Number of quantized content groups |
| `embedding_dim` | Dimension of each codebook embedding |
| `style_dim` | Continuous style-vector width |
| `operator_image_weight` | Weight of the decoded-sum image loss |
| `operator_latent_weight` | Weight of the sum-code loss |
| `symmetry_weight` | Weight of the enabled symmetry MSE terms |
| `operator_vq_weight` | Weight of operator VQ losses |

The model computes `content_dim` as `num_content_groups * embedding_dim`.
Codebook size, embedding dimension, loss weights and learning rate are specified
in each experiment's configuration.

## Training parameters

| Parameter | Meaning |
| --- | --- |
| `num_epochs` | Exclusive upper bound of the zero-based training loop |
| `num_repeats` | Independent training repetitions |
| `num_heads_per_model` | Subtraction heads trained for each addition model |
| `is_random_split_data` | Enable a random train/test split of the supplied dataset |
| `train_data_ratio` | Training fraction for the random split |
| `random_split_seed` | Optional split seed; the batch launcher adds the repetition number |

For example, `num_epochs=50001` runs epochs 0 through 50000. Icon alignment uses
`ALIGN.EPOCHS` for its training duration.

## Checkpoints and records

VQ checkpoints and records are stored under `VQ/exp/<experiment>/<repetition>/`.
See [REPRODUCING.md](REPRODUCING.md) for checkpoint selection, expected filenames
and evaluation commands.

The result key `substraction_one` contains subtraction head 1's results across
addition repetitions. A `cycle` suffix denotes decode–encode evaluation.
