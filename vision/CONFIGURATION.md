# Configuration naming and compatibility

The naming refactor changes identifiers, not experiment values or objectives.
Dead fields and inactive branches remain until the subsequent cleanup.

## Experiment names

The [machine-readable registry](reproduction/experiment_aliases.json) contains all
24 VQ name mappings and the icon-alignment mapping. For example:

```text
2025.05.18_10vq_Zc[2]_Zs[0]_edim1_[0-20]_plus1024_1_tripleSet_Fullsymm
→ dots_elpis

2026.1.17_10codes_iconic
→ icon_alignment
```

The training, subtraction, VQ pipeline, alignment training and alignment statistics
entry points accept historical names as aliases. Outputs always use the canonical
short directory names. Keep historical names quoted when using a shell.

Archived VQ checkpoints and records can be copied under the mapped directory,
keeping their repetition numbers and filenames. Model state-dict parameter keys
are unchanged; checkpoint conversion is not needed. These are repetition IDs,
not a reconstruction of the original random seeds.

## Renamed fields and internal attributes

| Previous name | Current name | Meaning |
| --- | --- | --- |
| `embeddings_num` | `codebook_size` | Entries in the shared codebook |
| `latent_embedding_1` | `num_content_groups` | Number of quantized content groups |
| `latent_code_1` | `content_dim` | Derived content-vector width |
| `latent_code_2` | `style_dim` | Continuous style-vector width |
| `plus_recon_loss_scalar` | `operator_image_weight` | Weight of the decoded-sum image loss |
| `z_plus_loss_scalar` | `operator_latent_weight` | Weight of the sum-code loss |
| `associative_z_loss_scalar` | `symmetry_weight` | Weight of the enabled symmetry MSE terms |
| `VQPlus_eqLoss_scalar` | `operator_vq_weight` | Weight of operator VQ losses |
| `max_iter_num` | `num_epochs` | Exclusive epoch-loop upper bound |
| `num_sub_exp` in addition/alignment | `num_repeats` | Independent training repetitions |
| `num_sub_exp` in subtraction | `num_heads_per_model` | Heads trained for each addition model |

`embedding_dim` keeps its existing name. `content_dim` is computed as
`num_content_groups * embedding_dim`; the old checkpoint module names such as
`encoder`, `decoder`, `vq_layer` and `plus_net` are preserved.

The old `max_iter_num=50001` becomes `num_epochs=50001`, preserving the original
0–50000 loop. This is not silently shortened to 50000 epochs. The existing LM
nested `ALIGN.EPOCHS` setting also retains its value and spelling.

`experiment_registry.normalize_config(config, kind)` translates archived keys,
with `kind` equal to `train`, `other` or `align`. The shared configuration loader
applies this conversion. Archived scripts which directly construct classes should
normalize their configs first. Supplying both spellings of a field is rejected.
The key mapping is also available as [JSON](reproduction/config_key_aliases.json).

## Output compatibility

Historical text-record columns and JSON metric keys remain unchanged. In particular,
`substraction_one` is the existing serialized subtraction key; the spelling of
internal Python variables has been corrected without breaking archived results.
A `cycle` suffix still denotes decode–encode evaluation, and is not an alias for
an unsuffixed metric.

Public scientific settings remain explicit even when shared by all retained runs.
For example, codebook size, embedding dimension and learning rate were renamed or
retained, not removed as redundant constants. `plus_by_embedding` is still present
in this naming-only change; its removal with the explicit reconstruction-only
training path belongs to the next cleanup stage.
