# Numerical datasets

Use the released **`vision_data.zip`** as the default reproduction input. It contains
the original fixed images and split membership retained from S3Plus, without
regenerating or changing any training data. The archive is accompanied by
`vision_data.zip.sha256`.

The validated ZIP release is **38,104,876 bytes (38.10 MB)**. Its size includes ZIP
headers for the many small files and the release manifests; the image payload alone
is smaller, as listed below.

The fixed corpus contains **14 directories, 46,155 PNG files and 27,382,562 bytes
of image-file contents** (27.38 MB / 26.11 MiB, before ZIP packaging). Images are
64×64 RGBA PNGs; the training loaders convert them to RGB.

## Download / unpack / verify

Place the ZIP and its checksum file alongside `vision/`, then from `vision/`:

```bash
(cd .. && sha256sum -c vision_data.zip.sha256)

# Only needed when dataset/ is absent. Do not overwrite an existing data tree.
test ! -e dataset && unzip -q ../vision_data.zip 'dataset/*' -d .

python reproduction/data_bundle.py verify
```

If `dataset/` is already present, skip extraction and run verification. Verification
requires the exact file set and checks every PNG against the released SHA-256 list;
missing, extra or changed files cause a nonzero exit. It does not repair or overwrite
files. Use the code and metadata from the same release as the archive.

The archive contains `dataset/...`, `reproduction/datasets.txt`, and the metadata
under `reproduction/data/`. It excludes model checkpoints, logs and generated
training outputs. The bundled icon font and VQ weights are listed in
[lm_align_assets.txt](reproduction/lm_align_assets.txt). DINO weights are downloaded
on first use and cached by PyTorch; see [REPRODUCING.md](REPRODUCING.md).

## Contents and split sizes

A *triplet* is a directory containing `a-<m>.png`, `b-<n>.png` and `c-<result>.png`.
Counts below include visual styles; prototype rows count individual images.

| Dataset directory under `dataset/` | Training triplets | Held-out triplets / prototypes | PNG files |
| --- | ---: | --- | ---: |
| `(0,20)-FixedPos-oneStyle` | — | 21 blue-dot prototypes | 21 |
| `(0,20)-FixedPos-oneStyle_EU_tally` | — | 21 Western tally prototypes | 21 |
| `(0,20)-FixedPos-oneStyle_ZHENG` | — | 21 Eastern tally prototypes | 21 |
| `multi_style_eval_(0,20)_FixedPos_TrainStyle` | — | 21 numbers × 16 styles | 336 |
| `single_style_pairs(0,20)_tripleSet` | 68 | test_1: 101; test_2: 62 | 693 |
| `single_style_pairs(0,20)_tripleSet_EU_tally` | 68 | test_1: 101; test_2: 62 | 693 |
| `single_style_pairs(0,20)_tripleSet_ZHENG` | 68 | test_1: 101; test_2: 62 | 693 |
| `single_style_pairs(0,20)_tripleSet_trainAll` | 231 | test_1: 101; test_2: 62 | 1,182 |
| `multi_style_(4,4)_realPairs_plus(0,20)` | 1,312 | test: 2,384 | 11,088 |
| `multi_style_(4,4)_realPairs_plus(0,20)_trainAll` | 3,696 | test: 2,384 | 18,240 |
| `single_style_pairs_minus(0,20)` | 152 | test: 79 | 693 |
| `single_style_pairs_minus(0,20)_EU_tally` | 152 | test: 79 | 693 |
| `single_style_pairs_minus(0,20)_ZHENG` | 152 | test: 79 | 693 |
| `multi_style_pairs_minus(0,20)` | 2,432 | test: 1,264 | 11,088 |

The multi-style addition data contain 82 train / 149 held-out ordered numerical
pairs, each with 16 visual styles. The counts above report the actual packaged data;
they are not inferred from a nominal train ratio. Full-data ceiling directories
intentionally retain evaluation pairs that also occur in their full training set.

Numbers range from 0 to 20. Addition labels satisfy `m+n=result<=20`; subtraction
labels satisfy `m-n=result>=0`. Blue-dot zero is a hollow marker. The four multi-style
markers are square, circle, downward triangle and thin diamond; the four colours
are purple, salmon, olive and blue. Filenames retain historical spellings such as
`o`/`circle` and `default` rather than renaming dataset paths or sample identifiers.

Experiment-to-dataset and paper mappings are in [EXPERIMENTS.md](EXPERIMENTS.md).

## What fixes the dataset identity?

The original split random seeds are not known. Re-running an old random splitter
with a newly chosen seed cannot recover the experimental split. Instead, this
release records the actual sample membership:

- [splits.json](reproduction/data/splits.json): all prototype filenames, split names,
  triplet directory names and constituent image filenames.
- [files.sha256](reproduction/data/files.sha256): every original PNG's byte checksum,
  with paths relative to `vision/`.
- [inventory.json](reproduction/data/inventory.json): per-dataset counts, sizes and formats.
- [pixel_hashes.json](reproduction/data/pixel_hashes.json): decoded RGB hashes and
  dimensions indexed by the original PNG checksum, matching loader input semantics.

Together these distinguish a changed split, a changed image and a change limited
to PNG encoding/metadata. These are release reference files, not manifests to
regenerate automatically when verification fails.

## Optional regeneration of fixed images

A focused wrapper uses the existing drawing primitives in
[dataMaker_commonFunc.py](dataMaker_commonFunc.py), with dot/tally coordinates,
marker sizes, line widths and colour settings. It renders reusable prototypes and
places copies at the exact archived filenames and split positions. It does not
resample train/test pairs.

The release rendering profile explicitly preserves visible dot-image axes/borders
and the short horizontal zero mark in both tally families. These differ from the
later defaults in the broad historical drawing scripts. The scatter helper's new
`show_axes` argument defaults to its previous behavior; only the release renderer
selects the historical border. The original dataset itself is untouched.

```bash
# The output directory must not exist; dataset/ will not be changed.
python reproduction/data_bundle.py regenerate \
  --output /absolute/path/to/a/new/generated_dataset

# After generation completes, compare training-visible pixels.
python reproduction/data_bundle.py verify \
  --data-root /absolute/path/to/a/new/generated_dataset --pixels

# Optional stronger check of the encoded PNG bytes, including metadata.
python reproduction/data_bundle.py verify \
  --data-root /absolute/path/to/a/new/generated_dataset
```

The original broad `dataMaker_*` scripts contain historical main blocks for other
experiments; use the documented wrapper instead of running those main blocks.
Regeneration is supplemental: the original archive remains the authoritative
input even if a renderer/library change prevents exact pixel reproduction.

### Regeneration validation

With Python 3.10.14, Matplotlib 3.9.2 and Pillow 10.2.0, all **46,155 regenerated
images match the original RGB pixels and dimensions exactly**, with identical file
names and split membership. The encoded PNG bytes differ, so regenerated files do
not pass the original byte-level checksum list; use `--pixels` for redraw validation.
The ZIP contains the untouched original PNG bytes. See the machine-readable
[generation validation](reproduction/data/generation_validation.json) and
[generator/font source hashes](reproduction/data/generator_sources.json).

## Online blur

Blur experiments use the same packaged blue-dot images, with independent online
Gaussian blur from `VQ/common_func.py`. Configs specify kernel sizes `(5,7,9)`, sigma
`(0.5,3.0)`, no-blur probability 0 and 16 visual augmentations. There is no separate
blurred-image download. This packaging work does not change the training or
sampling seeds in the historical configurations.

## Online icons and optional frozen samples

Icon training uses the existing `gen_recurrent_data` implementation in
[LM_align/synthData/SynthCommon_win.py](LM_align/synthData/SynthCommon_win.py), invoked
by `onlineGenDataset`. Keep this source, its dataset wrapper and the bundled
`LM_align/synthData/fonts/seguiemj.ttf`. Font substitution can change the input pixels.

For inspection or future independent evaluation, export a new reproducible sample:

```bash
python reproduction/generate_icon_data.py \
  --split seen --seed 20261008 --samples 256 \
  --output /absolute/path/to/a/new/icons_seen

python reproduction/generate_icon_data.py \
  --split unseen --seed 20261008 --samples 256 \
  --output /absolute/path/to/a/new/icons_unseen
```

Each export contains numbered directories with `image_a.png`, `image_b.png`,
`image_c.png` and `label.json`, plus a seed/version/font/source manifest and file
checksums. Reusing a seed under the same software and font versions reproduces the
export. Seen and unseen exports use their respective object lists; using the same
seed is not a claim that their recursive layouts are paired.

These are newly generated frozen samples, not recovered historical evaluation
images. Current alignment training continues to generate online data; the existing
statistics entry point reads training logs and does not automatically consume these
exports. No historical training/evaluation protocol is changed by packaging them.

## Rebuilding the release ZIP

Only package a dataset that passes the original byte-level manifest:

```bash
python reproduction/data_bundle.py pack \
  --output /absolute/path/to/a/new/vision_data.zip
```

The packer fixes ZIP timestamps, ordering and permissions, verifies every data file,
and creates `vision_data.zip.sha256` next to the archive. It refuses to overwrite
an existing archive or checksum. Reproducibility of compressed bytes additionally
depends on the Python/zlib build; always distribute the checksum of the actual ZIP.
