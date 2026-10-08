# Validation scope

The following checks have been completed using a CUDA GPU:

- One optimizer update for representatives covering the visual training datasets,
  including blurred inputs; temporary outputs were removed.
- Subtraction-head training and evaluation startup using a completed addition
  checkpoint.
- Full evaluation of `dots_full_data`: all 360 values across 18 metrics and 20
  repetitions matched the historical results.
- Configuration and experiment-name compatibility checks. Under deterministic
  validation settings, 17 representative configurations produced identical initial
  weights, losses and parameters after one optimizer update before and after the
  naming changes.
- Icon-alignment model loading and forward computation, plus statistics aggregation
  using controlled log fixtures.
- DINO automatic download and cache reuse. The downloaded file's full SHA-256,
  loaded model weights and a forward pass match the original bundled checkpoint;
  explicit local checkpoint paths remain supported.
- Original image checksums, regenerated image pixels and archive extraction.
  See [data verification results](data/generation_validation.json).

These checks establish execution and regression coverage. They do not establish
that all paper results have been reproduced by full retraining. A fresh environment
installation and full icon-alignment training have not been validated. See the
[reproduction guide](../REPRODUCING.md) and [experiment coverage](../EXPERIMENTS.md)
for the remaining limitations.
