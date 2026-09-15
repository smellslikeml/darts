# xPatch Model — Validation Plan

This document describes what has been validated for the `XPatchModel` contribution
(`darts/models/forecasting/xpatch_model.py`) and what is deferred to a later,
GPU-based validation pass.

## Validated here (CI-runnable, CPU-only)

`darts/tests/models/forecasting/test_xpatch.py` covers, on CPU with tiny synthetic
data and no downloads:

- **Construction / input validation** — invalid `ma_type`, `padding_patch`,
  `patch_len > input_chunk_length`, `output_chunk_length < 2` (the trend-stream
  pooling head requires at least 2 output steps), and smoothing factors outside
  `(0, 1)` all raise `ValueError` at model creation (darts' "make it hard for
  users to make mistakes" principle).
- **Fit / predict smoke test** — 1–2 epoch fits on small synthetic series for all
  three decomposition modes (`ma_type="ema"`, `"dema"`, `"reg"`) and with
  `use_reversible_instance_norm=True`; forecast output shape matches
  `(n, n_components)`; predictions are finite; short (`n=1`) and autoregressive
  (`n > output_chunk_length`) predictions work.
- **Constant-series learning check** (DLinear/NLinear test idiom) — a model trained
  on a constant-1000 series is outperformed, on the constant-10 series, by a model
  trained on the constant-10 series.
- **Multivariate + past covariates** — each component is forecast by its own
  channel-independent stream; past covariates are accepted as additional input
  channels.
- **Probabilistic forecasting** — with `GaussianLikelihood`, sampled predictions
  (`num_samples=20`) and direct likelihood-parameter predictions
  (`predict_likelihood_parameters=True`) return the expected shapes.
- **Backward / loss-decrease check** — a manual 20-step Adam loop on the fitted
  `_XPatchModule` on CPU verifies that gradients flow (finite gradients on the
  final head of both streams) and that the training loss decreases. This check is
  only possible because the ported EMA/DEMA layers are **device-agnostic** — the
  reference implementation hardcodes `.to("cuda")` / `.to(device="cuda")` in
  `layers/ema.py` and `layers/dema.py`, which would crash on CPU.
- **Shape-flow verification** — the patching/conv/pooling arithmetic (patch count
  with end padding, depthwise/pointwise conv dimensions, `AvgPool1d` flooring,
  `LayerNorm` sizes, final reshape to `(batch, pred_len, channels, nr_params)`)
  was additionally verified exhaustively (reference defaults, the reference's
  ILI configuration, non-divisible strides, odd horizons) against a shape
  simulation during porting.

## Deferred (human, GPU — NOT this run)

Full multi-dataset accuracy parity against the paper's tables. The parity oracle
is the reference repository's own scripts and reported results:

- **Oracle**: `stitsyuk/xPatch` — `scripts/xPatch_unified.sh`,
  `scripts/xPatch_fair.sh`, `scripts/xPatch_search.sh`, run via the reference's
  `run.py`; paper tables (Tables 1–4 in [arXiv:2412.17323](https://arxiv.org/abs/2412.17323)):
  ETT (ETTh1/h2, ETTm1/m2), weather, electricity, traffic, exchange, solar, and
  national-illness, at horizons 96/192/336/720 (24/36/48/60 for ILI).
- **Reference hyperparameters to mirror** (from `run.py` defaults /
  `scripts/xPatch_unified.sh`): `seq_len=96` (`36` for ILI), `patch_len=16`
  (`6` for ILI), `stride=8` (`3` for ILI), `padding_patch="end"`,
  `ma_type="ema"`, `alpha=0.3`, `beta=0.3`, RevIN enabled — in Darts these map to
  `input_chunk_length`, `patch_len`, `stride`, `padding_patch`, `ma_type`,
  `alpha`, `beta`, and `use_reversible_instance_norm=True` respectively.
- **Training-side differences (intentional)**: the reference trains with an
  arctangent loss and a sigmoid learning-rate schedule defined in its `exp/` and
  `utils/` scripts — not part of the ported `nn.Module`. Darts users supply
  `loss_fn` / `lr_scheduler_cls` / `lr_scheduler_kwargs` themselves; for parity
  runs, an arctan loss can be passed as a custom `loss_fn` (a small
  `torch.nn.Module`), and the sigmoid LR schedule via `lr_scheduler_cls` with a
  custom `torch.optim.lr_scheduler.LambdaLR`. Neither is wired as a default.
- **Suggested procedure**: on a GPU machine (or Colab), fit `XPatchModel` with
  the paper's per-dataset hyperparameters on the standard splits and compare
  MSE/MAE against the paper's tables and against the reference implementation's
  own outputs for identical seeds/horizons. Tolerances should account for the
  intentional differences above (loss function, scheduler, batching, and Darts'
  sample-based training windows vs. the reference's fixed split dataloader).

This plan is intentionally **not** a blocker for this contribution: the model
architecture, darts integration, and CPU correctness checks above are complete.
