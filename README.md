# Investigating the Impact of Frequency-Aligned Positional Encodings

This repository is the submission codebase for the master's project and report on frequency-aligned positional encodings for time series forecasting.

The project studies whether aligning the internal frequency ladder of positional encodings with dataset spectral structure improves forecasting performance. The main contribution is a dataset-aware calibration of RoPE, together with a parallel calibration of sinusoidal positional encoding (SinePE), evaluated on ETTh1, ETTm1, Solar, and Weather.

## Start Here

If you opened this repository from the report, these are the most useful entry points:

- [PROJECT_REPRODUCTION.md](./PROJECT_REPRODUCTION.md): project-specific code map, reproduction commands, and report artifact traceability
- [ts_benchmark/baselines/tst/tst.py](./ts_benchmark/baselines/tst/tst.py): custom TST backbone with RoPE and SinePE support
- [scripts/project_reproduce_artifacts.sh](./scripts/project_reproduce_artifacts.sh): rebuild the report-facing summaries, plots, and statistics from saved `result/` folders

## What Is Original In This Repository

The project is built on top of the upstream [TFB benchmark repository](https://github.com/decisionintelligence/TFB). The submission-specific work in this branch includes:

- a custom simplified TST implementation used in the report
- frequency-calibrated RoPE and SinePE support
- shell runners for the seeded RoPE and SinePE experiments
- aggregation scripts for multi-seed analysis
- plotting scripts for the report figures
- appendix statistical testing utilities

## Key Files

- `ts_benchmark/baselines/tst/tst.py`
  Custom TST model used in the experiments.
- `rope_base.py`
  Frequency-to-base calibration helper for RoPE.
- `scripts/multivariate_forecast/run_rope_seed_selected.sh`
  Main RoPE experiment runner used for the seeded report comparisons.
- `scripts/multivariate_forecast/run_sine_seed_selected.sh`
  Main SinePE experiment runner used for the seeded report comparisons.
- `rope_seed_summary.py`
  Aggregates RoPE seeded runs into the report tables.
- `sine_seed_summary.py`
  Aggregates SinePE seeded runs into the report tables.
- `plot_dataset_frequency_spectra.py`
  Generates the FFT spectrum figure used in the report.
- `plot_relative_wape_improvement.py`
  Generates the relative WAPE comparison plots.
- `plot_wape_eye_test.py`
  Generates the qualitative prediction-window plots.

## Quick Reproduction

### Rebuild the report artifacts from saved results

If the `result/` folders from the finished experiments are already present, the fastest path is:

```bash
bash scripts/project_reproduce_artifacts.sh
```

This regenerates the summary CSVs, figures, and statistical outputs used in the report without retraining models.

### Rerun the main experiments

RoPE:

```bash
bash scripts/multivariate_forecast/run_rope_seed_selected.sh ETTh1 2021 0
```

SinePE:

```bash
bash scripts/multivariate_forecast/run_sine_seed_selected.sh ETTh1 2021 0
```

The arguments are `dataset`, `seed`, and `gpu_id`. Supported datasets are `ETTh1`, `ETTm1`, `Weather`, and `Solar`.

For the full command map, expected result folders, and figure/table traceability, see [PROJECT_REPRODUCTION.md](./PROJECT_REPRODUCTION.md).

## Environment And Compute

For reproducibility, this project follows the upstream TFB recommendation of using pinned dependencies from `requirements-docker.txt`, with Python `3.8` as the safest target environment.

Experiments for this project were run on the Freiburg server with:

- `8 x NVIDIA GeForce RTX 2080 Ti`
- `11264 MiB` memory per GPU
- NVIDIA driver `535.288.01`

The shell runners use one GPU per run and take the GPU id explicitly as the last argument.

The current shell on this machine reports:

- `Python 3.13.11`

## Data

The report experiments use the following forecasting datasets under `dataset/forecasting/`:

- `ETTh1.csv`
- `ETTm1.csv`
- `Weather.csv`
- `Solar_ci64_every2.csv`

## Repository Note

This branch is intentionally submission-oriented. The root README is project-first so that the repository link in the report lands directly on the custom code and reproduction instructions rather than the full upstream TFB landing page.
