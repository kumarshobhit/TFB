# Frequency-Aligned Positional Encoding Project Guide

This repository is based on the upstream TFB benchmark codebase. The project for the master's report adds:

- a custom TST implementation with RoPE and SinePE support,
- shell runners for the selected RoPE and SinePE experiments,
- summary scripts for multi-seed comparisons,
- plotting scripts for the report figures,
- statistical tests used in the appendix.

This guide documents the project-specific layer so the code submission is understandable without reverse-engineering the entire repo.

## 1. Core Project Files

### Custom model / calibration logic

- `ts_benchmark/baselines/tst/tst.py`
  - custom simplified TST backbone used in the report
  - supports `pos_encoding="rotary"` and `pos_encoding="sinespe"`
- `rope_base.py`
  - computes calibrated RoPE bases from dominant dataset frequency and ladder index `k`

### Main experiment runners

- `scripts/multivariate_forecast/run_rope_seed_selected.sh`
  - main multi-seed RoPE experiments used for Table 6.1
- `scripts/multivariate_forecast/run_sine_seed_selected.sh`
  - main multi-seed SinePE experiments used for Table 6.2
- `scripts/multivariate_forecast/run_rope_missing_k_seed2021.sh`
  - extra seed-2021 RoPE runs used to fill in k-screening coverage for Appendix A.1

### Aggregation / analysis scripts

- `rope_seed_summary.py`
  - aggregates seeded RoPE runs into `result/rope_seed_analysis`
- `sine_seed_summary.py`
  - aggregates seeded SinePE runs into `result/sine_seed_analysis`
- `k_seed2021_screen_summary.py`
  - summarizes seed-2021 k screening coverage
- `sine_k_summary.py`
  - summarizes reduced SinePE k experiments
- `scripts/stat_tests/run_stat_tests.py`
  - produces the appendix statistical-test outputs

### Figure scripts

- `plot_dataset_frequency_spectra.py`
  - Figure 5.1 style FFT spectra
- `plot_relative_wape_improvement.py`
  - relative-improvement bar plots for RoPE and SinePE
- `plot_wape_eye_test.py`
  - qualitative prediction-window comparisons
- `plot_sine_frequency_demo.py`
  - auxiliary illustration for SinePE frequency behavior

## 2. Environment and Data

The project inherits the TFB environment assumptions:

- Python `3.8` is the safest target
- install dependencies with `requirements-docker.txt` for reproducibility
- place datasets under `dataset/forecasting/`

Expected dataset files used in the report:

- `dataset/forecasting/ETTh1.csv`
- `dataset/forecasting/ETTm1.csv`
- `dataset/forecasting/Weather.csv`
- `dataset/forecasting/Solar_ci64_every2.csv`

## 3. Fastest Reproduction Path

If the `result/` folders from the project runs are already present, the easiest way to regenerate the report artifacts is:

```bash
bash scripts/project_reproduce_artifacts.sh
```

This does **not** retrain the models. It regenerates the summary CSVs, plots, and statistics used by the report from the saved run outputs already stored in `result/`.

## 4. Full Experiment Commands

### 4.1 RoPE seeded experiments

Run the selected calibrated/base-control experiments for one dataset and one seed:

```bash
bash scripts/multivariate_forecast/run_rope_seed_selected.sh ETTh1 2021 0
bash scripts/multivariate_forecast/run_rope_seed_selected.sh ETTh1 2022 0
bash scripts/multivariate_forecast/run_rope_seed_selected.sh ETTh1 2023 0
```

Replace `ETTh1` with one of:

- `ETTh1`
- `ETTm1`
- `Weather`
- `Solar`

The third argument is the GPU id.

### 4.2 RoPE k-screening completion runs

These runs were used to complete the seed-2021 screening coverage for Appendix A.1:

```bash
bash scripts/multivariate_forecast/run_rope_missing_k_seed2021.sh ETTm1 0
bash scripts/multivariate_forecast/run_rope_missing_k_seed2021.sh Weather 0
bash scripts/multivariate_forecast/run_rope_missing_k_seed2021.sh Solar 0
```

### 4.3 SinePE seeded experiments

Run the selected calibrated/base-control SinePE experiments:

```bash
bash scripts/multivariate_forecast/run_sine_seed_selected.sh ETTh1 2021 0
bash scripts/multivariate_forecast/run_sine_seed_selected.sh ETTh1 2022 0
bash scripts/multivariate_forecast/run_sine_seed_selected.sh ETTh1 2023 0
```

Again, replace `ETTh1` with `ETTm1`, `Weather`, or `Solar` as needed.

## 5. Regenerating Analysis Tables

### 5.1 RoPE seed summary

```bash
python rope_seed_summary.py \
  --seed-folder ETTh1:2021=result/ETTh1_22ndfeb/TST_rotary \
  --seed-folder ETTh1:2022=result/ETTh1_seed2022/TST_rotary \
  --seed-folder ETTh1:2023=result/ETTh1_seed2023/TST_rotary \
  --seed-folder ETTm1:2021=result/ETTm1_seed2021/TST_rotary \
  --seed-folder ETTm1:2022=result/ETTm1_seed2022/TST_rotary \
  --seed-folder ETTm1:2023=result/ETTm1_seed2023/TST_rotary \
  --seed-folder Weather:2021=result/Weather_seed2021/TST_rotary \
  --seed-folder Weather:2022=result/Weather_seed2022/TST_rotary \
  --seed-folder Weather:2023=result/Weather_seed2023/TST_rotary \
  --seed-folder Solar:2021=result/Solar_seed2021/TST_rotary_ci64_every2 \
  --seed-folder Solar:2022=result/Solar_seed2022/TST_rotary_ci64_every2 \
  --seed-folder Solar:2023=result/Solar_seed2023/TST_rotary_ci64_every2 \
  --best-base-table result/rope_k_analysis/rope_k_best_vs_default.csv \
  --extra-base random500=500 \
  --extra-base random50000=50000 \
  --output-dir result/rope_seed_analysis
```

### 5.2 SinePE seed summary

```bash
python sine_seed_summary.py \
  --seed-folder ETTh1:2021=result/ETTh1_10thmar/TST_sine \
  --seed-folder ETTh1:2022=result/ETTh1_sine_seed2022/TST_sine \
  --seed-folder ETTh1:2023=result/ETTh1_sine_seed2023/TST_sine \
  --seed-folder ETTm1:2021=result/ETTm1_17thmar/TST_sine \
  --seed-folder ETTm1:2022=result/ETTm1_sine_seed2022/TST_sine \
  --seed-folder ETTm1:2023=result/ETTm1_sine_seed2023/TST_sine \
  --seed-folder Weather:2021=result/Weather_17thmar/TST_sine \
  --seed-folder Weather:2022=result/Weather_sine_seed2022/TST_sine \
  --seed-folder Weather:2023=result/Weather_sine_seed2023/TST_sine \
  --seed-folder Solar:2021=result/Solar_17thmar/TST_sine_ci64_every2 \
  --seed-folder Solar:2022=result/Solar_sine_seed2022/TST_sine_ci64_every2 \
  --seed-folder Solar:2023=result/Solar_sine_seed2023/TST_sine_ci64_every2 \
  --best-base ETTh1=14.58091716442549 \
  --best-base ETTm1=1436.1719287648493 \
  --best-base Weather=16.233809339931238 \
  --best-base Solar=17237.391423477242 \
  --extra-base random500=500 \
  --extra-base random5000=5000 \
  --extra-base random50000=50000 \
  --d-model 16 \
  --output-dir result/sine_seed_analysis
```

### 5.3 Seed-2021 RoPE k screening summary

```bash
python k_seed2021_screen_summary.py \
  --encoding rope \
  --dataset ETTh1=result/ETTh1_22ndfeb/TST_rotary \
  --dataset ETTm1=result/ETTm1_22ndfeb/TST_rotary \
  --dataset Weather=result/Weather_22ndfeb/TST_rotary \
  --dataset Solar=result/Solar_22ndfeb/TST_rotary_ci64_every2 \
  --freq ETTh1=0.04168 \
  --freq ETTm1=0.01042 \
  --freq Weather=0.01389 \
  --freq Solar=0.01389 \
  --output-dir result/rope_seed2021_screen
```

## 6. Regenerating Figures

### FFT spectra

```bash
python plot_dataset_frequency_spectra.py \
  --output figures/dataset_frequency_spectra_updated.png \
  --summary figures/dataset_frequency_spectra_updated_summary.csv
```

### Relative WAPE improvements

```bash
python plot_relative_wape_improvement.py
```

### Qualitative eye-test figures

RoPE batch figures:

```bash
python plot_wape_eye_test.py --batch-mode --batch-output-dir result/eye_test_rotary_batch
```

SinePE ETTh1 figure:

```bash
python plot_wape_eye_test.py \
  --default-archive result/ETTh1_sine_seed2022/TST_sine/TST_sinespe_default.tar.gz \
  --calibrated-archive result/ETTh1_sine_seed2022/TST_sine/TST_sinespe_14.5809.tar.gz \
  --default-feature-metrics result/ETTh1_sine_seed2022/TST_sine/TST_per_feature_metrics_sinespe_default.csv \
  --calibrated-feature-metrics result/ETTh1_sine_seed2022/TST_sine/TST_per_feature_metrics_sinespe_14.5809.csv \
  --output-dir figures \
  --output-prefix etth1_sine_eye_test
```

## 7. Statistical Tests

```bash
python scripts/stat_tests/run_stat_tests.py
```

This writes its outputs to `result/stat_tests/`.

## 8. Report Artifact Map

- Table 6.1 (RoPE seeded comparison)
  - inputs: `result/*/TST_rotary/test_report_rope_*.csv`
  - script: `rope_seed_summary.py`
  - output: `result/rope_seed_analysis/summary.csv`

- Table 6.2 (SinePE seeded comparison)
  - inputs: `result/*/TST_sine*/test_report_sinespe_*.csv`
  - script: `sine_seed_summary.py`
  - output: `result/sine_seed_analysis/summary.csv`

- Figure 5.1 (dataset FFT spectra)
  - script: `plot_dataset_frequency_spectra.py`
  - output: `figures/dataset_frequency_spectra_updated.png`

- Figure 6.1 / 6.2 (relative WAPE improvements)
  - script: `plot_relative_wape_improvement.py`
  - outputs:
    - `figures/wape_relative_improvement_rope.png`
    - `figures/wape_relative_improvement_sine.png`

- Figure 6.3 / 6.4 and appendix qualitative plots
  - script: `plot_wape_eye_test.py`
  - outputs under `figures/` and `result/eye_test_rotary_batch/`

- Appendix A.2 statistical test summary
  - script: `scripts/stat_tests/run_stat_tests.py`
  - output: `result/stat_tests/`

## 9. Submission Advice

For code submission, the easiest way to present the repo cleanly is:

1. Keep the upstream TFB README as-is.
2. Point the examiner to this file for the project-specific additions.
3. Mention that the custom implementation lives mainly in:
   - `ts_benchmark/baselines/tst/tst.py`
   - `rope_base.py`
   - the root analysis / plotting scripts
   - the three project runner scripts under `scripts/multivariate_forecast/`

If needed later, this project-specific layer can be moved into its own cleaned repository, but this file should already make the current workspace understandable enough for review.
