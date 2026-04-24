#!/usr/bin/env bash
set -euo pipefail

# Regenerate the report-facing analysis outputs from the existing saved result
# folders. This script does not retrain models; it rebuilds summaries, figures,
# and statistical tables from the run outputs already stored under result/.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

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

python plot_dataset_frequency_spectra.py \
  --output figures/dataset_frequency_spectra_updated.png \
  --summary figures/dataset_frequency_spectra_updated_summary.csv

python plot_relative_wape_improvement.py

python plot_wape_eye_test.py --batch-mode --batch-output-dir result/eye_test_rotary_batch

python plot_wape_eye_test.py \
  --default-archive result/ETTh1_sine_seed2022/TST_sine/TST_sinespe_default.tar.gz \
  --calibrated-archive result/ETTh1_sine_seed2022/TST_sine/TST_sinespe_14.5809.tar.gz \
  --default-feature-metrics result/ETTh1_sine_seed2022/TST_sine/TST_per_feature_metrics_sinespe_default.csv \
  --calibrated-feature-metrics result/ETTh1_sine_seed2022/TST_sine/TST_per_feature_metrics_sinespe_14.5809.csv \
  --output-dir figures \
  --output-prefix etth1_sine_eye_test

python scripts/stat_tests/run_stat_tests.py

echo "Report-facing artifacts regenerated."
