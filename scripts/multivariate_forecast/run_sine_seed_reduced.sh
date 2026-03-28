#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset required}"
SEED="${2:?seed required}"
GPU="${3:-0}"

run_cmd() {
  local data_name="$1"
  local save_path="$2"
  local hyper_json="$3"
  CUDA_VISIBLE_DEVICES="${GPU}" python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${data_name}" \
    --strategy-args '{"horizon": 96}' \
    --seed "${SEED}" \
    --model-name "tst.TST" \
    --model-hyper-params "${hyper_json}" \
    --deterministic "full" \
    --gpus 0 \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "${save_path}" \
    --save-true-pred True
}

case "${DATASET}" in
  ETTh1)
    SAVE_PATH="ETTh1_sine_seed${SEED}/TST_sine"
    DATA_NAME="ETTh1.csv"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 14.58091716442549, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 500.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 50000.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    ;;
  ETTm1)
    SAVE_PATH="ETTm1_sine_seed${SEED}/TST_sine"
    DATA_NAME="ETTm1.csv"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 233.29467463080783, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 500.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 50000.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    ;;
  Weather)
    SAVE_PATH="Weather_sine_seed${SEED}/TST_sine"
    DATA_NAME="Weather.csv"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 16.233809339931238, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 500.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 50000.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    ;;
  Solar)
    SAVE_PATH="Solar_sine_seed${SEED}/TST_sine_ci64_every2"
    DATA_NAME="Solar_ci64_every2.csv"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 17237.391423477242, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 500.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" '{"pos_encoding": "sinespe", "base_freq": 50000.0, "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac
