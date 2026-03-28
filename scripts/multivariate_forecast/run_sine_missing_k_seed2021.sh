#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset required}"
GPU="${2:-0}"

run_cmd() {
  local data_name="$1"
  local save_path="$2"
  local hyper_json="$3"
  CUDA_VISIBLE_DEVICES="${GPU}" python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${data_name}" \
    --strategy-args '{"horizon": 96}' \
    --model-name "tst.TST" \
    --model-hyper-params "${hyper_json}" \
    --deterministic "full" \
    --gpus "${GPU}" \
    --num-workers 1 \
    --timeout 60000 \
    --save-path "${save_path}" \
    --save-true-pred True
}

case "${DATASET}" in
  ETTh1)
    SAVE_PATH="ETTh1_10thmar/TST_sine"
    DATA_NAME="ETTh1.csv"
    COMMON='{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 35.62157547700813/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 8.531548699810296/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 5.968381311294388/')"
    ;;
  ETTm1)
    SAVE_PATH="ETTm1_17thmar/TST_sine"
    DATA_NAME="ETTm1.csv"
    COMMON='{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 1436.1719287648493/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 78.40140765639345/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 37.89685908838421/')"
    ;;
  Weather)
    SAVE_PATH="Weather_17thmar/TST_sine"
    DATA_NAME="Weather.csv"
    COMMON='{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 667.2896094911222/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 49.49811011201147/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 25.83194939394087/')"
    ;;
  Solar)
    SAVE_PATH="Solar_17thmar/TST_sine_ci64_every2"
    DATA_NAME="Solar_ci64_every2.csv"
    COMMON='{"pos_encoding": "sinespe", "d_model": 16, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 667.2896094911222/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 49.49811011201147/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${COMMON}" | sed 's/"pos_encoding": "sinespe"/"pos_encoding": "sinespe", "base_freq": 25.83194939394087/')"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac
