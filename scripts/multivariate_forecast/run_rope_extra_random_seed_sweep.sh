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
    DATA_NAME="ETTh1.csv"
    if [[ "${SEED}" == "2021" ]]; then
      SAVE_PATH="ETTh1_22ndfeb/TST_rotary"
    else
      SAVE_PATH="ETTh1_seed${SEED}/TST_rotary"
    fi
    HYPER_BASE='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  ETTm1)
    DATA_NAME="ETTm1.csv"
    SAVE_PATH="ETTm1_seed${SEED}/TST_rotary"
    HYPER_BASE='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 512, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  Weather)
    DATA_NAME="Weather.csv"
    SAVE_PATH="Weather_seed${SEED}/TST_rotary"
    HYPER_BASE='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 256, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  Solar)
    DATA_NAME="Solar_ci64_every2.csv"
    SAVE_PATH="Solar_seed${SEED}/TST_rotary_ci64_every2"
    HYPER_BASE='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 256, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 3}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "$(echo "${HYPER_BASE}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac
