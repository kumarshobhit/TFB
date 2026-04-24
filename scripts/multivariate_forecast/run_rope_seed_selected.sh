#!/usr/bin/env bash
set -euo pipefail

# Run the selected RoPE paper comparisons for one dataset and one seed.
# This launches the calibrated base, the default base, and the fixed control
# bases used in the main RoPE results table.

DATASET="${1:?dataset required}"
SEED="${2:?seed required}"
GPU="${3:-0}"

report_exists() {
  local save_path="$1"
  local report_stem="$2"
  compgen -G "result/${save_path}/test_report_rope_${report_stem}.csv" > /dev/null || \
    compgen -G "result/${save_path}/test_report_rope_${report_stem}_[0-9]*.csv" > /dev/null
}

run_cmd() {
  local data_name="$1"
  local save_path="$2"
  local report_stem="$3"
  local hyper_json="$4"
  if report_exists "${save_path}" "${report_stem}"; then
    echo "[skip] ${save_path} already has test_report_rope_${report_stem}*.csv"
    return
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" python ./scripts/run_benchmark.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "${data_name}" \
    --strategy-args '{"horizon": 96}' \
    --seed "${SEED}" \
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
    DATA_NAME="ETTh1.csv"
    if [[ "${SEED}" == "2021" ]]; then
      SAVE_PATH="ETTh1_22ndfeb/TST_rotary"
    else
      SAVE_PATH="ETTh1_seed${SEED}/TST_rotary"
    fi
    COMMON='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "45200.1" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 45200.09741519552/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "10000" "${COMMON}"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "5000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 5000.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "500" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "50000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  ETTm1)
    DATA_NAME="ETTm1.csv"
    SAVE_PATH="ETTm1_seed${SEED}/TST_rotary"
    COMMON='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 512, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 16, "lr": 0.0001, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "1436.17" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 1436.1719287648493/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "10000" "${COMMON}"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "5000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 5000.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "500" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "50000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  Weather)
    DATA_NAME="Weather.csv"
    SAVE_PATH="Weather_seed${SEED}/TST_rotary"
    COMMON='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 256, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0005, "num_epochs": 100, "patience": 5}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "131.291" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 131.29124656075606/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "10000" "${COMMON}"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "5000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 5000.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "500" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "50000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  Solar)
    DATA_NAME="Solar_ci64_every2.csv"
    SAVE_PATH="Solar_seed${SEED}/TST_rotary_ci64_every2"
    COMMON='{"pos_encoding": "rotary", "d_model": 128, "n_heads": 8, "num_layers": 1, "dim_feedforward": 256, "dropout": 0.1, "norm": true, "channel_independence": true, "revin": true, "seq_len": 96, "horizon": 96, "batch_size": 4, "lr": 0.0005, "num_epochs": 100, "patience": 3}'
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "17237.4" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 17237.391423477242/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "10000" "${COMMON}"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "5000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 5000.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "500" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 500.0/')"
    run_cmd "${DATA_NAME}" "${SAVE_PATH}" "50000" "$(echo "${COMMON}" | sed 's/"pos_encoding": "rotary"/"pos_encoding": "rotary", "base_freq": 50000.0/')"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac
