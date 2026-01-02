python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Weather.csv" --strategy-args '{"horizon": 96}' \
--model-name "tst.TST" \
--model-hyper-params '{"pos_encoding": "sinespe", "d_model": 64, "n_heads": 4, "num_layers": 1, "dim_feedforward": 256, "dropout": 0.1, "norm": true, "seq_len": 192, "horizon": 96, "batch_size": 64, "lr": 0.001, "num_epochs": 10, "patience": 3}' \
--deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Weather/TST_fast_run_1stjan_seq192" \