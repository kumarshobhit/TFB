python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Traffic.csv" --strategy-args '{"horizon": 96}' \
--model-name "tst.TST" \
--model-hyper-params '{"pos_encoding": "sinespe", "d_model": 512, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 8, "lr": 0.0001, "num_epochs": 100, "patience": 10}' \
--deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "Traffic/TST_dmodel512_sinespe_1stjan" \