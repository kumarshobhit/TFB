python ./scripts/run_benchmark.py \
--config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' \
--model-name "ts_benchmark.baselines.tst.TST" \
--model-hyper-params '{"period": 36, "d_model": 128, "n_heads": 8, "num_layers": 2, "dim_feedforward": 256, "dropout": 0.1, "pos_encoding": "sinespe", "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0001, "num_epochs": 100, "patience": 10}' \
--deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/TST_8thDec" --save-true-pred True
