python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1.csv" --strategy-args '{"horizon": 96}' \
--model-name "tst.TST" \
--model-hyper-params '{"channel_independence": true,"d_model": 512, "n_heads": 8, "num_layers": 1, "dim_feedforward": 2048, "dropout": 0.1, "pos_encoding": "learned", "norm": true, "seq_len": 96, "horizon": 96, "batch_size": 32, "lr": 0.0001, "num_epochs": 100, "patience": 10}' \
--deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1/TST_dmodel512_matched_learned"
