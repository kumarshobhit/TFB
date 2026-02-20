# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, NoReturn, List

import torch
import numpy as np

from scipy.spatial.distance import cosine
from scipy import signal
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend


sys.path.insert(0, THIRD_PARTY_PATH)

warnings.filterwarnings("ignore")


def str_to_bool(value: str) -> bool:
    """
    Converts a string to a boolean: True for 'True', '1', or 'T'; False for 'False', '0', or 'F'.
    """
    if value.lower() in ["true", "1", "t"]:
        return True
    elif value.lower() in ["false", "0", "f"]:
        return False
    else:
        raise ValueError("Invalid boolean value. Please enter 'True' or 'False'.")

def compute_psd(data, fs=1.0):
    """Compute Power Spectral Density using Periodogram."""
    freqs, psd = signal.periodogram(data, fs=fs, detrend='linear')
    return freqs, psd

def analyze_frequency_alignment(data, seq_len, d_model=512):
    """Analyze the frequency alignment of input data."""
    # 1. Generate a "reference" sine wave (you can customize this)
    t = np.arange(seq_len)
    ref_signal = np.sin(2 * np.pi * t / 24)  # Example: 24-hour period

    # 2. Calculate PSD for the reference signal
    f_ref, psd_ref = compute_psd(ref_signal)

    # 3. Calculate PSD for the input data
    f_data, psd_data = compute_psd(data[:seq_len])

    # 4. Resize the arrays if they have different length
    min_len = min(len(psd_ref), len(psd_data))
    psd_ref = psd_ref[:min_len]
    psd_data = psd_data[:min_len]

    # 5. Normalize the PSDs to compare shape, not magnitude
    psd_ref /= psd_ref.sum()
    psd_data /= psd_data.sum()

    # 6. Calculate Cosine Similarity
    similarity = 1 - cosine(psd_ref.flatten(), psd_data.flatten())
    return similarity

def extract_per_feature_metrics(preds, actuals):
    """
    Calculate per-feature metrics (MSE, MAE)
    """
    mse_list = []
    mae_list = []
    for i in range(preds.shape[1]):  # Iterate over features
        mse = np.mean((actuals[:, i] - preds[:, i]) ** 2)
        mae = np.mean(np.abs(actuals[:, i] - preds[:, i]))
        mse_list.append(mse)
        mae_list.append(mae)

    return mse_list, mae_list


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    if args.model_hyper_params is not None:
        args.model_hyper_params = [
            None if item == "None" else item for item in args.model_hyper_params
        ]
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend(
                [None] * (len(args.model_name) - len(args.model_hyper_params))
            )
    else:
        args.model_hyper_params = [None] * len(args.model_name)

    for adapter, model_name, model_hyper_params in zip(
        args.adapter, args.model_name, args.model_hyper_params
    ):
        model_config["models"].append(
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": (
                    json.loads(model_hyper_params)
                    if model_hyper_params is not None
                    else {}
                ),
            }
        )

    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]
    evaluation_config["save_path"] = args.save_path

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = (
        json.loads(args.strategy_args) if args.strategy_args else None
    )

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed
    if args.save_true_pred is not None:
        default_strategy_args["save_true_pred"] = args.save_true_pred
    default_strategy_args["deterministic"] = args.deterministic

    if args.base_freq is not None:
         default_strategy_args["base_freq"] = args.base_freq

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Evaluation config file path",
    )

    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="List of series names entered by the user",
    )

    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user,"
        "only takes effect when data_name_list is not specified",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapter used to adapt the method to our pipeline",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="The relative path of the model that needs to be evaluated",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # evaluation_config
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )
    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
        "by default, use the seed value in the config file",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="efficient",
        choices=["full", "efficient", "none"],
        help="Specify the type of deterministic behavior for the algorithm. Options are: "
        "'full': Enables full deterministic mode. "
        "'efficient': Fixes only some seeds for efficiency. "
        "'none': No deterministic behavior is applied.",
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of gpu devices to use, only available in ray backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Time limit for each evaluation task, in seconds",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )

    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )

    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="The relative path for saving evaluation results, relative to the result folder",
    )

    parser.add_argument(
        "--save-true-pred",
        type=str_to_bool,
        default=None,
        help="If true, saves the model's prediction results "
        "and the true values in evaluation result file",
    )
    parser.add_argument(
        "--base_freq",
        type=float,
        default=None,
        help="Base frequency for RoPE",
    )


    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )

    try:
        log_filenames = pipeline(
            data_config,
            model_config,
            evaluation_config,
        )

        # Extract per-feature metrics and alignment after pipeline execution
        all_metrics = []
        for log_file in log_filenames:
            # Load data from the log file
            df_results = pd.read_csv(log_file)

            # Extract predictions and actual values (assuming they're saved)
            if 'inference_data' in df_results.columns and 'actual_data' in df_results.columns:
                preds = np.array(json.loads(df_results['inference_data'][0]))
                actuals = np.array(json.loads(df_results['actual_data'][0]))

                # Calculate per-feature metrics
                mse_list, mae_list = extract_per_feature_metrics(preds, actuals)

                # Analyze frequency alignment (using the first feature for simplicity)
                alignment_score = analyze_frequency_alignment(actuals[:, 0], seq_len=data_config['seq_len'])

                # Store results
                df_results['Per-Feature MSE'] = [mse_list]
                df_results['Per-Feature MAE'] = [mae_list]
                df_results['cosine_similarity'] = alignment_score
                df_results.to_csv(log_file, index=False) # Save back to the log file

                print(f"Saving results to {log_file}")
                print(f"Per-feature metrics and cosine similarity added to {log_file}")
            else:
                print("Warning: 'inference_data' or 'actual_data' not found in log file.")

    finally:
        ParallelBackend().close(force=True)

    report_config["log_files_list"] = log_filenames
    if args.report_method == "csv":
        filename = get_unique_file_suffix()
        leaderboard_file_name = "test_report" + filename
        report_config["leaderboard_file_name"] = leaderboard_file_name
    report(report_config, report_method=args.report_method)
