# -*- coding: utf-8 -*-

from __future__ import absolute_import

import base64
import io
import itertools
import json
import logging
import os
import os.path
import pickle
from io import StringIO
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import FORECASTING_DATASET_PATH, ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import (
    build_pe_tag_from_model_params,
    resolve_nonconflicting_path,
)

logger = logging.getLogger(__name__)


PER_FEATURE_COLUMNS = [
    "feature_idx",
    "feature_name",
    "wape",
    "mae",
    "mse"
]

INTERNAL_PER_FEATURE_COLUMNS = [
    "model_name",
    "model_params",
    "strategy_args",
    "file_name",
    "feature_idx",
    "feature_name",
    "wape",
    "mae",
    "mse",
    "sum_abs_err",
    "sum_abs_actual",
    "num_points",
]

COSINE_COLUMNS = [
    "pe_type",
    "spe_freq",
    "spe_k",
    "base_freq",
    "n_heads",
    "seq_len",
    "d_model",
    "cosine_sim",
    "cosine_sim_default",
    "delta_cosine",
]


def _get_result_path(save_path: Optional[str]) -> str:
    if save_path is not None:
        result_path = (
            os.path.join(ROOT_PATH, "result", save_path)
            if not os.path.isabs(save_path)
            else save_path
        )
    else:
        result_path = os.path.join(ROOT_PATH, "result")
    os.makedirs(result_path, exist_ok=True)
    return result_path


def _decode_artifact(value: Any) -> Any:
    if not isinstance(value, str) or value == "":
        return None
    try:
        return pickle.loads(base64.b64decode(value))
    except Exception:
        return None


def _to_2d_array_with_columns(data: Any) -> Tuple[np.ndarray, Optional[List[str]]]:
    if isinstance(data, pd.DataFrame):
        return data.to_numpy(), [str(col) for col in data.columns]

    if isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim == 0:
            return arr.reshape(1, 1), None
        if arr.ndim == 1:
            return arr.reshape(-1, 1), None
        if arr.ndim == 2:
            return arr, None
        return arr.reshape(-1, arr.shape[-1]), None

    if isinstance(data, list):
        arrays = []
        column_names = None
        for item in data:
            item_arr, item_columns = _to_2d_array_with_columns(item)
            if item_arr.size == 0:
                continue
            if column_names is None and item_columns is not None:
                column_names = item_columns
            arrays.append(item_arr)
        if not arrays:
            return np.empty((0, 0)), column_names
        feat_dim = arrays[0].shape[1]
        if any(arr.shape[1] != feat_dim for arr in arrays):
            raise ValueError("Inconsistent feature dimensions in decoded artifacts.")
        return np.concatenate(arrays, axis=0), column_names

    raise TypeError(f"Unsupported decoded artifact type: {type(data)}")


def get_dataset_feature_names(file_name: Any) -> Optional[List[str]]:
    """
    Load feature names from dataset/forecasting/<file_name> for fallback labeling.
    """
    if not isinstance(file_name, str) or not file_name:
        return None
    dataset_df = _load_dataset_series(file_name)
    if dataset_df is None:
        return None
    return [str(col) for col in dataset_df.columns]


def _is_generic_feature_name(name: Any) -> bool:
    if name is None:
        return True
    name_str = str(name)
    return name_str == "" or name_str.startswith("feature_")


def compute_per_feature_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-feature metrics from decoded true/pred artifacts in a result DataFrame.
    """
    metric_rows = []
    dataset_feature_cache = {}
    for _, row in result_df.iterrows():
        actual_data = _decode_artifact(row.get("actual_data"))
        inference_data = _decode_artifact(row.get("inference_data"))
        if actual_data is None or inference_data is None:
            continue

        try:
            actual_arr, feature_names = _to_2d_array_with_columns(actual_data)
            pred_arr, _ = _to_2d_array_with_columns(inference_data)
        except (TypeError, ValueError):
            continue

        if actual_arr.shape != pred_arr.shape:
            continue
        if actual_arr.size == 0:
            continue

        num_points = actual_arr.shape[0]
        abs_diff = np.abs(actual_arr - pred_arr)
        sq_diff = np.square(actual_arr - pred_arr)
        sum_abs_err = np.sum(abs_diff, axis=0)
        sum_abs_actual = np.sum(np.abs(actual_arr), axis=0)
        sum_sq_err = np.sum(sq_diff, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            wape = np.where(sum_abs_actual > 0, sum_abs_err / sum_abs_actual * 100, np.nan)
        mae = sum_abs_err / num_points
        mse = sum_sq_err / num_points

        feature_count = actual_arr.shape[1]
        for feature_idx in range(feature_count):
            decoded_feature_name = (
                feature_names[feature_idx]
                if feature_names is not None and feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )
            feature_name = decoded_feature_name
            if _is_generic_feature_name(feature_name):
                file_name = row.get("file_name")
                if file_name not in dataset_feature_cache:
                    dataset_feature_cache[file_name] = get_dataset_feature_names(
                        file_name
                    )
                dataset_feature_names = dataset_feature_cache[file_name]
                if (
                    dataset_feature_names is not None
                    and 0 <= feature_idx < len(dataset_feature_names)
                ):
                    feature_name = dataset_feature_names[feature_idx]
            metric_rows.append(
                {
                    "model_name": row.get("model_name"),
                    "model_params": row.get("model_params"),
                    "strategy_args": row.get("strategy_args"),
                    "file_name": row.get("file_name"),
                    "feature_idx": feature_idx,
                    "feature_name": feature_name,
                    "wape": float(wape[feature_idx]),
                    "mae": float(mae[feature_idx]),
                    "mse": float(mse[feature_idx]),
                    "sum_abs_err": float(sum_abs_err[feature_idx]),
                    "sum_abs_actual": float(sum_abs_actual[feature_idx]),
                    "num_points": int(num_points),
                }
            )

    if not metric_rows:
        return pd.DataFrame(columns=INTERNAL_PER_FEATURE_COLUMNS)
    return pd.DataFrame(metric_rows, columns=INTERNAL_PER_FEATURE_COLUMNS)


def _is_tst_pe_row(row: pd.Series) -> bool:
    model_name = str(row.get("model_name", "")).lower()
    if "tst" not in model_name:
        return False
    params = _parse_model_params(row.get("model_params"))
    return str(params.get("pos_encoding", "")).lower() in {"sinespe", "rotary"}


def _parse_model_params(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value == "":
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _build_pe_config(row: pd.Series) -> Optional[dict]:
    params = _parse_model_params(row.get("model_params"))
    pos_encoding = str(params.get("pos_encoding", "")).lower()
    if pos_encoding not in {"sinespe", "rotary"}:
        return None

    seq_len = params.get("seq_len")
    d_model = params.get("d_model")
    if seq_len is None or d_model is None:
        return None

    try:
        seq_len = int(seq_len)
        d_model = int(d_model)
    except (TypeError, ValueError):
        return None

    if seq_len <= 1 or d_model <= 1:
        return None

    if pos_encoding == "sinespe":
        try:
            base_freq = float(params.get("base_freq", 10000.0))
        except (TypeError, ValueError):
            base_freq = 10000.0
        try:
            n_heads = int(params.get("n_heads", 8))
        except (TypeError, ValueError):
            n_heads = 8
        return {
            "pe_type": "sinespe",
            "spe_freq": np.nan,
            "spe_k": np.nan,
            "base_freq": base_freq,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "d_model": d_model,
        }

    try:
        base_freq = float(params.get("base_freq", 10000.0))
    except (TypeError, ValueError):
        base_freq = 10000.0
    try:
        n_heads = int(params.get("n_heads", 8))
    except (TypeError, ValueError):
        n_heads = 8
    if n_heads <= 0 or d_model % n_heads != 0:
        return None
    return {
        "pe_type": "rope",
        "spe_freq": np.nan,
        "spe_k": np.nan,
        "base_freq": base_freq,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "d_model": d_model,
    }

def _load_dataset_series(file_name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(FORECASTING_DATASET_PATH, file_name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "cols" in df.columns and "data" in df.columns:
        try:
            df = df.pivot(index="date", columns="cols", values="data").reset_index()
        except Exception:
            return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "date" in numeric_cols:
        numeric_cols = numeric_cols.drop("date", errors="ignore")
    if len(numeric_cols) == 0:
        return None
    return df[numeric_cols]


def _get_feature_values(
    dataset_df: pd.DataFrame, feature_name: Any, feature_idx: Any
) -> Optional[np.ndarray]:
    if feature_name in dataset_df.columns:
        data = dataset_df[feature_name].to_numpy(dtype=float, copy=False)
        return data

    feature_name_str = str(feature_name)
    if feature_name_str in dataset_df.columns:
        data = dataset_df[feature_name_str].to_numpy(dtype=float, copy=False)
        return data

    try:
        idx = int(feature_idx)
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= dataset_df.shape[1]:
        return None
    return dataset_df.iloc[:, idx].to_numpy(dtype=float, copy=False)


def _linear_detrend(signal: np.ndarray) -> np.ndarray:
    x = np.arange(signal.shape[0], dtype=float)
    coeffs = np.polyfit(x, signal, deg=1)
    return signal - (coeffs[0] * x + coeffs[1])


def _periodogram_1d(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 2:
        return np.array([])
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.array([])
    detrended = _linear_detrend(values)
    spectrum = np.fft.rfft(detrended)
    return np.abs(spectrum) ** 2


def _normalized_pe_spectrum(
    pe_type: str,
    seq_len: int,
    d_model: int,
    base_freq: float = 10000.0,
    n_heads: int = 8,
) -> Optional[np.ndarray]:
    positions = np.arange(seq_len, dtype=float)[:, None]
    if pe_type == "sinespe":
        if base_freq <= 0:
            return None
        div_term = np.exp(np.arange(0, d_model, 2, dtype=float) * -(np.log(base_freq) / d_model))
        encoding = np.zeros((seq_len, d_model), dtype=float)
        encoding[:, 0::2] = np.sin(positions * div_term)
        encoding[:, 1::2] = np.cos(positions * div_term)
    elif pe_type == "rope":
        if n_heads <= 0 or d_model % n_heads != 0:
            return None
        head_dim = d_model // n_heads
        if head_dim <= 1:
            return None
        theta = 1.0 / (base_freq ** (np.arange(0, head_dim, 2, dtype=float) / head_dim))
        idx_theta = positions * theta[None, :]
        head_waves = np.concatenate([np.cos(idx_theta), np.sin(idx_theta)], axis=1)
        if head_waves.shape[1] < head_dim:
            pad = np.zeros((seq_len, head_dim - head_waves.shape[1]))
            head_waves = np.concatenate([head_waves, pad], axis=1)
        head_waves = head_waves[:, :head_dim]
        encoding = np.tile(head_waves, (1, n_heads))
        encoding = encoding[:, :d_model]
    else:
        return None

    pe_psd = None
    for i in range(encoding.shape[1]):
        p = _periodogram_1d(encoding[:, i])
        if p.size == 0:
            continue
        if pe_psd is None:
            pe_psd = np.zeros_like(p)
        pe_psd += p
    if pe_psd is None or pe_psd.sum() <= 0:
        return None
    return pe_psd / (pe_psd.sum() + 1e-12)


def _normalized_feature_spectrum(values: np.ndarray, seq_len: int) -> Optional[np.ndarray]:
    if values is None or len(values) < seq_len:
        return None
    stride = max(1, seq_len // 2)
    psd_sum = None
    count = 0
    for start in range(0, len(values) - seq_len + 1, stride):
        window = values[start : start + seq_len]
        if np.std(window) < 1e-6:
            continue
        window = (window - np.mean(window)) / (np.std(window) + 1e-12)
        p = _periodogram_1d(window)
        if p.size == 0:
            continue
        if psd_sum is None:
            psd_sum = np.zeros_like(p)
        psd_sum += p
        count += 1
    if psd_sum is None or count == 0 or psd_sum.sum() <= 0:
        return None
    avg_psd = psd_sum / count
    return avg_psd / (avg_psd.sum() + 1e-12)


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    if a.shape != b.shape:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def enrich_per_feature_with_cosine(per_feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich per-feature metrics with cosine alignment for TST SineSPE/RoPE runs.
    """
    if per_feature_df.empty:
        return per_feature_df

    target_mask = per_feature_df.apply(_is_tst_pe_row, axis=1)
    if not target_mask.any():
        return per_feature_df

    enriched = per_feature_df.copy()
    for col in COSINE_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = np.nan

    data_cache = {}
    feature_spec_cache = {}
    pe_spec_cache = {}
    pe_default_cache = {}

    for idx, row in enriched[target_mask].iterrows():
        pe_cfg = _build_pe_config(row)
        if pe_cfg is None:
            continue

        file_name = row.get("file_name")
        if file_name not in data_cache:
            data_cache[file_name] = _load_dataset_series(file_name)
        dataset_df = data_cache[file_name]
        if dataset_df is None:
            continue

        seq_len = pe_cfg["seq_len"]
        feature_name = row.get("feature_name")
        feature_idx = row.get("feature_idx")
        spec_key = (file_name, feature_name, feature_idx, seq_len)
        if spec_key not in feature_spec_cache:
            values = _get_feature_values(dataset_df, feature_name, feature_idx)
            feature_spec_cache[spec_key] = _normalized_feature_spectrum(values, seq_len)
        feature_spec = feature_spec_cache[spec_key]
        if feature_spec is None:
            continue

        pe_key = (
            pe_cfg["pe_type"],
            pe_cfg["seq_len"],
            pe_cfg["d_model"],
            pe_cfg["base_freq"],
            pe_cfg["n_heads"],
        )
        if pe_key not in pe_spec_cache:
            pe_spec_cache[pe_key] = _normalized_pe_spectrum(
                pe_cfg["pe_type"],
                pe_cfg["seq_len"],
                pe_cfg["d_model"],
                base_freq=pe_cfg["base_freq"],
                n_heads=pe_cfg["n_heads"],
            )
        pe_spec = pe_spec_cache[pe_key]

        default_key = (
            pe_cfg["pe_type"],
            pe_cfg["seq_len"],
            pe_cfg["d_model"],
            pe_cfg["n_heads"],
        )
        if default_key not in pe_default_cache:
            if pe_cfg["pe_type"] == "sinespe":
                pe_default_cache[default_key] = _normalized_pe_spectrum(
                    "sinespe",
                    pe_cfg["seq_len"],
                    pe_cfg["d_model"],
                    base_freq=10000.0,
                    n_heads=pe_cfg["n_heads"],
                )
            else:
                pe_default_cache[default_key] = _normalized_pe_spectrum(
                    "rope",
                    pe_cfg["seq_len"],
                    pe_cfg["d_model"],
                    base_freq=10000.0,
                    n_heads=pe_cfg["n_heads"],
                )
        pe_default = pe_default_cache[default_key]

        cosine_sim = _cosine_similarity(feature_spec, pe_spec)
        cosine_default = _cosine_similarity(feature_spec, pe_default)
        delta = (
            cosine_sim - cosine_default
            if np.isfinite(cosine_sim) and np.isfinite(cosine_default)
            else np.nan
        )

        enriched.at[idx, "pe_type"] = pe_cfg["pe_type"]
        enriched.at[idx, "spe_freq"] = pe_cfg["spe_freq"]
        enriched.at[idx, "spe_k"] = pe_cfg["spe_k"]
        enriched.at[idx, "base_freq"] = pe_cfg["base_freq"]
        enriched.at[idx, "n_heads"] = pe_cfg["n_heads"]
        enriched.at[idx, "seq_len"] = pe_cfg["seq_len"]
        enriched.at[idx, "d_model"] = pe_cfg["d_model"]
        enriched.at[idx, "cosine_sim"] = cosine_sim
        enriched.at[idx, "cosine_sim_default"] = cosine_default
        enriched.at[idx, "delta_cosine"] = delta

    return enriched


def save_per_feature_metrics(
    per_feature_df: pd.DataFrame, save_path: Optional[str], file_prefix: str
) -> Optional[str]:
    """
    Save pre-computed per-feature metrics to an additional CSV.
    """
    if per_feature_df.empty:
        return None

    result_path = _get_result_path(save_path)
    model_params = per_feature_df.iloc[0].get("model_params", None)
    pe_tag = build_pe_tag_from_model_params(model_params)
    if pe_tag:
        file_name = f"{file_prefix}_per_feature_metrics_{pe_tag}.csv"
    else:
        file_name = f"{file_prefix}_per_feature_metrics.csv"
    file_path = os.path.join(result_path, file_name)
    file_path = resolve_nonconflicting_path(file_path)
    export_columns = [
        c for c in (PER_FEATURE_COLUMNS + COSINE_COLUMNS) if c in per_feature_df.columns
    ]
    if not export_columns:
        export_columns = list(per_feature_df.columns)
    per_feature_df.to_csv(file_path, index=False, columns=export_columns)
    return file_path


def read_record_file(fn: str) -> pd.DataFrame:
    """
    Reads a single record file.

    The format of the file is currently determined by the extension name.

    :param fn: Path to the record file.
    :return: Benchmarking records in DataFrame format.
    """
    ext = os.path.splitext(fn)[1]
    compress_method = get_compress_method_from_ext(ext)
    if compress_method is None:
        return pd.read_csv(fn)
    else:
        with open(fn, "rb") as fh:
            data = fh.read()
        data = decompress(data, method=compress_method)
        ret = []
        for k, v in data.items():
            ret.append(pd.read_csv(StringIO(v.decode("utf8"))))
        return pd.concat(ret, axis=0)


def write_record_file(
    result_df: pd.DataFrame,
    file_path: str,
    compress_method: Optional[str] = None,
) -> str:
    """
    Write to a single record file.

    :param result_df: Benchmarking records in DataFrame format.
    :param file_path: Path to the record file to save.
    :param compress_method: The format used to compress the record file, if None is given,
        no compression is applied.
    :return: Path to the record file written.
    """
    if compress_method is not None:
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        write_data = compress(
            {os.path.basename(file_path): buf.getvalue()}, method=compress_method
        )
        file_path = f"{file_path}.{get_compress_file_ext(compress_method)}"

        with open(file_path, "wb") as fh:
            fh.write(write_data)
    else:
        result_df.to_csv(file_path, index=False)

    return file_path


def load_record_data(
    record_files: List[str], drop_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads benchmarking records from multiple record files.

    :param record_files: The list of paths to the record files. Each item in the list can either
        be the path to a directory or a file. If it is a path to a directory, then all record files
        in the directory are loaded; Otherwise, the file specified by the path is loaded.
    :param drop_columns: The columns to drop during loading.
        This parameter is mainly used to save memory.
    :return: The loaded benchmarking records in DataFrame format.
    """
    record_files = itertools.chain.from_iterable(
        [
            [fn] if not os.path.isdir(fn) else find_record_files(fn)
            for fn in record_files
        ]
    )

    ret = []
    for fn in record_files:
        logger.info("loading log file %s", fn)
        try:
            cur_record = read_record_file(fn)
            if drop_columns:
                cur_record = cur_record.drop(columns=drop_columns)
            ret.append(cur_record)
        except (FileNotFoundError, PermissionError, KeyError, ParserError):
            # TODO: it is ugly to identify log files by artifact columns...
            logger.info("unrecognized log file format, skipping %s...", fn)
    return pd.concat(ret, axis=0)


def find_record_files(directory: str) -> List[str]:
    """
    Finds records files in a directory.

    :param directory: The path to the directory.
    :return: The list of file paths to the record files that are found in the give directory.
    """
    record_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            # TODO: this is a temporary solution, any good methods to identify a log file?
            if file.endswith(".csv") or file.endswith(".tar.gz"):
                record_files.append(os.path.join(root, file))
    return record_files


def save_log(
    result_df: pd.DataFrame, save_path, file_prefix: str, compress_method: str = "gz"
) -> str:
    """
    Save log data.

    Save the evaluation results, model hyperparameters, model evaluation configuration, and model name to a log file.

    :param result_df: Benchmarking records in DataFrame format.
    :param save_path: Path to the directory where the records are saved.
    :param file_prefix: Prefix of the file name to save the records.
    :param compress_method: The compression method for the output file.
    :return: The path to the output file.
    """
    if result_df["log_info"].any():
        error_itr = filter(None, result_df["log_info"])
        for error in itertools.islice(error_itr, 3):
            logger.info(error)
        if any(error_itr):
            logger.info(
                "-------------More error messages can be found in the record files!-------------"
            )

    result_path = _get_result_path(save_path)

    model_params = result_df.iloc[0].get("model_params", None)
    pe_tag = build_pe_tag_from_model_params(model_params)
    record_filename = f"{file_prefix}_{pe_tag}" if pe_tag else file_prefix
    file_path = os.path.join(result_path, record_filename)

    if compress_method is not None:
        ext = get_compress_file_ext(compress_method)
        final_path = resolve_nonconflicting_path(f"{file_path}.{ext}")
        file_path = final_path[: -(len(ext) + 1)]
    else:
        file_path = resolve_nonconflicting_path(file_path)

    return write_record_file(result_df, file_path, compress_method)
