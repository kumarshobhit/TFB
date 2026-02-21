# -*- coding: utf-8 -*-

from __future__ import absolute_import

import base64
import io
import itertools
import logging
import os
import os.path
import pickle
from io import StringIO
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import ParserError

from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.utils.compress import (
    get_compress_method_from_ext,
    decompress,
    compress,
    get_compress_file_ext,
)
from ts_benchmark.utils.get_file_name import get_unique_file_suffix

logger = logging.getLogger(__name__)


PER_FEATURE_COLUMNS = [
    "feature_idx",
    "feature_name",
    "wape",
    "mae",
    "mse",
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


def compute_per_feature_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-feature metrics from decoded true/pred artifacts in a result DataFrame.
    """
    metric_rows = []
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
            feature_name = (
                feature_names[feature_idx]
                if feature_names is not None and feature_idx < len(feature_names)
                else f"feature_{feature_idx}"
            )
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
        return pd.DataFrame(columns=PER_FEATURE_COLUMNS)
    return pd.DataFrame(metric_rows, columns=PER_FEATURE_COLUMNS)


def save_per_feature_metrics(
    per_feature_df: pd.DataFrame, save_path: Optional[str], file_prefix: str
) -> Optional[str]:
    """
    Save pre-computed per-feature metrics to an additional CSV.
    """
    if per_feature_df.empty:
        return None

    result_path = _get_result_path(save_path)
    file_name = f"{file_prefix}_per_feature_metrics{get_unique_file_suffix()}.csv"
    file_path = os.path.join(result_path, file_name)
    per_feature_df.to_csv(file_path, index=False)
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

    record_filename = file_prefix + get_unique_file_suffix()
    file_path = os.path.join(result_path, record_filename)

    return write_record_file(result_df, file_path, compress_method)
