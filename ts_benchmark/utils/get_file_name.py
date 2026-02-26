# -*- coding: utf-8 -*-
import ast
import json
import os
import re
import socket
import time
from typing import Any, Dict


def get_unique_file_suffix():
    """
    Generate a log file name suffix that includes the following information:

    - Hostname
    - The current timestamp, in seconds, is the number of seconds since the Unix era
    - PID (process identifier) of the process

    Return:
    str: The name of the generated log file, in the format '.timestamp.hostname.pid.csv'

    For example, if the host name is' myhost ', the current timestamp is 1631655702, and the current process ID is 12345
    The returned file name may be '.1631655702.myhost.12345.csv'.
    """
    # Get Host Name
    hostname = socket.gethostname()

    # Get current timestamp (seconds since Unix era)
    timestamp = int(time.time())

    # Obtain the PID (process identifier) of the process
    pid = os.getpid()

    # Build file name
    log_filename = f".{timestamp}.{hostname}.{pid}.csv"
    return log_filename


def parse_dict_like(value: Any) -> Dict:
    """
    Parse dict-like values from dict, JSON string, or Python-literal string.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or value == "":
        return {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, dict):
            return parsed
    except (SyntaxError, ValueError):
        pass
    return {}


def format_float_compact(value: Any) -> str:
    """
    Format numeric values in a compact and readable way.
    """
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def sanitize_filename_token(value: Any) -> str:
    """
    Convert arbitrary text into a filesystem-safe token.
    """
    token = str(value).strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_.-")
    return token


def build_pe_tag_from_model_params(model_params: Any) -> str:
    """
    Build a PE tag from model params:
    - rope_<base>
    - sinespe_<freq>
    - sinespe_default
    """
    params = parse_dict_like(model_params)
    pos_encoding = str(params.get("pos_encoding", "")).lower()
    if pos_encoding == "rotary":
        base_freq = params.get("base_freq", 10000.0)
        return sanitize_filename_token(f"rope_{format_float_compact(base_freq)}")
    if pos_encoding == "sinespe":
        spe_freq = params.get("spe_freq", None)
        if spe_freq is None:
            return "sinespe_default"
        try:
            spe_float = float(spe_freq)
        except (TypeError, ValueError):
            return "sinespe_default"
        if spe_float <= 0:
            return "sinespe_default"
        return sanitize_filename_token(f"sinespe_{format_float_compact(spe_float)}")
    return ""


def resolve_nonconflicting_path(path: str) -> str:
    """
    Return a non-conflicting path by appending _N before extension.
    Supports multi-part extension .tar.gz.
    """
    if not os.path.exists(path):
        return path

    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    if basename.endswith(".tar.gz"):
        stem = basename[: -len(".tar.gz")]
        ext = ".tar.gz"
    else:
        stem, ext = os.path.splitext(basename)

    i = 1
    while True:
        candidate = os.path.join(dirname, f"{stem}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1
