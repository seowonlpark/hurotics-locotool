"""
runner.py

run the algorithm exe file
"""

from __future__ import annotations

import os
import subprocess
import tempfile

def run_algorithm(exe_path: str, input_csv: str, output_csv: str | None = None, *, timeout: float | None = None):
    exe_path = os.path.abspath(exe_path)
    input_csv = os.path.abspath(input_csv)
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"algorithm not found: {exe_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    made_tmp = output_csv is None
    if made_tmp:
        fd, output_csv = tempfile.mkstemp(suffix="_pred.csv")
        os.close(fd)
    output_csv = os.path.abspath(output_csv)

    exe_dir = os.path.dirname(exe_path) or None
    cmd = [exe_path, input_csv, output_csv]

    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                  cwd=exe_dir, timeout=timeout)
        except OSError as e:
            raise RuntimeError(f"could not launch '{exe_path}': {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"algorithm timed out after {timeout} seconds") from e

        if proc.returncode != 0:
            raise RuntimeError(f"algorithm exited with code {proc.returncode}")

        if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
            raise RuntimeError("algorithm ran but produced no output CSV")
    except BaseException:
        if made_tmp and os.path.exists(output_csv):
            try:
                os.remove(output_csv)
            except OSError:
                pass
        raise

    return output_csv, proc.stdout
