"""
report.py — generate evaluation reports
"""

from __future__ import annotations
import dataclasses
import json
import math
import os
import numpy as np
from .core import AlignedData
from .measure import frame_metrics
from .diagnose import diagnose

def assemble(a: AlignedData) -> dict:
    fm = frame_metrics(a)
    d = diagnose(a, fm = fm)
    rep = {
        "health": dataclasses.asdict(a.health),
        "frame_metrics": dataclasses.asdict(fm),
        "transition_timing": d.timing,
        "error_buckets": dataclasses.asdict(d.buckets),
        "in_gap_rows": a.health.gap_idx,  # data-quality note only; doesn't affect calc
        "findings": [dataclasses.asdict(f) for f in d.findings],
    }
    return rep

def print_problems(rep: dict) -> None:
    print("PROBLEMS")
    if rep.get("health", {}).get("notes"):
        for n in rep["health"]["notes"]:
            print(f"  - {n}")
    in_gap_rows = rep.get("in_gap_rows") or []
    if in_gap_rows:
        print(f"  - data quality: {len(in_gap_rows)} row(s) flagged in_gap: {_fmt_rows(in_gap_rows)}")
    findings = rep.get("findings", [])
    if not findings:
        print("  None")
        return
    for f in findings:
        print(_fmt_finding(f))

def _fmt_rows(rows, max_shown = 20):
    if len(rows) <= max_shown:
        return ", ".join(map(str, rows))
    return ", ".join(map(str, rows[:max_shown])) + f", … (+{len(rows) - max_shown} more)"

def _fmt_finding(f, indent = "  "):
    sev = f["severity"].upper()
    return f"{indent}[{sev:<8}] {f['claim']}"


# recursively convert numpy scalars/arrays (which json.dump can't serialize) to plain
# python types; NaN/Inf become None rather than invalid JSON literals
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# write one trial's technical + nontechnical reports as
# report_technical_<basename>.json / report_nontechnical_<basename>.json under dest_dir
def write_reports(rep: dict, nontechnical: dict, dest_dir: str, basename: str) -> tuple[str, str]:
    os.makedirs(dest_dir, exist_ok=True)
    technical_path = os.path.join(dest_dir, f"report_technical_{basename}.json")
    nontechnical_path = os.path.join(dest_dir, f"report_nontechnical_{basename}.json")
    with open(technical_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(rep), fh, indent=2, ensure_ascii=False)
    with open(nontechnical_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(nontechnical), fh, indent=2, ensure_ascii=False)
    return technical_path, nontechnical_path
