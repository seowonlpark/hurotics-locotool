"""
core.py

joins together the csv

alignment model: rows are always joined by position (row index)
    - timestamp agreement and feature-echo agreement between input/output are corroborating checks on that join, surfaced as flags (time_align_ok, feature_echo)
    - invalidate when
        - row-count mismatch across files
        - any label is NaN (missing/unparseable) in either gt or pred
    * note gaps in timeline are flagged but does not raise error
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# tunable constants
TIME_ALIGN_TOL_MS = 10.0   # max abs diff (ms) for input/output timestamps to count as aligned (100Hz)
ECHO_ALIGN_TOL = 1.0       # max abs diff (feature units, deg / deg/s) for input/output feature echo to count as aligned
GAP_MAX = 15.0             # gap threshold in ms

# locomotion modes (current algorithm) - more can be added in the future
LOCO_NAMES = {0: "STANDING", 5: "BENDING", 10: "WALKING",
              20: "SIT_TO_STAND", 25: "STAND_TO_SIT", 30: "SITTING", 255: "UNKNOWN"}

def cname(c) -> str:
    return LOCO_NAMES.get(int(c), f"CLASS_{int(c)}")

@dataclass
class HealthReport:
    n_rows: int
    feature_echo: bool
    time_align_ok: bool
    n_gaps: int             # number of gaps in timeline
    gap_idx: list           # indices of rows where gaps start (first row of gap)
    duration_s: float       # total duration of timeline (s)
    nominal_dt_ms: float
    alignment_ok: bool      # time_align_ok AND feature_echo
    alignment_method: str = "row_index"
    notes: list = field(default_factory = list)

@dataclass
class AlignedData:
    df: pd.DataFrame
    health: HealthReport

# col detection candidates
TIME_CANDS  = ["time", "00_time", "time_ms"]
LABEL_CANDS = ["label", "gt", "ground_truth", "truth"]
PRED_CANDS  = ["prediction", "pred", "status"]

# best-effort feature echo pairs
ECHO_PAIRS = [
    ("l_ang",    ["L_ang_LPF",    "02_L_Deg_Y"],  ["L_ang"]),
    ("r_ang",    ["R_ang_LPF",    "11_R_Deg_Y"],  ["R_ang"]),
    ("l_angvel", ["L_angvel_LPF", "06_L_Gyro_Z"], ["L_angvel"]),
    ("r_angvel", ["R_angvel_LPF", "15_R_Gyro_Z"], ["R_angvel"]),
]


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# in case of capitalization or other variance, find col that matches any candidates
def _find_col(cols, candidates):
    lower_case = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_case:
            return lower_case[cand.lower()]
    return None

# convert vals to numbers
def _num(col):
    return pd.to_numeric(col, errors = "coerce")

# calculate dt array from time array (prepend NaN for first row)
def _dt_ms(t: np.ndarray) -> np.ndarray:
    return np.diff(t, prepend = np.nan)

MAX_BAD_ROWS_SHOWN = 5

# source csv is malformed if any label is NaN (missing/unparseable)
def _check_nan_label(df: pd.DataFrame, gt_source: str, pred_source: str) -> None:
    for col, source in (("gt", gt_source), ("pred", pred_source)):
        bad = np.where(df[col].isna().to_numpy())[0]
        if bad.size:
            shown = ", ".join(str(int(i)) for i in bad[:MAX_BAD_ROWS_SHOWN])
            more = f", +{bad.size - MAX_BAD_ROWS_SHOWN} more" if bad.size > MAX_BAD_ROWS_SHOWN else ""
            raise ValueError(
                f"{source} has unparseable label "
                f"at row(s) {shown}{more}; labels must be integers, "
                f"file is malformed")

# align input, output, and ground-truth csvs into one dataframe (by row index)
def load_aligned(input_csv: str, output_csv: str, gt_csv: str) -> AlignedData:
    input_df = _read(input_csv)
    output_df = _read(output_csv)
    gt_src = _read(gt_csv)

    notes = []

    time_col = _find_col(input_df.columns, TIME_CANDS)
    pred_col = _find_col(output_df.columns, PRED_CANDS)
    label_col = _find_col(gt_src.columns, LABEL_CANDS)

    # error checks for missing required cols
    if time_col is None:
        raise ValueError("input csv has no recognizable time column "
                         "(time column is required for alignment)")
    if pred_col is None:
        raise ValueError(f"output csv has no recognizable prediction column "
                         f"(looked for {PRED_CANDS}); columns were: {list(output_df.columns)}")
    if label_col is None:
        raise ValueError(f"ground-truth csv has no recognizable label column "
                         f"(looked for {LABEL_CANDS}); columns were: {list(gt_src.columns)}")

    # error check for row-count mismatch
    if not (len(input_df) == len(output_df) == len(gt_src)):
        raise ValueError(f"row-count mismatch across files: "
                         f"input = {len(input_df)}, output = {len(output_df)}, gt = {len(gt_src)}")

    n = len(input_df)
    t = _num(input_df[time_col]).to_numpy(dtype=float)

    df = pd.DataFrame({
        "time_ms": t,
        "gt":   _num(gt_src[label_col]).round().astype("Int64").to_numpy(),
        "pred": _num(output_df[pred_col]).round().astype("Int64").to_numpy(),
    })

    # checking just the label cols for any NaNs
    _check_nan_label(
        df,
        gt_source = f"ground-truth csv '{gt_csv}' (column '{label_col}')",
        pred_source = f"output csv '{output_csv}' (column '{pred_col}')"
    )

    # verify alignment by comparing timestamps in input and output
    time_align_ok = True
    output_time_col = _find_col(output_df.columns, TIME_CANDS)
    if output_time_col:
        output_t = _num(output_df[output_time_col]).to_numpy(dtype = float)
        m = ~(np.isnan(t) | np.isnan(output_t)) # only compare rows where both timestamps are valid
        if m.any() and not np.allclose(t[m], output_t[m], atol = TIME_ALIGN_TOL_MS, rtol = 0):
            time_align_ok = False
            notes.append(f"input/output timestamps disagree beyond {TIME_ALIGN_TOL_MS}ms tolerance")
    else:
        notes.append("no recognizable time column in output csv")

    # confirms the row mapping when columns are present (echo)
    feature_echo_ok = True
    echo_checked = 0
    for canon, in_cands, out_cands in ECHO_PAIRS:
        ic = _find_col(input_df.columns, in_cands)
        oc = _find_col(output_df.columns, out_cands)
        if ic and oc:
            df[canon] = _num(input_df[ic]).to_numpy()
            a = df[canon].to_numpy(dtype = float)
            b = _num(output_df[oc]).to_numpy(dtype = float)
            m = ~(np.isnan(a) | np.isnan(b))
            echo_checked += 1
            if m.any() and not np.allclose(a[m], b[m], atol = ECHO_ALIGN_TOL, rtol = 0):
                feature_echo_ok = False
                notes.append(f"feature echo '{canon}' disagrees beyond {ECHO_ALIGN_TOL} tolerance")

    if echo_checked == 0:
        notes.append("could not verify based on matching feature columns); "
                     "alignment only relies on row index")

    # timeline / gaps
    tt = df["time_ms"].to_numpy(dtype = float)
    dt = _dt_ms(tt)
    med = float(np.nanmedian(dt[1:]))
    in_gap = dt > GAP_MAX 
    df["dt_ms"] = dt
    df["in_gap"] = in_gap

    feature_echo_flag = feature_echo_ok if echo_checked else True
    health = HealthReport(
        n_rows = n,
        feature_echo = feature_echo_flag,
        time_align_ok = time_align_ok,
        n_gaps = int(np.nansum(in_gap)),
        gap_idx = [int(i) for i in np.where(in_gap)[0]],
        duration_s = float((tt[-1]-tt[0])/1000.0) if n > 1 else 0.0,
        nominal_dt_ms = med,
        alignment_ok = (time_align_ok and feature_echo_flag),
        alignment_method = "row_index",
        notes = notes,
    )
    return AlignedData(df = df, health = health)

# create AlignedData from arrays of ground-truth, prediction, and time (ms)
# used for selftest
def aligned_from_arrays(gt, pred, time_ms) -> AlignedData:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    t = np.asarray(time_ms, dtype = float)

    notes = []
    n = len(gt)

    df = pd.DataFrame({"time_ms": t,
                       "gt": pd.array(gt, dtype = "Int64"),
                       "pred": pd.array(pred, dtype = "Int64")})

    _check_nan_label(df, gt_source="gt array", pred_source="pred array")

    df["dt_ms"] = _dt_ms(t)
    df["in_gap"] = False
    med = float(np.nanmedian(_dt_ms(t)[1:])) if n > 1 else 10.0

    health = HealthReport(
        n_rows = n,
        feature_echo = True,
        time_align_ok = True,
        n_gaps = 0,
        gap_idx = [],
        duration_s = float((t[-1]-t[0])/1000) if n > 1 else 0.0,
        nominal_dt_ms = med,
        alignment_ok = True,
        alignment_method = "row_index",
        notes = notes,
    )
    return AlignedData(df, health)

# each transition point
def transitions(labels: np.ndarray, times: np.ndarray):
    a = np.asarray(labels)
    t = np.asarray(times, dtype=float)

    # indices @ point of change
    idx = np.where(a[1:] != a[:-1])[0] + 1

    # returns list of order (idx), instant it happened (time), the loco label it is from, and the label it is to
    return [{"idx": int(i), "time": float(t[i]), "frm": int(a[i-1]), "to": int(a[i])} for i in idx]

# each contiguous segment
def segments(labels: np.ndarray, times: np.ndarray):
    a = np.asarray(labels)
    t = np.asarray(times, dtype=float)
    n = len(a)

    if n == 0: return []

    bounds = [0] + list(np.where(a[1:] != a[:-1])[0] + 1) + [n]
    segs = []

    last_end_t = t[-1] + (t[-1] - t[-2]) if n > 1 else t[-1]
    for s, e in zip(bounds[:-1], bounds[1:]):
        end_t = t[e] if e < n else last_end_t
        segs.append({"label": int(a[s]), "start": s, "end": e,
                     "start_time": float(t[s]), "end_time": float(end_t),
                     "dur_ms": float(end_t - t[s])})
    
    # returns list of loco label, start index, end index, start time, end time, and duration (ms)
    return segs