#!/usr/bin/env python3
"""
selftest.py

self test for simplified base cases
"""
import sys

import numpy as np
from locoeval import aligned_from_arrays, transition_timing, timing_summary, bucket_errors
from locoeval import faults, assemble_nontechnical
from locoeval.core import segments
from locoeval.nontech import WRONG_STEADY_STATE, OTHER

DT = 10.0  # discrete time step in ms for synthetic data (10ms = 100Hz)
results = []

def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

# test adding known latency to prediction
def test_known_latency():
    gt, t = faults.make_synthetic(DT)
    for injected in (500.0, 2000.0):
        pred = faults.delay_transitions(gt, t, injected)
        a = aligned_from_arrays(gt, pred, t)
        summ = timing_summary(transition_timing(a))
        meds = [g["median_late_ms"] for g in summ if g["median_late_ms"] is not None]
        recovered = np.median(meds) if meds else None
        ok = recovered is not None and abs(recovered - injected) <= DT * 1.5
        check(f"recover {injected:.0f}ms latency", ok, f"recovered ≈ {recovered}ms")

#  test adding known omission to prediction (a real bout swallowed by the flanking label)
def test_omission():
    # ground truth with standing bout we then delete from the prediction
    gt = np.array([10]*400 + [0]*30 + [10]*400, dtype=int)  # 300ms standing bout at 10ms dt
    t = np.arange(len(gt)) * DT
    pred = faults.swallow_short_bouts(gt, t, label=0, max_dur_ms=400)
    a = aligned_from_arrays(gt, pred, t)
    summ = timing_summary(transition_timing(a))
    missed = sum(g["missed"] for g in summ)
    b = bucket_errors(a)
    ok = missed >= 1 and b.counts["swallowed"] > 0
    check("detect omission of short bout (swallowed B -> A -> B)", ok,
          f"missed = {missed}, swallowed_rows = {b.counts['swallowed']}")

# test adding known confusion bucket to prediction (sustained misclassification & flicker)
def test_confusion_bucket():
    gt, t = faults.make_synthetic(DT)
    pred = gt.copy()
    # find a WALKING(10) segment and relabel a 1s middle chunk to BENDING (5)
    seg = next(s for s in segments(gt, t) if s["label"] == 10 and s["dur_ms"] > 3000)
    mid = (seg["start"] + seg["end"]) // 2
    pred[mid:mid + 100] = 5 # 100 rows * 10ms = 1000ms block
    a = aligned_from_arrays(gt, pred, t)
    b = bucket_errors(a)
    ok = b.counts["steady_confusion"] > 0
    check("sustained confusion -> steady_confusion", ok,
          f"steady_confusion_rows = {b.counts['steady_confusion']}, flicker_rows = {b.counts['flicker']}")

# test that a long early run (covering most of the preceding segment) is flagged sustained_early
# and excluded from the clean latency stats -- distinct from full_segment_miss, which requires
# the run to cover the *entire* segment
def test_sustained_early():
    gt = np.array([0]*1000 + [10]*400, dtype=int)
    t = np.arange(len(gt)) * DT
    pred = gt.copy()
    pred[400:1000] = 10   # pred flips to WALKING 60% of the way through the STANDING segment,
                           # well before gt actually transitions at row 1000
    a = aligned_from_arrays(gt, pred, t)
    tr = transition_timing(a)
    summ = timing_summary(tr)
    r, g = tr[0], summ[0]
    ok = (r.sustained_early and not r.full_segment_miss
          and g["sustained_early"] == 1 and g["offsets_ms"] == [])
    check("long early run (60% of segment) -> sustained_early, excluded from clean stats", ok,
          f"sustained_early={r.sustained_early} full_segment_miss={r.full_segment_miss} "
          f"excluded_count={g['sustained_early']} offsets={g['offsets_ms']}")

# mirror of test_sustained_early: a long late run (covering most of the segment pred is catching
# up *into*) is flagged sustained_late and excluded from the clean latency stats
def test_sustained_late():
    gt = np.array([0]*400 + [10]*1000, dtype=int)
    t = np.arange(len(gt)) * DT
    pred = gt.copy()
    pred[400:1000] = 0    # pred stays STANDING 60% of the way through the WALKING segment,
                           # well after gt already transitioned at row 400
    a = aligned_from_arrays(gt, pred, t)
    tr = transition_timing(a)
    summ = timing_summary(tr)
    r, g = tr[0], summ[0]
    ok = (r.sustained_late and g["sustained_late"] == 1 and g["offsets_ms"] == [])
    check("long late run (60% of segment) -> sustained_late, excluded from clean stats", ok,
          f"sustained_late={r.sustained_late} excluded_count={g['sustained_late']} "
          f"offsets={g['offsets_ms']}")

# test ideal prediction data (clean)
def test_clean():
    gt, t = faults.make_synthetic(DT)
    a = aligned_from_arrays(gt, gt.copy(), t)
    summ = timing_summary(transition_timing(a))
    # every catch-up is instantaneous (offset 0), so median late/early lag is ~0 —
    # check the raw clean offsets directly for a tight per-event bound.
    all_offs = [o for g in summ for o in g["offsets_ms"]]
    max_lag = max((abs(o) for o in all_offs), default=0)
    missed = sum(g["missed"] for g in summ)
    ok = max_lag <= DT and missed == 0
    check("clean data -> ~zero lag, no misses", ok, f"max_lag = {max_lag}ms missed = {missed}")

# test that every row lands in exactly one bucket: correct + error buckets == all rows
def test_exhaustive():
    gt, t = faults.make_synthetic(DT)
    pred = gt.copy()
    seg = next(s for s in segments(gt, t) if s["label"] == 10 and s["dur_ms"] > 3000)
    mid = (seg["start"] + seg["end"]) // 2
    pred[mid:mid + 100] = 5
    a = aligned_from_arrays(gt, pred, t)
    b = bucket_errors(a)
    n = len(a.df)
    accounted = b.correct_rows + b.total_error_rows
    check("bucket assignment is exhaustive", accounted == n, f"accounted = {accounted}/{n}")

# test that a class present in GT but never predicted scores F1=0 (never NaN/None),
# so it can't silently drop out of macro-F1 or dodge the weak-class finding
def test_missed_class_f1():
    from locoeval import frame_metrics
    from locoeval.diagnose import build_findings
    gt   = np.array([0]*100 + [5]*100 + [0]*100)
    pred = np.zeros_like(gt)                     # BENDING(5) fully missed
    a = aligned_from_arrays(gt, pred, np.arange(len(gt)) * DT)
    fm = frame_metrics(a)
    f1 = fm.per_class["BENDING"]["f1"]
    weak = any(f.kind == "weak_class" and "BENDING" in f.claim for f in build_findings(a, fm=fm))
    ok = f1 == 0.0 and fm.macro_f1 is not None and fm.macro_f1 < 0.6 and weak
    check("fully-missed class -> F1 0, drags macro-F1, flagged weak", ok,
          f"f1 = {f1}, macro_f1 = {fm.macro_f1}, weak_finding = {weak}")

# test that assemble_nontechnical's transition_pairs entries recover an injected latency
# once it clears the default threshold (mirrors test_known_latency, nontechnical side).
# missed+late+on_time must equal total for every pair -- one shared denominator.
def test_nontech_late_transition():
    gt, t = faults.make_synthetic(DT)
    pred = faults.delay_transitions(gt, t, 2000.0)
    a = aligned_from_arrays(gt, pred, t)
    rep = assemble_nontechnical(a)
    pairs = rep["transition_pairs"]
    late_pairs = [p for p in pairs if p["late"] > 0]
    ok = (len(late_pairs) > 0
          and all(p["late"] == p["total"] for p in late_pairs)
          and all(p["missed"] == 0 for p in pairs)
          and all(p["missed"] + p["late"] + p["on_time"] == p["total"] for p in pairs))
    check("2000ms injected delay -> every occurrence late, nothing missed", ok,
          f"late_pairs={late_pairs}")

# test that assemble_nontechnical's transition_pairs entry catches a swallowed bout
# (mirrors test_omission, nontechnical side) and that the event sub-score matches
def test_nontech_missed_transition():
    gt = np.array([10]*400 + [0]*30 + [10]*400, dtype=int)
    t = np.arange(len(gt)) * DT
    pred = faults.swallow_short_bouts(gt, t, label=0, max_dur_ms=400)
    a = aligned_from_arrays(gt, pred, t)
    rep = assemble_nontechnical(a)
    pairs = {(p["from_state"], p["to_state"]): p for p in rep["transition_pairs"]}
    m = pairs.get(("WALKING", "STANDING"))
    ts = rep["sub_scores"]["transition_mishandled"]
    ok = (m is not None and m["missed"] == 1 and m["total"] == 1
          and ts["count"] == 1 and ts["denominator"] == 2)
    check("swallowed bout -> transition_pairs WALKING->STANDING missed 1/1, sub-score 1/2", ok,
          f"pair={m}, sub_score={ts}")

# test that a transient pred-only class (never appears in gt) surfaces in false_transitions,
# attributed to the surrounding gt state, with translated=False on the CLASS_n side
def test_nontech_false_transition():
    gt, t = faults.make_synthetic(DT)
    pred = gt.copy()
    seg = next(s for s in segments(gt, t) if s["label"] == 10 and s["dur_ms"] > 3000)
    mid = (seg["start"] + seg["end"]) // 2
    pred[mid:mid + 15] = 99   # 150ms transient blip to an unmapped class, mid-WALKING segment
    a = aligned_from_arrays(gt, pred, t)
    rep = assemble_nontechnical(a)
    hits = [e for e in rep["false_transitions"] if e["to_state"] == "CLASS_99"]
    ok = (len(hits) == 1 and hits[0]["from_state"] == "WALKING" and hits[0]["count"] == 1
          and hits[0]["translated"] is False)
    check("transient CLASS_99 blip -> false_transitions WALKING->CLASS_99, translated=False", ok,
          f"hits={hits}")

# test that WRONG_STEADY_STATE + OTHER frame counts exactly sum to the technical report's
# total_error_rows (mirrors test_confusion_bucket / test_exhaustive, nontechnical side)
def test_nontech_exhaustive():
    gt, t = faults.make_synthetic(DT)
    pred = gt.copy()
    seg = next(s for s in segments(gt, t) if s["label"] == 10 and s["dur_ms"] > 3000)
    mid = (seg["start"] + seg["end"]) // 2
    pred[mid:mid + 100] = 5
    a = aligned_from_arrays(gt, pred, t)
    b = bucket_errors(a)
    rep = assemble_nontechnical(a)
    ws = sum(e["count"] for e in rep["steady_state_errors"] if e["bucket_id"] == WRONG_STEADY_STATE)
    other = sum(e["count"] for e in rep["steady_state_errors"] if e["bucket_id"] == OTHER)
    ok = (ws + other == b.total_error_rows and ws == b.counts["steady_confusion"])
    check("WRONG_STEADY_STATE + OTHER frame counts == technical total_error_rows", ok,
          f"WRONG_STEADY_STATE={ws} OTHER={other} total_error_rows={b.total_error_rows}")

# test that a missing label is rejected at construction (unknown must be 255, never NaN)
def test_nan_label_rejected():
    t = np.arange(4) * DT
    for name, gt, pred in (("gt", [0, None, 0, None], [0, 0, 0, 0]),
                           ("pred", [0, 0, 0, 0], [0, 0, None, 0])):
        try:
            aligned_from_arrays(gt, pred, t)
            check(f"missing {name} label raises ValueError", False, "no exception r" \
            "aised")
        except ValueError as e:
            check(f"missing {name} label raises ValueError", True, str(e))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    print("SELFTEST — fault injection (no real data)\n")
    test_known_latency()
    test_omission()
    test_sustained_early()
    test_sustained_late()
    test_confusion_bucket()
    test_clean()
    test_exhaustive()
    test_missed_class_f1()
    test_nontech_late_transition()
    test_nontech_missed_transition()
    test_nontech_false_transition()
    test_nontech_exhaustive()
    test_nan_label_rejected()
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    raise SystemExit(0 if all(results) else 1)
