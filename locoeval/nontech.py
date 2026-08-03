"""
nontech.py — nontechnical (QA-facing) evaluation report

Three separate, non-comparable structures (deliberately not merged into one
flat list — they answer different questions with different denominators):

  transition_pairs   — per (from,to) ground-truth transition: total/missed/late/on_time.
                        missed+late+on_time == total always, one denominator per pair.
  false_transitions  — predicted transitions with no gt counterpart. Denominator is the
                        SAME number (total predicted transitions in the corpus) on every
                        row -- it is not specific to the (from,to) pair shown.
  steady_state_errors — steady-state frame misclassifications. Denominator is that
                        from_state class's total frame support.
"""

from __future__ import annotations

import numpy as np

from .core import AlignedData, cname, transitions
from .measure import frame_metrics
from .diagnose import transition_timing, row_buckets, LAG_MAX_MS

WRONG_STEADY_STATE = "WRONG_STEADY_STATE"   # steady-state frames misclassified
OTHER = "OTHER"                             # catch-all; every non-OTHER-eligible error still lands here

DEFAULT_LATE_THRESHOLD_MS = LAG_MAX_MS


def _translated(name: str) -> bool:
    return not name.startswith("CLASS_")


def _f(x):
    return round(x, 4) if x is not None else None


def assemble_nontechnical(a: AlignedData, *, late_threshold_ms: float = DEFAULT_LATE_THRESHOLD_MS) -> dict:
    df = a.df
    n = len(df)

    fm = frame_metrics(a)
    tr = transition_timing(a)
    buckets_arr = row_buckets(a, transition_results=tr)

    # (a) composite: frame-level error rate
    total_frames = fm.scored_rows
    frame_error_rate = None if fm.overall_accuracy is None else round(1.0 - fm.overall_accuracy, 4)

    # (b) event sub-score: fraction of gt transitions mishandled
    total_gt_transitions = len(tr)
    n_missed = sum(1 for r in tr if r.omission_kind is not None)
    n_late_over = sum(1 for r in tr if r.omission_kind is None and r.offset_ms is not None
                       and r.offset_ms >= late_threshold_ms)
    mishandled = n_missed + n_late_over
    sub_scores = {
        "transition_mishandled": {
            "count": mishandled,
            "denominator": total_gt_transitions,
            "denominator_kind": "transitions",
            "fraction": (round(mishandled / total_gt_transitions, 4)
                         if total_gt_transitions else None),
            "late_threshold_ms": late_threshold_ms,
        }
    }

    # (c1) transition_pairs: one consolidated row per (from,to) — total/missed/late/on_time,
    # always summing back to total, so there is exactly one denominator per pair.
    pair_totals: dict[tuple[str, str], dict] = {}
    for r in tr:
        pt = pair_totals.setdefault((r.frm, r.to), {"total": 0, "missed": 0, "matched": 0, "late_over": 0})
        pt["total"] += 1
        if r.omission_kind is not None:
            pt["missed"] += 1
        else:
            pt["matched"] += 1
            if r.offset_ms is not None and r.offset_ms >= late_threshold_ms:
                pt["late_over"] += 1

    transition_pairs = []
    for (frm, to), pt in pair_totals.items():
        on_time = pt["matched"] - pt["late_over"]
        transition_pairs.append({
            "from_state": frm,
            "to_state": to,
            "total": pt["total"],
            "missed": pt["missed"],
            "late": pt["late_over"],
            "on_time": on_time,
            "translated": _translated(frm) and _translated(to),
        })

    # (c2) false_transitions: pred-side transitions with no gt counterpart. Denominator is
    # the corpus-wide total predicted transition count -- identical on every row here,
    # NOT specific to the (from,to) pair shown, so keep this list visually separate from
    # transition_pairs above.
    false_transitions: list[dict] = []
    total_pred_transitions = 0
    if n:
        gt_arr = df["gt"].to_numpy(dtype=int)
        pred_arr = df["pred"].to_numpy(dtype=int)
        t = df["time_ms"].to_numpy(dtype=float)
        consumed = {r.resolved_pred_idx for r in tr if r.resolved_pred_idx is not None}
        pred_transitions = transitions(pred_arr, t)
        total_pred_transitions = len(pred_transitions)
        false_pairs: dict[tuple[str, str], int] = {}
        for pt_ in pred_transitions:
            idx = pt_["idx"]
            if idx in consumed:
                continue
            key = (cname(gt_arr[idx]), cname(pt_["to"]))
            false_pairs[key] = false_pairs.get(key, 0) + 1
        for (frm, to), c in false_pairs.items():
            false_transitions.append({
                "from_state": frm,
                "to_state": to,
                "count": c,
                "denominator": total_pred_transitions,
                "probability": _f(c / total_pred_transitions) if total_pred_transitions else None,
                "translated": _translated(frm) and _translated(to),
            })

        # (c3) steady_state_errors: frame-level residue, by (gt, pred). Denominator is
        # that from_state class's own frame support -- also not comparable to the two
        # transition-based structures above.
        support_rows = {name: st["support_rows"] for name, st in fm.per_class.items()}
        steady_pairs: dict[tuple[str, str], int] = {}
        other_pairs: dict[tuple[str, str], int] = {}
        error_idxs = np.where(buckets_arr != "correct")[0]
        for idx in error_idxs:
            key = (cname(gt_arr[idx]), cname(pred_arr[idx]))
            target = steady_pairs if buckets_arr[idx] == "steady_confusion" else other_pairs
            target[key] = target.get(key, 0) + 1

        steady_state_errors: list[dict] = []
        for bucket_id, pairs in ((WRONG_STEADY_STATE, steady_pairs), (OTHER, other_pairs)):
            for (frm, to), c in pairs.items():
                denom = support_rows.get(frm, 0)
                steady_state_errors.append({
                    "bucket_id": bucket_id,
                    "from_state": frm,
                    "to_state": to,
                    "count": c,
                    "denominator": denom,
                    "probability": _f(c / denom) if denom else None,
                    "translated": _translated(frm) and _translated(to),
                })
    else:
        steady_state_errors = []

    return {
        "frame_error_rate": frame_error_rate,
        "total_frames": total_frames,
        "sub_scores": sub_scores,
        "transition_pairs": transition_pairs,
        "false_transitions": false_transitions,
        "steady_state_errors": steady_state_errors,
        "total_pred_transitions": total_pred_transitions,
    }
