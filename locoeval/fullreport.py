"""
fullreport.py — corpus-level algorithm overview across all annotated trials.
"""

from __future__ import annotations
import csv as _csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .core import load_aligned, _find_col, TIME_CANDS, LABEL_CANDS
from .diagnose import ERROR_BUCKETS, MIN_EVENTS_FOR_STATISTIC
from .report import assemble, write_reports
from .runner import run_algorithm
from .nontech import assemble_nontechnical, DEFAULT_LATE_THRESHOLD_MS

_BUCKETS = ERROR_BUCKETS

@dataclass
class Trial:
    group_id: str   # containing folder path relative to input_root; "" if the file sits directly in input_root
    trial_id: str   # file path relative to input_root -- unique identifier for this trial
    path: str       # absolute path to the exe-input csv
    gt_path: str    # absolute path to the matching ground-truth csv (under gt_root)

@dataclass
class TrialRecord:
    group_id: str
    trial_id: str
    path: str
    ok: bool
    duration_s: float
    scored_rows: int
    alignment_ok: bool
    nominal_dt_ms: float
    balanced_accuracy: float | None
    overall_accuracy: float | None
    macro_f1: float | None        # headline metric (mirrors measure.frame_metrics)
    per_class: dict               # name -> {recall, precision, f1, support_rows, support_s}
    transitions: dict
    buckets: dict                 # {counts, fractions, dominant, total_error_rows}
    notes: list
    n_in_gap: int = 0             # count of rows flagged in_gap (data-quality note only)
    confusion: dict = field(default_factory=dict)   # gt_name -> pred_name -> count
    findings: list = field(default_factory=list)    # list[dict], from diagnose.Finding
    nontechnical: dict = field(default_factory=dict)     # nontech.assemble_nontechnical() output
    total_pred_transitions: int = 0                      # nontechnical's FALSE_TRANSITION shared denominator

@dataclass
class CorpusResult:
    records: list
    failures: list                              # list[{group_id, trial_id, path, error}]
    skipped: list = field(default_factory=list)  # list[{path, reason}] -- csvs that aren't annotated trials

def _sniff_columns(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return [c.strip() for c in df.columns]

def discover_trials(input_root: str, gt_root: str, *, exclude: str | None = None) -> tuple[list[Trial], list[dict]]:
    input_root_path = Path(input_root).resolve()
    gt_root_path = Path(gt_root).resolve()
    exclude_path = Path(exclude).resolve() if exclude else None

    trials: list[Trial] = []
    skipped: list[dict] = []
    for p in sorted(input_root_path.rglob("*.csv")):
        p = p.resolve()
        if exclude_path is not None and (p == exclude_path or exclude_path in p.parents):
            continue
        rel = p.relative_to(input_root_path)
        trial_id = rel.as_posix()
        group_id = "" if rel.parent == Path(".") else rel.parent.as_posix()
        try:
            cols = _sniff_columns(p)
        except Exception as e:
            skipped.append({"path": trial_id, "reason": f"unreadable csv: {e}"})
            continue
        if _find_col(cols, TIME_CANDS) is None:
            skipped.append({"path": trial_id, "reason": "no recognizable time column"})
            continue

        gt_p = gt_root_path / rel
        if not gt_p.exists():
            skipped.append({"path": trial_id, "reason": f"no matching ground-truth csv found at {gt_p}"})
            continue
        try:
            gt_cols = _sniff_columns(gt_p)
        except Exception as e:
            skipped.append({"path": trial_id, "reason": f"unreadable ground-truth csv: {e}"})
            continue
        if _find_col(gt_cols, LABEL_CANDS) is None:
            skipped.append({"path": trial_id, "reason": "matching ground-truth csv has no recognizable label column"})
            continue

        trials.append(Trial(group_id=group_id, trial_id=trial_id, path=str(p), gt_path=str(gt_p)))

    trials.sort(key=lambda t: (t.group_id, t.trial_id))
    return trials, skipped

# run the evaluation pipeline on every discovered (input, gt) trial pair
def run_corpus(input_root: str, gt_root: str, exe_path: str, *, timeout: float | None = None,
               keep_dir: str | None = None, report_dir: str | None = None,
               late_threshold_ms: float = DEFAULT_LATE_THRESHOLD_MS, progress: bool = True) -> CorpusResult:
    trials, skipped = discover_trials(input_root, gt_root, exclude=keep_dir)
    records: list[TrialRecord] = []
    failures: list[dict] = []

    for t in trials:
        if progress:
            print(f"  {t.trial_id}")
        out_csv: str | None = None
        try:
            if keep_dir:
                dest_dir = os.path.join(keep_dir, t.group_id) if t.group_id else keep_dir
                os.makedirs(dest_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(t.trial_id))[0]
                dest = os.path.join(dest_dir, f"output_{base}.csv")
            else:
                dest = None
            out_csv, _ = run_algorithm(exe_path, t.path, dest, timeout=timeout)

            a = load_aligned(t.path, out_csv, gt_csv=t.gt_path)
            rep = assemble(a)
            nontech_rep = assemble_nontechnical(a, late_threshold_ms=late_threshold_ms)

            if report_dir:
                report_dest_dir = os.path.join(report_dir, t.group_id) if t.group_id else report_dir
                base = os.path.splitext(os.path.basename(t.trial_id))[0]
                write_reports(rep, nontech_rep, report_dest_dir, base)

            records.append(_summarize(rep, nontech_rep, t.group_id, t.trial_id, t.path))

        except Exception as e:
            failures.append({"group_id": t.group_id, "trial_id": t.trial_id, "path": t.path, "error": str(e)})

        finally:
            if not keep_dir and out_csv and os.path.exists(out_csv):
                try:
                    os.remove(out_csv)
                except OSError:
                    pass

    return CorpusResult(records=records, failures=failures, skipped=skipped)


def _summarize(rep: dict, nontech_rep: dict, group_id: str, trial_id: str, csv_path: str) -> TrialRecord:
    h = rep["health"]
    fm = rep["frame_metrics"]
    tt = rep["transition_timing"]
    eb = rep["error_buckets"]
    per_class = {
        name: {
            "recall": st["recall"],
            "precision": st["precision"],
            "f1": st["f1"],
            "support_rows": st["support_rows"],
            "support_s": st["support_s"],
        }
        for name, st in fm["per_class"].items()
    }
    transitions = {
        g["transition"]: {
            "late_offsets_ms": g["late_offsets_ms"],
            "early_offsets_ms": g["early_offsets_ms"],
            "matched": g["matched"],
            "missed": g["missed"],
            "full_segment_miss": g["full_segment_miss"],
            "sustained_early": g["sustained_early"],
            "sustained_late": g["sustained_late"],
        }
        for g in tt
    }
    buckets = {
        "counts": dict(eb["counts"]),
        "fractions": dict(eb["fractions"]),
        "dominant": eb["dominant"],
        "total_error_rows": eb["total_error_rows"],
    }
    return TrialRecord(
        group_id = group_id,
        trial_id = trial_id,
        path = csv_path,
        ok = True,
        duration_s = h["duration_s"],
        scored_rows = fm["scored_rows"],
        alignment_ok = h["alignment_ok"],
        nominal_dt_ms = h["nominal_dt_ms"],
        balanced_accuracy = fm["balanced_accuracy"],
        overall_accuracy = fm["overall_accuracy"],
        macro_f1 = fm["macro_f1"],
        per_class = per_class,
        transitions = transitions,
        buckets = buckets,
        notes = list(h["notes"]),
        n_in_gap = h["n_gaps"],
        confusion = {t: dict(preds) for t, preds in fm["confusion"].items()},
        findings = list(rep["findings"]),
        nontechnical = nontech_rep,
        total_pred_transitions = nontech_rep.get("total_pred_transitions", 0),
    )

# corpus level summary for all successful trials
def aggregate(result: CorpusResult) -> dict:
    recs = result.records
    fails = result.failures
    skipped = result.skipped
    n_ok = len(recs)

    group_counts: dict[str, int] = {}
    for r in recs:
        group_counts[r.group_id] = group_counts.get(r.group_id, 0) + 1
    for f in fails:
        group_counts[f["group_id"]] = group_counts.get(f["group_id"], 0) + 1
    n_align_ok = sum(1 for r in recs if r.alignment_ok)
    coverage = {
        "n_trials_found": n_ok + len(fails),
        "n_trials_ok": n_ok,
        "n_failures": len(fails),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "groups": {g: group_counts[g] for g in sorted(group_counts)},
        "total_duration_s": round(sum(r.duration_s for r in recs), 2),
        "total_scored_rows": sum(r.scored_rows for r in recs),
        "alignment_pass_rate": _f(n_align_ok / n_ok) if n_ok else None,
        "total_in_gap_rows": sum(r.n_in_gap for r in recs),
        "trials_with_gaps": sum(1 for r in recs if r.n_in_gap),
        "in_gap_by_trial": [{"trial_id": r.trial_id, "n_in_gap": r.n_in_gap}
                            for r in recs],
        "failures": [{"trial_id": f["trial_id"], "error": f["error"][:120]}
                     for f in fails],
    }

    dom_freq = {b: 0 for b in _BUCKETS}
    pooled_counts = {b: 0 for b in _BUCKETS}
    pooled_seconds = {b: 0.0 for b in _BUCKETS}
    bucket_weighted: dict[str, list] = {b: [] for b in _BUCKETS}  # (trial_id, row_count) for concentration
    total_error_pool = 0
    for r in recs:
        d = r.buckets.get("dominant")
        if d in dom_freq:
            dom_freq[d] += 1
        dt_s = (r.nominal_dt_ms or 0.0) / 1000.0
        for b in _BUCKETS:
            c = r.buckets["counts"].get(b, 0)
            pooled_counts[b] += c
            pooled_seconds[b] += c * dt_s
            if c:
                bucket_weighted[b].append((r.trial_id, c))
        total_error_pool += r.buckets.get("total_error_rows", 0) or 0

    modal_bucket = max(dom_freq, key=dom_freq.get) if any(dom_freq.values()) else None
    modal_count = dom_freq.get(modal_bucket, 0) if modal_bucket else 0
    modal_share = modal_count / n_ok if n_ok else None
    consistency = (f"{modal_bucket} dominant in {modal_count}/{n_ok} trials "
                   f"({round(modal_share * 100)}%)") if modal_bucket and modal_share is not None \
        else "no dominant error bucket identified"
    error_source = {
        "dominant_frequency": dom_freq,
        "modal_bucket": modal_bucket,
        "modal_share": _f(modal_share),
        "pooled_fractions": {b: _f(pooled_counts[b] / total_error_pool) if total_error_pool else None
                             for b in _BUCKETS},
        "pooled_seconds": {b: round(pooled_seconds[b], 1) for b in _BUCKETS},
        "consistency": consistency,
        # is a bucket's pooled row share actually corpus-wide, or is it one noisy trial?
        "provenance": {b: _concentration(bucket_weighted[b]) for b in _BUCKETS},
    }

    # latency over succesfful trials
    trans_pool: dict[str, dict] = {}
    for r in recs:
        for key, td in r.transitions.items():
            tp = trans_pool.setdefault(key, {
                "late_offsets": [], "late_trials": [],
                "early_offsets": [], "early_trials": [],
                "matched": 0, "missed": 0,
                "full_segment_miss": 0, "sustained_early": 0, "sustained_late": 0, "n_trials": 0,
            })
            tp["late_offsets"].extend(td["late_offsets_ms"])
            tp["late_trials"].extend([r.trial_id] * len(td["late_offsets_ms"]))
            tp["early_offsets"].extend(td["early_offsets_ms"])
            tp["early_trials"].extend([r.trial_id] * len(td["early_offsets_ms"]))
            tp["matched"] += td["matched"]
            tp["missed"] += td["missed"]
            tp["full_segment_miss"] += td["full_segment_miss"]
            tp["sustained_early"] += td["sustained_early"]
            tp["sustained_late"] += td["sustained_late"]
            tp["n_trials"] += 1

    # evidence for relevance through: for a pooled stat driven by a handful of clean events, show whether
    # those events came from one trial or were spread across the corpus
    def _provenance(trial_tags: list[str]) -> dict:
        if not trial_tags:
            return {"n_trials": 0, "top_trial": None, "top_trial_n": 0, "top_trial_share": None}
        counts: dict[str, int] = {}
        for tg in trial_tags:
            counts[tg] = counts.get(tg, 0) + 1
        top = max(counts, key=counts.get)
        return {
            "n_trials": len(counts),
            "top_trial": top,
            "top_trial_n": counts[top],
            "top_trial_share": _f(counts[top] / len(trial_tags)),
        }

    # late (+) and early (-) are kept separate: a pooled signed median would let a large late
    # lag and a large early lag cancel into a falsely reassuring near-zero figure
    latency: dict[str, dict] = {}
    for key, tp in trans_pool.items():
        late_offs = np.array(tp["late_offsets"], dtype=float)
        early_offs = np.array(tp["early_offsets"], dtype=float)
        total_tr = tp["matched"] + tp["missed"]

        late_stats = _dist(late_offs)
        early_stats = _dist(early_offs, worst_pct=10)

        latency[key] = {
            "n_events": int(late_offs.size + early_offs.size),
            "n_trials_present": tp["n_trials"],
            "n_late": int(late_offs.size),
            "n_early": int(early_offs.size),
            "matched": tp["matched"],
            "missed": tp["missed"],
            "full_segment_miss": tp["full_segment_miss"],
            "sustained_early": tp["sustained_early"],
            "sustained_late": tp["sustained_late"],
            "miss_rate": _f(tp["missed"] / total_tr) if total_tr else None,
            "late": {
                "lag_median_ms": late_stats["median"],
                "lag_mean_ms": late_stats["mean"],
                "lag_std_ms": late_stats["std"],
                "lag_p90_ms": late_stats["p90"],
                "confidence": "low_sample" if late_offs.size < MIN_EVENTS_FOR_STATISTIC else "confirmed",
                "provenance": _provenance(tp["late_trials"]),
            },
            "early": {
                "lag_median_ms": _abs(early_stats["median"]),
                "lag_mean_ms": _abs(early_stats["mean"]),
                "lag_std_ms": _abs(early_stats["std"]),
                "lag_p90_ms": _abs(early_stats["p90"]),
                "confidence": "low_sample" if early_offs.size < MIN_EVENTS_FOR_STATISTIC else "confirmed",
                "provenance": _provenance(tp["early_trials"]),
            },
        }

    # confusion matrix
    corpus_confusion: dict[str, dict[str, int]] = {}
    for r in recs:
        for t_true, row in r.confusion.items():
            crow = corpus_confusion.setdefault(t_true, {})
            for t_pred, c in row.items():
                crow[t_pred] = crow.get(t_pred, 0) + c
    confusion_stats = _confusion_stats(corpus_confusion)

    # accuracy
    class_pool: dict[str, dict] = {}
    class_error_weighted: dict[str, list] = {}  # (trial_id, error_rows)
    for r in recs:
        for name, st in r.per_class.items():
            cp = class_pool.setdefault(name, {"rw": [], "recalls": [], "precisions": [], "f1s": [],
                                              "support_s": 0.0, "support_rows": 0, "n_trials": 0})
            recall = st.get("recall")
            prec = st.get("precision")
            f1v = st.get("f1")
            support = st.get("support_rows", 0) or 0
            if recall is not None and support > 0:
                cp["rw"].append((recall, support))
                cp["recalls"].append(recall)
            if prec is not None:
                cp["precisions"].append(prec)
            if f1v is not None:
                cp["f1s"].append(f1v)
            cp["support_s"] += st.get("support_s", 0.0) or 0.0
            cp["support_rows"] += support
            cp["n_trials"] += 1
            if support > 0 and r.confusion:
                tp = r.confusion.get(name, {}).get(name, 0)
                class_error_weighted.setdefault(name, []).append((r.trial_id, support - tp))

    class_accuracy: dict[str, dict] = {}
    corpus_recalls: list[float] = []   # per-class pooled recalls; UNKNOWN + pred-only excluded
    for name, cp in class_pool.items():
        gt_present = cp["support_rows"] > 0
        if cp["rw"]:
            tw = sum(s for _, s in cp["rw"])
            pooled_recall = sum(rl * s for rl, s in cp["rw"]) / tw if tw else None
            mean_recall = statistics.mean(cp["recalls"])
            std_recall = statistics.stdev(cp["recalls"]) if len(cp["recalls"]) > 1 else 0.0
        else:
            pooled_recall = mean_recall = std_recall = None
        if gt_present and name != "UNKNOWN" and pooled_recall is not None:
            corpus_recalls.append(pooled_recall)
        mean_prec = statistics.mean(cp["precisions"]) if cp["precisions"] else None
        std_prec = statistics.stdev(cp["precisions"]) if len(cp["precisions"]) > 1 else \
            (0.0 if cp["precisions"] else None)
        mean_f1 = statistics.mean(cp["f1s"]) if cp["f1s"] else None
        std_f1 = statistics.stdev(cp["f1s"]) if len(cp["f1s"]) > 1 else (0.0 if cp["f1s"] else None)
        cs = confusion_stats.get(name, {})
        class_accuracy[name] = {
            "gt_present": gt_present,             # False -> prediction-only in THIS corpus
            "pooled_recall": _f(pooled_recall),
            "pooled_precision": cs.get("pooled_precision"),  # exact, from summed confusion counts
            "pooled_f1": cs.get("pooled_f1"),                # exact, from summed confusion counts
            "mean_recall": _f(mean_recall),
            "std_recall": _f(std_recall),
            "mean_precision": _f(mean_prec),
            "std_precision": _f(std_prec),
            "mean_f1": _f(mean_f1),
            "std_f1": _f(std_f1),
            "n_trials_present": cp["n_trials"],
            "total_support_s": round(cp["support_s"], 2),
            # does one trial account for most of this class's errors, or is it spread out?
            "error_concentration": _concentration(class_error_weighted.get(name, [])),
        }

    ba_vals = [r.balanced_accuracy for r in recs if r.balanced_accuracy is not None]
    mf1_vals = [r.macro_f1 for r in recs if r.macro_f1 is not None]
    accuracy = {
        "per_class": class_accuracy,
        # headline, mirrors the per-trial headline in measure.frame_metrics
        "macro_f1": {
            "mean": _f(statistics.mean(mf1_vals)) if mf1_vals else None,
            "std": _f(statistics.stdev(mf1_vals)) if len(mf1_vals) > 1 else (_f(0.0) if mf1_vals else None),
            "min": _f(min(mf1_vals)) if mf1_vals else None,
            "max": _f(max(mf1_vals)) if mf1_vals else None,
        },
        "balanced_accuracy": {
            "mean": _f(statistics.mean(ba_vals)) if ba_vals else None,
            "std": _f(statistics.stdev(ba_vals)) if len(ba_vals) > 1 else (_f(0.0) if ba_vals else None),
            "min": _f(min(ba_vals)) if ba_vals else None,
            "max": _f(max(ba_vals)) if ba_vals else None,
        },
        # same definition as the per-trial balanced accuracy, but each class's recall is
        # pooled (support-weighted) across the whole corpus before averaging
        "corpus_balanced_accuracy": _f(statistics.mean(corpus_recalls))
                                    if corpus_recalls else None,
    }

    # by group (folder-derived; may be a single group if the corpus is flat)
    group_groups: dict[str, list[TrialRecord]] = {}
    for r in recs:
        group_groups.setdefault(r.group_id, []).append(r)
    group_stats: dict[str, dict] = {}
    for g, group in group_groups.items():
        bas = [r.balanced_accuracy for r in group if r.balanced_accuracy is not None]
        mf1s = [r.macro_f1 for r in group if r.macro_f1 is not None]
        dom_counts: dict[str, int] = {}
        for r in group:
            d = r.buckets.get("dominant")
            if d:
                dom_counts[d] = dom_counts.get(d, 0) + 1
        group_late: dict[str, list] = {}
        group_early: dict[str, list] = {}
        for r in group:
            for key, td in r.transitions.items():
                if td["late_offsets_ms"]:
                    group_late.setdefault(key, []).extend(td["late_offsets_ms"])
                if td["early_offsets_ms"]:
                    group_early.setdefault(key, []).extend(td["early_offsets_ms"])
        group_stats[g] = {
            "n_trials": len(group),
            "mean_macro_f1": _f(statistics.mean(mf1s)) if mf1s else None,
            "mean_balanced_accuracy": _f(statistics.mean(bas)) if bas else None,
            "dominant_bucket": max(dom_counts, key=dom_counts.get) if dom_counts else None,
            "mean_late_lag_per_transition": {k: _f(float(np.mean(v))) for k, v in group_late.items()},
            "mean_early_lag_per_transition": {k: _abs(_f(float(np.mean(v)))) for k, v in group_early.items()},
        }

    # groups with zero successful trials still need to rank (as worst) rather than vanish
    fail_counts: dict[str, int] = {}
    for f in fails:
        fail_counts[f["group_id"]] = fail_counts.get(f["group_id"], 0) + 1
    for g, cnt in fail_counts.items():
        if g not in group_stats:
            group_stats[g] = {
                "n_trials": cnt,
                "mean_macro_f1": None,
                "mean_balanced_accuracy": None,
                "dominant_bucket": None,
                "mean_late_lag_per_transition": {},
                "mean_early_lag_per_transition": {},
            }

    # ranked by macro-F1 (the headline metric, consistent with the per-trial headline)
    ranked = sorted(group_stats.items(),
                    key=lambda kv: kv[1]["mean_macro_f1"]
                    if kv[1]["mean_macro_f1"] is not None else -1.0,
                    reverse=True)
    ranking = [{"group_id": g, "mean_macro_f1": st["mean_macro_f1"],
                "mean_balanced_accuracy": st["mean_balanced_accuracy"]}
               for g, st in ranked]

    # best/worst callouts — only meaningful with >1 candidate; GT-present classes only
    gt_classes = {n: c for n, c in class_accuracy.items()
                  if c["gt_present"] and c["pooled_recall"] is not None}
    by_group = {
        "per_group": group_stats,
        "ranking": ranking,
        "best_group": ranked[0][0] if ranked else None,
        "worst_group": ranked[-1][0] if len(ranked) > 1 else None,
        "best_transition": (min(latency, key=lambda k: latency[k]["miss_rate"]
                                if latency[k]["miss_rate"] is not None else 1.0)
                            if len(latency) > 1 else None),
        "worst_transition": (max(latency, key=lambda k: latency[k]["miss_rate"]
                                 if latency[k]["miss_rate"] is not None else -1.0)
                             if len(latency) > 1 else None),
        "best_class": (max(gt_classes, key=lambda n: gt_classes[n]["pooled_recall"])
                       if len(gt_classes) > 1 else None),
        "worst_class": (min(gt_classes, key=lambda n: gt_classes[n]["pooled_recall"])
                        if len(gt_classes) > 1 else None),
    }

    # worst / best trials
    ranked_trials = sorted(
        ({"trial_id": r.trial_id, "group_id": r.group_id, "macro_f1": r.macro_f1,
          "balanced_accuracy": r.balanced_accuracy, "dominant_bucket": r.buckets.get("dominant")}
         for r in recs if r.macro_f1 is not None),
        key=lambda d: d["macro_f1"],
    )
    trial_ranking = {
        "worst": ranked_trials[:10],
        "best": list(reversed(ranked_trials[-10:])) if len(ranked_trials) > 1 else [],
    }

    # findings roll up
    all_findings = [{"trial_id": r.trial_id, "group_id": r.group_id, **f} for r in recs for f in r.findings]
    sev_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for f in all_findings:
        sev_counts[f["severity"]] = sev_counts.get(f["severity"], 0) + 1
        kind_counts[f["kind"]] = kind_counts.get(f["kind"], 0) + 1
    findings_summary = {
        "total": len(all_findings),
        "by_severity": sev_counts,
        "by_kind": kind_counts,
        "critical": [f for f in all_findings if f["severity"] == "critical"],
        "top_kinds": sorted(({"kind": k, "count": c} for k, c in kind_counts.items()),
                             key=lambda d: d["count"], reverse=True)[:8],
    }

    return {
        "coverage": coverage,
        "error_source": error_source,
        "confusion": corpus_confusion,
        "latency": latency,
        "accuracy": accuracy,
        "by_group": by_group,
        "trials": trial_ranking,
        "findings": findings_summary,
    }


def aggregate_nontechnical(result: CorpusResult) -> dict:
    recs = result.records

    # (a) composite
    # scored frame across every trial that produced a valid accuracy
    included = [r for r in recs if r.overall_accuracy is not None and r.scored_rows]
    total_frames = sum(r.scored_rows for r in recs)
    included_frames = sum(r.scored_rows for r in included)
    total_correct = sum(r.overall_accuracy * r.scored_rows for r in included)
    frame_error_rate = _f(1.0 - total_correct / included_frames) if included_frames else None

    # (b) event sub-score
    mis_count = 0
    mis_denom = 0
    late_threshold_ms = DEFAULT_LATE_THRESHOLD_MS
    for r in recs:
        ts = r.nontechnical.get("sub_scores", {}).get("transition_mishandled")
        if ts:
            mis_count += ts.get("count", 0)
            mis_denom += ts.get("denominator", 0)
            late_threshold_ms = ts.get("late_threshold_ms", late_threshold_ms)
    sub_scores = {
        "transition_mishandled": {
            "count": mis_count,
            "denominator": mis_denom,
            "denominator_kind": "transitions",
            "fraction": _f(mis_count / mis_denom) if mis_denom else None,
            "late_threshold_ms": late_threshold_ms,
        }
    }

    # (c1) transition_pairs: pool total/missed/late per (from,to) directly (not via a
    # generic count/denominator adder) -- on_time is recomputed after pooling so
    # missed+late+on_time == total holds at the corpus level too, same as per-trial.
    pair_pool: dict[tuple[str, str], dict] = {}
    for r in recs:
        for e in r.nontechnical.get("transition_pairs", []):
            key = (e["from_state"], e["to_state"])
            pb = pair_pool.setdefault(key, {"total": 0, "missed": 0, "late": 0, "translated": e["translated"]})
            pb["total"] += e["total"]
            pb["missed"] += e["missed"]
            pb["late"] += e["late"]

    transition_pairs = []
    for (frm, to), pb in pair_pool.items():
        transition_pairs.append({
            "from_state": frm, "to_state": to,
            "total": pb["total"], "missed": pb["missed"], "late": pb["late"],
            "on_time": pb["total"] - pb["missed"] - pb["late"],
            "translated": pb["translated"],
        })

    # (c2) false_transitions: denominator is the corpus-wide predicted-transition count --
    # identical on every row, not specific to the (from,to) pair.
    total_pred_transitions = sum(r.total_pred_transitions for r in recs)
    false_pool: dict[tuple[str, str], dict] = {}
    for r in recs:
        for e in r.nontechnical.get("false_transitions", []):
            key = (e["from_state"], e["to_state"])
            fb = false_pool.setdefault(key, {"count": 0, "translated": e["translated"]})
            fb["count"] += e["count"]

    false_transitions = [
        {
            "from_state": frm, "to_state": to,
            "count": fb["count"], "denominator": total_pred_transitions,
            "probability": _f(fb["count"] / total_pred_transitions) if total_pred_transitions else None,
            "translated": fb["translated"],
        }
        for (frm, to), fb in false_pool.items()
    ]

    # (c3) steady_state_errors: denominator is that from_state class's pooled frame support.
    support_rows_by_class: dict[str, int] = {}
    for r in recs:
        for name, st in r.per_class.items():
            support_rows_by_class[name] = support_rows_by_class.get(name, 0) + (st.get("support_rows") or 0)

    steady_pool: dict[tuple[str, str, str], dict] = {}
    for r in recs:
        for e in r.nontechnical.get("steady_state_errors", []):
            key = (e["bucket_id"], e["from_state"], e["to_state"])
            sb = steady_pool.setdefault(key, {"count": 0, "translated": e["translated"]})
            sb["count"] += e["count"]

    steady_state_errors = []
    for (bucket_id, frm, to), sb in steady_pool.items():
        denom = support_rows_by_class.get(frm, 0)
        steady_state_errors.append({
            "bucket_id": bucket_id, "from_state": frm, "to_state": to,
            "count": sb["count"], "denominator": denom,
            "probability": _f(sb["count"] / denom) if denom else None,
            "translated": sb["translated"],
        })

    return {
        "frame_error_rate": frame_error_rate,
        "total_frames": total_frames,
        "sub_scores": sub_scores,
        "transition_pairs": transition_pairs,
        "false_transitions": false_transitions,
        "steady_state_errors": steady_state_errors,
    }


def print_fullreport(agg: dict) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    line = "─" * 64
    print(line)
    print("CORPUS OVERVIEW — algorithm behavior across all trials")
    print(line)

    cov, es, lat = agg["coverage"], agg["error_source"], agg["latency"]
    acc, byg = agg["accuracy"], agg["by_group"]

    # headline: pooled late lag medians (detection lag); early events are called out separately
    lag_meds = [v["late"]["lag_median_ms"] for v in lat.values()
                if v["late"]["lag_median_ms"] is not None]
    typical_lag = float(np.median(lag_meds)) if lag_meds else None
    n_early_total = sum(v["n_early"] for v in lat.values())
    mean_mf1 = acc["macro_f1"].get("mean")
    mean_ba = acc["balanced_accuracy"].get("mean")

    if typical_lag is not None:
        lag_clause = f"typical detection lag ~{typical_lag:.0f} ms"
    else:
        lag_clause = "no clean transition events measured"
    prem = f"; premature/over-prediction in {n_early_total} event(s)" if n_early_total else ""
    mf1_str = f"{mean_mf1 * 100:.1f}%" if mean_mf1 is not None else "n/a"
    ba_str = f"{mean_ba * 100:.1f}%" if mean_ba is not None else "n/a"

    print()
    print(f"Across {cov['n_trials_ok']}/{cov['n_trials_found']} trials mean macro-F1 {mf1_str} "
          f"(balanced accuracy {ba_str}); dominant error source is "
          f"'{es.get('modal_bucket') or 'unknown'}' ({es.get('consistency','')}). "
              f"{lag_clause}{prem}. ")

    # COVERAGE
    print("\nCOVERAGE")
    print(f"  trials found={cov['n_trials_found']}  ok={cov['n_trials_ok']}  "
          f"failed={cov['n_failures']}  "
          f"alignment_pass={_pct(cov['alignment_pass_rate'])}")
    print(f"  total duration={cov['total_duration_s']:.1f}s  total scored rows={cov['total_scored_rows']}")
    if len(cov["groups"]) > 1:
        print("  groups: " + ", ".join(f"{g or '(ungrouped)'}({c})" for g, c in cov["groups"].items()))
    print(f"  total in_gap rows={cov['total_in_gap_rows']}  "
          f"trials with gaps={cov['trials_with_gaps']}/{cov['n_trials_ok']}"
          + ("  (per-trial breakdown: see --csv)" if cov["trials_with_gaps"] else ""))
    if cov["n_skipped"]:
        print(f"  skipped {cov['n_skipped']} csv file(s) not recognized as annotated trials "
              f"(see SKIPPED below)")
    # ERROR SOURCE
    CONCENTRATION_THRESHOLD = 0.5  # a stat can be dragged around by a single trial without ever going "low_sample"
    print("\nERROR SOURCE")
    print("  dominant per trial: " + "  ".join(f"{b}={es['dominant_frequency'][b]}" for b in _BUCKETS))
    pf = es["pooled_fractions"]
    print("  pooled row fracs:   " + "  ".join(f"{b}={_pct(pf.get(b))}" for b in _BUCKETS))
    ps = es["pooled_seconds"]
    print("  pooled seconds:     " + "  ".join(f"{b}={ps.get(b, 0):.0f}s" for b in _BUCKETS))
    print(f"  {es['consistency']}")
    for b in _BUCKETS:
        prov = es["provenance"].get(b, {})
        share = prov.get("top_trial_share")
        # n_trials <= 1 means the bucket only ever occurred in one trial to begin with --
        # share is trivially 100% there, not a concentration finding
        if share is not None and share >= CONCENTRATION_THRESHOLD and prov.get("n_trials", 0) > 1:
            print(f"      note: {share*100:.0f}% of '{b}' rows come from {prov['top_trial']} "
                  f"— {prov['n_trials']} trial(s) contributed to this bucket in total")

    # LATENCY — late and early medians are reported separately so a transition that's
    # sometimes very late and sometimes very early doesn't average out to a false "on time"
    print("\nLATENCY  (late = detected after gt transition; early = predicted prematurely)")
    print(f"  {'Transition':<26} {'n':>4} {'lateMed':>8} {'lateP90':>8} {'earlyMed':>9} "
          f"{'earlyP90':>9} {'late':>5} {'early':>5}")
    print(f"  {'-'*26} {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*5} {'-'*5}")
    for key, td in lat.items():
        late_ok = td["late"]["confidence"] != "low_sample"
        early_ok = td["early"]["confidence"] != "low_sample"
        late_med = _ms(td["late"]["lag_median_ms"]) if late_ok else "n/a"
        late_p90 = _ms(td["late"]["lag_p90_ms"]) if late_ok else "n/a"
        early_med = _ms(td["early"]["lag_median_ms"]) if early_ok else "n/a"
        early_p90 = _ms(td["early"]["lag_p90_ms"]) if early_ok else "n/a"
        print(f"  {key:<26} {td['n_events']:>4} {late_med:>8} "
              f"{late_p90:>8} {early_med:>9} "
              f"{early_p90:>9} {td['n_late']:>5} {td['n_early']:>5} "
            )
        for side in ("late", "early"):
            prov = td[side]["provenance"]
            share = prov["top_trial_share"]
            if prov["n_trials"] and share is not None and share >= CONCENTRATION_THRESHOLD:
                print(f"      {side}: {prov['top_trial_n']}/{td[f'n_{side}']} clean event(s) from "
                      f"{prov['top_trial']} ({share*100:.0f}%) — {prov['n_trials']} trial(s) "
                      f"contributed to this side in total")
    if any(td["late"]["confidence"] == "low_sample" or td["early"]["confidence"] == "low_sample"
           for td in lat.values()):
        print(f"  n/a: fewer than {MIN_EVENTS_FOR_STATISTIC} clean events for that side")

    # ACCURACY
    print("\nACCURACY")
    mf1 = acc["macro_f1"]
    print(f"  macro-F1 (headline): mean={_pct(mf1.get('mean'))}  std={_pct(mf1.get('std'))}  "
          f"min={_pct(mf1.get('min'))}  max={_pct(mf1.get('max'))}")
    ba = acc["balanced_accuracy"]
    print(f"  balanced accuracy:   mean={_pct(ba.get('mean'))}  std={_pct(ba.get('std'))}  "
          f"min={_pct(ba.get('min'))}  max={_pct(ba.get('max'))}")
    print(f"  corpus balanced accuracy (per-class pooled recall; UNKNOWN/pred-only excluded): "
          f"{_pct(acc.get('corpus_balanced_accuracy'))}")
    print(f"  {'Class':<16} {'pooled_R':>9} {'pooled_P':>9} {'pooled_F1':>10} {'trials':>7} {'total_s':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*10} {'-'*7} {'-'*8}")
    INSTABILITY_STD = 0.15  # trial-to-trial recall/F1 swing this big is worth flagging as unstable
    for name, cd in acc["per_class"].items():
        tag = "" if cd["gt_present"] else "  (pred-only)"
        print(f"  {name:<16} {_pct(cd['pooled_recall']):>9} {_pct(cd.get('pooled_precision')):>9} "
              f"{_pct(cd.get('pooled_f1')):>10} {cd['n_trials_present']:>7} "
              f"{cd['total_support_s']:>7.1f}s{tag}")
        std_r, std_f1 = cd.get("std_recall"), cd.get("std_f1")
        ec = cd.get("error_concentration") or {}
        share = ec.get("top_trial_share")
        unstable = (std_r is not None and std_r >= INSTABILITY_STD) or \
                   (std_f1 is not None and std_f1 >= INSTABILITY_STD)
        # concentration is trivially 100% with only one trial contributing -- not a finding
        concentrated = share is not None and share >= CONCENTRATION_THRESHOLD and ec.get("n_trials", 0) > 1
        if unstable or concentrated:
            bits = [f"mean_R={_pct(cd.get('mean_recall'))}±{_pct(std_r)}",
                    f"mean_F1={_pct(cd.get('mean_f1'))}±{_pct(std_f1)}"]
            if concentrated:
                bits.append(f"{share*100:.0f}% of errors from {ec['top_trial']}")
            print(f"      unstable across trials: " + ", ".join(bits))

    # BY GROUP — group is whatever folder structure the corpus happens to use;
    # collapses to a single group (no comparison table) when the corpus is flat
    print("\nBY GROUP")
    if len(byg["per_group"]) > 1:
        print(f"  {'Group':<28} {'n':>4} {'mean_F1':>8} {'mean_BA':>8} {'dominant_err'}")
        print(f"  {'-'*28} {'-'*4} {'-'*8} {'-'*8} {'-'*14}")
        for g, st in byg["per_group"].items():
            label = g or "(ungrouped)"
            print(f"  {label:<28} {st['n_trials']:>4} {_pct(st['mean_macro_f1']):>8} "
                  f"{_pct(st['mean_balanced_accuracy']):>8} {st['dominant_bucket'] or 'n/a'}")
            late_lag = st.get("mean_late_lag_per_transition") or {}
            early_lag = st.get("mean_early_lag_per_transition") or {}
            if late_lag:
                print("      late:  " + ", ".join(f"{k} {v:.0f}ms" for k, v in late_lag.items()))
            if early_lag:
                print("      early: " + ", ".join(f"{k} {v:.0f}ms" for k, v in early_lag.items()))
    else:
        print("  (single group — no cross-group comparison to show)")
    if byg.get("best_class"):
        print(f"  best class: {byg['best_class']}  worst class: {byg['worst_class']}")
    if byg.get("best_transition"):
        print(f"  best transition (fewest missed): {byg['best_transition']}  "
              f"worst: {byg['worst_transition']}")

    # WORST TRIALS
    worst = agg.get("trials", {}).get("worst", [])
    if len(worst) > 1:
        print("\nWORST TRIALS (lowest macro-F1 — start debugging here)")
        for wt in worst[:5]:
            print(f"  {wt['trial_id']}: macro-F1={_pct(wt['macro_f1'])}  "
                  f"BA={_pct(wt['balanced_accuracy'])}  dominant={wt['dominant_bucket'] or 'n/a'}")

    # FINDINGS
    fs = agg.get("findings", {})
    if fs.get("total"):
        print("\nFINDINGS")
        sc = fs["by_severity"]
        print(f"  total={fs['total']}  critical={sc.get('critical', 0)}  "
              f"warn={sc.get('warn', 0)}  info={sc.get('info', 0)}"
              + ("  (full detail: see --csv-dir)" if fs["total"] > len(fs["critical"]) else ""))
        if fs["critical"]:
            print("  critical:")
            for f in fs["critical"][:10]:
                print(f"    {f['trial_id']}: {f['claim']}")
        if fs["top_kinds"]:
            print("  most frequent: " + ", ".join(f"{d['kind']}({d['count']})" for d in fs["top_kinds"]))

    # FAILURES
    if cov["failures"]:
        print("\nFAILURES")
        for f in cov["failures"]:
            print(f"  {f['trial_id']}: {f['error']}")

    # SKIPPED
    if cov.get("skipped"):
        print("\nSKIPPED (csv found but not recognized as an annotated trial)")
        for s in cov["skipped"][:10]:
            print(f"  {s['path']}: {s['reason']}")
        if len(cov["skipped"]) > 10:
            print(f"  ... and {len(cov['skipped']) - 10} more")

    print(line)


def write_outputs(agg: dict, result: CorpusResult, *,
                  json_path: str | None = None, csv_path: str | None = None,
                  csv_dir: str | None = None,
                  nontechnical_agg: dict | None = None, json_nontechnical_path: str | None = None) -> None:
    
    if json_path:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_jsonify(agg), fh, indent=2, ensure_ascii=False)

    if json_nontechnical_path and nontechnical_agg is not None:
        with open(json_nontechnical_path, "w", encoding="utf-8") as fh:
            json.dump(_jsonify(nontechnical_agg), fh, indent=2, ensure_ascii=False)

    if csv_path:
        rows: list[dict] = []
        for r in result.records:
            rows.append({
                "trial_id": r.trial_id,
                "group_id": r.group_id,
                "file_name": os.path.basename(r.path),
                "ok": r.ok,
                "duration_s": r.duration_s,
                "scored_rows": r.scored_rows,
                "alignment_ok": r.alignment_ok,
                "n_in_gap": r.n_in_gap,
                "macro_f1": r.macro_f1,
                "balanced_accuracy": r.balanced_accuracy,
                "overall_accuracy": r.overall_accuracy,
                "dominant_bucket": r.buckets.get("dominant"),
                "total_error_rows": r.buckets.get("total_error_rows"),
                "notes": "; ".join(r.notes),
            })
        for f in result.failures:
            rows.append({"trial_id": f["trial_id"],
                         "group_id": f["group_id"],
                         "file_name": os.path.basename(f["path"]),
                         "ok": False,
                         "error": f["error"]}
                        )
        if rows:
            fieldnames = sorted({k for row in rows for k in row})
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                w = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
                w.writeheader()
                w.writerows(rows)

    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
        _write_csv(os.path.join(csv_dir, "confusion.csv"), _confusion_rows(result))
        _write_csv(os.path.join(csv_dir, "per_class_by_trial.csv"), _class_trial_rows(result))
        _write_csv(os.path.join(csv_dir, "findings.csv"), _findings_rows(result))
        _write_csv(os.path.join(csv_dir, "transition_events.csv"), _transition_event_rows(result))
        _write_csv(os.path.join(csv_dir, "skipped.csv"), result.skipped)


def _write_csv(path: str, rows: list) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def _confusion_rows(result: CorpusResult) -> list:
    return [
        {"trial_id": r.trial_id, "group_id": r.group_id, "gt": t_true, "pred": t_pred, "count": c}
        for r in result.records
        for t_true, row in r.confusion.items()
        for t_pred, c in row.items()
    ]


def _class_trial_rows(result: CorpusResult) -> list:
    return [
        {"trial_id": r.trial_id, "group_id": r.group_id, "class": name,
         "recall": st.get("recall"), "precision": st.get("precision"), "f1": st.get("f1"),
         "support_rows": st.get("support_rows"), "support_s": st.get("support_s")}
        for r in result.records
        for name, st in r.per_class.items()
    ]


def _findings_rows(result: CorpusResult) -> list:
    return [
        {"trial_id": r.trial_id, "group_id": r.group_id, "kind": f["kind"], "category": f["category"],
         "severity": f["severity"], "confidence": f["confidence"], "claim": f["claim"]}
        for r in result.records
        for f in r.findings
    ]


def _transition_event_rows(result: CorpusResult) -> list:
    rows = []
    for r in result.records:
        for transition, td in r.transitions.items():
            for off in td["late_offsets_ms"]:
                rows.append({"trial_id": r.trial_id, "group_id": r.group_id, "transition": transition,
                             "side": "late", "offset_ms": off})
            for off in td["early_offsets_ms"]:
                rows.append({"trial_id": r.trial_id, "group_id": r.group_id, "transition": transition,
                             "side": "early", "offset_ms": off})
    return rows


#helpers
def _concentration(weighted: list) -> dict:
    total = sum(w for _, w in weighted)
    if not weighted or total <= 0:
        return {"n_trials": 0, "top_trial": None, "top_trial_share": None}
    totals: dict = {}
    for tag, w in weighted:
        totals[tag] = totals.get(tag, 0.0) + w
    top = max(totals, key=totals.get)
    return {
        "n_trials": len(totals),
        "top_trial": top,
        "top_trial_share": _f(totals[top] / total),
    }

def _confusion_stats(conf: dict) -> dict:
    classes = set(conf.keys())
    for row in conf.values():
        classes.update(row.keys())
    stats = {}
    for c in classes:
        tp = conf.get(c, {}).get(c, 0)
        support = sum(conf.get(c, {}).values())          # row sum: true label count
        predicted = sum(row.get(c, 0) for row in conf.values())  # column sum: predicted count
        recall = tp / support if support else None
        precision = tp / predicted if predicted else None
        if not support:
            f1 = None
        elif tp == 0:
            f1 = 0.0   # fully-missed class: F1 is 0 by definition, never dropped as None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        stats[c] = {
            "support_rows": support,
            "predicted_rows": predicted,
            "pooled_recall": _f(recall),
            "pooled_precision": _f(precision),
            "pooled_f1": _f(f1),
        }
    return stats


# p90 checks
def _dist(arr: np.ndarray, *, worst_pct: float = 90) -> dict:
    if arr.size == 0:
        return {"mean": None, "std": None, "median": None, "p90": None}
    return {
        "mean": _f(float(np.mean(arr))),
        "std": _f(float(np.std(arr, ddof=1))) if arr.size > 1 else _f(0.0),
        "median": _f(float(np.median(arr))),
        "p90": _f(float(np.percentile(arr, worst_pct))),
    }


def _abs(x: float | None) -> float | None:
    return None if x is None else abs(x)


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 4)
    except (TypeError, ValueError):
        return None


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ms(x) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    return obj
