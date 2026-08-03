#!/usr/bin/env python3
"""
test_fullreport.py

corpus aggregation self-test (no .exe needed).
"""
import contextlib
import csv
import io
import os
import tempfile
from pathlib import Path

from locoeval.fullreport import (discover_trials, aggregate, aggregate_nontechnical, print_fullreport,
                                 write_outputs, CorpusResult, TrialRecord)
from locoeval.nontech import WRONG_STEADY_STATE

results = []

# helper to record and print a check result
def chk(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))

# helper to create a trial record with minimal info
def _trial(group_id, trial_id, *, ba, per_class, transitions, buckets, dt=10.0, mf1=None,
           confusion=None, findings=None, scored_rows=300, overall_accuracy=None,
           nontechnical=None, total_pred_transitions=0):
    return TrialRecord(group_id=group_id, trial_id=trial_id, path=f"{group_id}/{trial_id}", ok=True,
        duration_s=10.0, scored_rows=scored_rows, alignment_ok=True, nominal_dt_ms=dt, balanced_accuracy=ba,
        overall_accuracy=(ba if overall_accuracy is None else overall_accuracy),
        macro_f1=(ba if mf1 is None else mf1),
        per_class=per_class, transitions=transitions,
        buckets=buckets, notes=[], confusion=confusion or {}, findings=findings or [],
        nontechnical=nontechnical or {}, total_pred_transitions=total_pred_transitions)

# helper to create a bucket summary with counts, fractions, dominant type, and total error rows
def _bucket(counts):
    tot = sum(counts.values())
    return {"counts": counts, "fractions": {k: (v/tot if tot else 0) for k, v in counts.items()},
            "dominant": (max(counts, key=counts.get) if tot else None), "total_error_rows": tot}

# test that discovery recursively finds any input csv with a recognizable time column that
# also has a matching ground-truth csv (recognizable label column) at the same relative path
# under gt_root, regardless of naming/folder convention, and skips anything that doesn't
# sniff as a trial (missing time col, or no matching/valid gt csv)
def test_discovery():
    in_root = tempfile.mkdtemp()
    gt_root = tempfile.mkdtemp()
    for group, trial in [("2026-07-07/P01", "trial_01"), ("2026-07-07/P01", "trial_02"),
                         ("2026-07-07/P02", "trial_01"), ("flatfile", None)]:
        rel = f"{group}.csv" if trial is None else f"{group}/{trial}.csv"
        in_p = Path(in_root, rel); in_p.parent.mkdir(parents=True, exist_ok=True)
        in_p.write_text("Time,Feat\n0,1\n")
        gt_p = Path(gt_root, rel); gt_p.parent.mkdir(parents=True, exist_ok=True)
        gt_p.write_text("Time,Label\n0,0\n")
    junk_dir = Path(in_root, "2026-07-07", "P01")
    (junk_dir / "notes.txt").write_text("x")
    (junk_dir / "no_recognizable_columns.csv").write_text("foo,bar\n1,2\n")
    # input csv with no matching ground-truth csv at all
    (junk_dir / "no_matching_gt.csv").write_text("Time,Feat\n0,1\n")

    trials, skipped = discover_trials(in_root, gt_root)
    ids = sorted((t.group_id, t.trial_id) for t in trials)
    chk("discovery finds real trials regardless of folder depth/naming",
        ids == sorted([
            ("2026-07-07/P01", "2026-07-07/P01/trial_01.csv"),
            ("2026-07-07/P01", "2026-07-07/P01/trial_02.csv"),
            ("2026-07-07/P02", "2026-07-07/P02/trial_01.csv"),
            ("", "flatfile.csv"),
        ]), detail=str(ids))
    chk("non-conforming csv (no time column) skipped, not fatal",
        any(s["path"].endswith("no_recognizable_columns.csv") for s in skipped))
    chk("input csv with no matching gt csv skipped, not fatal",
        any(s["path"].endswith("no_matching_gt.csv") for s in skipped))
    chk(".txt files never considered", not any("notes.txt" in s["path"] for s in skipped))

# test that exclude= (mirrors --keep-dir) keeps previously-written output csvs
# from being re-ingested as trials on a subsequent run
def test_discovery_excludes_keep_dir():
    in_root = tempfile.mkdtemp()
    gt_root = tempfile.mkdtemp()
    p = Path(in_root, "P01"); p.mkdir(parents=True, exist_ok=True)
    (p / "trial_01.csv").write_text("Time,Feat\n0,1\n")
    gp = Path(gt_root, "P01"); gp.mkdir(parents=True, exist_ok=True)
    (gp / "trial_01.csv").write_text("Time,Label\n0,0\n")
    keep_dir = Path(in_root, "_out")
    keep_dir.mkdir(parents=True, exist_ok=True)
    (keep_dir / "output_trial_01.csv").write_text("Time,Prediction\n0,0\n")

    trials, _ = discover_trials(in_root, gt_root, exclude=str(keep_dir))
    chk("excluded keep_dir's csvs are not re-discovered as trials",
        all("_out" not in t.trial_id for t in trials))

# test that aggregation produces expected numbers from hand-computed trial records
def test_aggregation():
    tA = {"STANDING -> WALKING": {"late_offsets_ms": [100.0, 200.0], "early_offsets_ms": [-5000.0],
                                  "matched": 3, "missed": 0, "full_segment_miss": 0,
                                  "sustained_early": 1, "sustained_late": 1}}
    tB = {"STANDING -> WALKING": {"late_offsets_ms": [300.0], "early_offsets_ms": [-4000.0],
                                  "matched": 2, "missed": 1, "full_segment_miss": 0,
                                  "sustained_early": 0, "sustained_late": 0}}
    pcA = {"STANDING": {"recall": 0.6, "precision": 0.9, "support_rows": 100, "support_s": 1.0},
           "WALKING": {"recall": 0.9, "precision": 0.5, "support_rows": 200, "support_s": 2.0},
           "SITTING": {"recall": None, "precision": 0.0, "support_rows": 0, "support_s": 0.0}}
    pcB = {"STANDING": {"recall": 0.8, "precision": 0.95, "support_rows": 300, "support_s": 3.0},
           "WALKING": {"recall": 1.0, "precision": 0.6, "support_rows": 100, "support_s": 1.0}}
    rA = _trial("g1", "g1/trial_01.csv", ba=0.75, mf1=0.70, per_class=pcA, transitions=tA,
                buckets=_bucket({"omission": 20, "flicker": 0, "late": 80, "early": 0, "steady_confusion": 0}))
    rB = _trial("g1", "g1/trial_02.csv", ba=0.90, mf1=0.95, per_class=pcB, transitions=tB,
                buckets=_bucket({"omission": 40, "flicker": 0, "late": 10, "early": 0, "steady_confusion": 0}))
    res = CorpusResult(records=[rA, rB],
                       failures=[{"group_id": "g2", "trial_id": "g2/trial_02.csv",
                                  "path": "g2/trial_02.csv", "error": "file not found"}])
    agg = aggregate(res)

    cov = agg["coverage"]
    chk("aggregation includes both trials", cov["n_trials_ok"] == 2)
    chk("failure surfaced", cov["n_failures"] == 1)

    lat = agg["latency"]["STANDING -> WALKING"]
    chk("late/early counts from sign", lat["n_late"] == 3 and lat["n_early"] == 2)
    chk("late lag median = 200 (100,200,300)", lat["late"]["lag_median_ms"] == 200.0)
    chk("early lag median = 4500 magnitude (-5000,-4000)", lat["early"]["lag_median_ms"] == 4500.0)
    chk("miss rate ~= 1/6", abs(lat["miss_rate"] - 1/6) < 5e-4)
    chk("sustained_early accumulated across trials (1+0=1)", lat["sustained_early"] == 1)
    chk("sustained_late accumulated across trials (1+0=1)", lat["sustained_late"] == 1)

    acc = agg["accuracy"]
    chk("support-weighted pooled recall (STANDING=0.75)", abs(acc["per_class"]["STANDING"]["pooled_recall"] - 0.75) < 1e-9)
    chk("mean precision (WALKING=0.55)", abs(acc["per_class"]["WALKING"]["mean_precision"] - 0.55) < 1e-9)
    chk("pred-only class kept + flagged", acc["per_class"]["SITTING"]["gt_present"] is False)
    exp = (0.75 + (0.9*200 + 1.0*100)/300) / 2   # SITTING (pred-only) excluded
    chk("corpus balanced accuracy excludes pred-only/UNKNOWN",
        abs(acc["corpus_balanced_accuracy"] - exp) < 1e-4)
    chk("macro-F1 mean threaded from trials (mean of 0.70, 0.95)", abs(acc["macro_f1"]["mean"] - 0.825) < 1e-9)

    es = agg["error_source"]
    chk("pooled omission fraction (60/150=0.4)", abs(es["pooled_fractions"]["omission"] - 0.4) < 1e-9)
    chk("omission seconds (60 rows * 10ms = 0.6s)", abs(es["pooled_seconds"]["omission"] - 0.6) < 1e-6)

    chk("single-transition best/worst suppressed", agg["by_group"]["best_transition"] is None)
    chk("worst_group reported when another group exists", agg["by_group"]["worst_group"] == "g2")
    chk("by-group ranking is keyed on macro-F1 (the headline)",
        abs(agg["by_group"]["per_group"]["g1"]["mean_macro_f1"] - 0.825) < 1e-9)
    return agg

# test confusion-matrix pooling, exact pooled precision/F1, error concentration,
# worst-trial ranking, and the findings rollup -- all previously computed by diagnose.py/
# measure.py and then discarded before reaching the corpus report
def test_confusion_findings_and_worst_trials():
    pcA = {"STANDING": {"recall": 0.6, "precision": 1.0, "f1": 0.75, "support_rows": 100, "support_s": 1.0},
           "WALKING": {"recall": 1.0, "precision": 100/140, "f1": 0.8333333, "support_rows": 100, "support_s": 1.0},
           "SITTING": {"recall": 0.0, "precision": None, "f1": 0.0, "support_rows": 20, "support_s": 0.2}}
    pcB = {"STANDING": {"recall": 0.9, "precision": 1.0, "f1": 0.9473684, "support_rows": 100, "support_s": 1.0},
           "WALKING": {"recall": 1.0, "precision": 50/60, "f1": 0.9090909, "support_rows": 50, "support_s": 0.5}}
    confA = {"STANDING": {"STANDING": 60, "WALKING": 40}, "WALKING": {"WALKING": 100},
             "SITTING": {"WALKING": 20}}
    confB = {"STANDING": {"STANDING": 90, "WALKING": 10}, "WALKING": {"WALKING": 50}}
    findA = [{"kind": "alignment", "category": "data_quality", "claim": "Alignment issue",
              "evidence": {}, "severity": "critical", "confidence": "confirmed"},
             {"kind": "weak_class", "category": "class_performance", "claim": "Class SITTING weak",
              "evidence": {}, "severity": "warn", "confidence": "confirmed"}]
    findB = [{"kind": "dominant_error", "category": "overview", "claim": "dominant is omission",
              "evidence": {}, "severity": "info", "confidence": "confirmed"}]
    tA = {"STANDING -> WALKING": {"late_offsets_ms": [50.0], "early_offsets_ms": [],
                                  "matched": 1, "missed": 0, "full_segment_miss": 0,
                                  "sustained_early": 0, "sustained_late": 0}}
    tB = {"STANDING -> WALKING": {"late_offsets_ms": [30.0, 40.0], "early_offsets_ms": [-20.0],
                                  "matched": 2, "missed": 0, "full_segment_miss": 0,
                                  "sustained_early": 0, "sustained_late": 0}}
    bucket = _bucket({"omission": 10, "flicker": 0, "late": 0, "early": 0, "steady_confusion": 0})
    rA = _trial("g5", "g5/trial_01.csv", ba=0.75, mf1=0.70, per_class=pcA, transitions=tA, buckets=bucket,
                confusion=confA, findings=findA)
    rB = _trial("g5", "g5/trial_02.csv", ba=0.90, mf1=0.95, per_class=pcB, transitions=tB, buckets=bucket,
                confusion=confB, findings=findB)
    result = CorpusResult(records=[rA, rB], failures=[])
    agg = aggregate(result)

    acc = agg["accuracy"]["per_class"]
    chk("exact pooled recall from summed confusion (STANDING 150/200=0.75)",
        abs(acc["STANDING"]["pooled_recall"] - 0.75) < 1e-9)
    # predicted-WALKING column also picks up the 20 SITTING rows pred-misclassified as WALKING
    # (150 tp / (50 from STANDING row + 150 from WALKING row + 20 from SITTING row) = 150/220)
    chk("exact pooled precision from summed confusion, incl. cross-class contamination (150/220)",
        abs(acc["WALKING"]["pooled_precision"] - 150/220) < 1e-4)  # values are rounded to 4dp by _f()
    chk("exact pooled F1 from summed confusion (STANDING 6/7)",
        abs(acc["STANDING"]["pooled_f1"] - 6/7) < 1e-4)
    chk("fully-missed class gets pooled_f1=0.0, not None/NaN", acc["SITTING"]["pooled_f1"] == 0.0)
    chk("fully-missed class has no predicted rows -> pooled_precision None",
        acc["SITTING"]["pooled_precision"] is None)
    chk("SITTING's errors are 100% concentrated in one trial",
        acc["SITTING"]["error_concentration"]["top_trial_share"] == 1.0
        and acc["SITTING"]["error_concentration"]["top_trial"] == "g5/trial_01.csv")

    worst = agg["trials"]["worst"]
    chk("worst trial ranked first (lowest macro-F1)", worst[0]["trial_id"] == "g5/trial_01.csv")
    chk("best trial ranked last among worst list", worst[-1]["trial_id"] == "g5/trial_02.csv")

    fs = agg["findings"]
    chk("findings pooled across trials (2 + 1 = 3)", fs["total"] == 3)
    chk("severity counts split correctly", fs["by_severity"] == {"critical": 1, "warn": 1, "info": 1})
    chk("critical finding tagged with its trial", fs["critical"][0]["trial_id"] == "g5/trial_01.csv")
    return result, agg

# test that the csv_dir breakdown CSVs are written with the expected rows
def test_csv_dir_outputs(result):
    d = tempfile.mkdtemp()
    write_outputs(aggregate(result), result, csv_dir=d)

    with open(os.path.join(d, "confusion.csv")) as fh:
        conf_rows = list(csv.DictReader(fh))
    chk("confusion.csv has one row per gt/pred/trial cell",
        len(conf_rows) == sum(len(row) for r in result.records for row in r.confusion.values()))

    with open(os.path.join(d, "per_class_by_trial.csv")) as fh:
        class_rows = list(csv.DictReader(fh))
    chk("per_class_by_trial.csv has one row per (trial, class)",
        len(class_rows) == sum(len(r.per_class) for r in result.records))

    with open(os.path.join(d, "findings.csv")) as fh:
        finding_rows = list(csv.DictReader(fh))
    chk("findings.csv has one row per finding across all trials",
        len(finding_rows) == sum(len(r.findings) for r in result.records))

    with open(os.path.join(d, "transition_events.csv")) as fh:
        event_rows = list(csv.DictReader(fh))
    chk("transition_events.csv has one row per late+early offset",
        len(event_rows) == sum(len(td["late_offsets_ms"]) + len(td["early_offsets_ms"])
                                for r in result.records for td in r.transitions.values()))

# test corpus-level pooling of the nontechnical report (aggregate_nontechnical) --
# specifically that a trial which observed a state pair but had zero errors on it still
# contributes its true denominator to the pooled probability, rather than being silently
# excluded (which would inflate the pooled rate)
def test_nontechnical_aggregation():
    pcA = {"STANDING": {"support_rows": 60}, "WALKING": {"support_rows": 40}}
    pcB = {"STANDING": {"support_rows": 30}, "WALKING": {"support_rows": 20}}
    nontechA = {
        "frame_error_rate": 0.1, "total_frames": 100,
        "sub_scores": {"transition_mishandled": {"count": 1, "denominator": 2,
                                                  "denominator_kind": "transitions",
                                                  "fraction": 0.5, "late_threshold_ms": 1000.0}},
        "transition_pairs": [
            {"from_state": "STANDING", "to_state": "WALKING",
             "total": 2, "missed": 1, "late": 0, "on_time": 1, "translated": True},
        ],
        "false_transitions": [
            {"from_state": "STANDING", "to_state": "CLASS_99",
             "count": 1, "denominator": 3, "probability": round(1/3, 4), "translated": False},
        ],
        "steady_state_errors": [
            {"bucket_id": WRONG_STEADY_STATE, "from_state": "STANDING", "to_state": "WALKING",
             "count": 5, "denominator": 60, "probability": round(5/60, 4), "translated": True},
        ],
        "total_pred_transitions": 3,
    }
    nontechB = {
        "frame_error_rate": 0.2, "total_frames": 50,
        "sub_scores": {"transition_mishandled": {"count": 0, "denominator": 1,
                                                  "denominator_kind": "transitions",
                                                  "fraction": 0.0, "late_threshold_ms": 1000.0}},
        "transition_pairs": [
            # this trial saw the STANDING->WALKING transition too, but caught it clean --
            # its total must still land in the pooled denominator, not just A's
            {"from_state": "STANDING", "to_state": "WALKING",
             "total": 1, "missed": 0, "late": 0, "on_time": 1, "translated": True},
        ],
        "false_transitions": [],
        # no FALSE_TRANSITION entry at all this trial -- but it still had 2 pred
        # transitions, which must count toward the pooled false_transitions denominator
        "steady_state_errors": [
            {"bucket_id": WRONG_STEADY_STATE, "from_state": "STANDING", "to_state": "WALKING",
             "count": 2, "denominator": 30, "probability": round(2/30, 4), "translated": True},
        ],
        "total_pred_transitions": 2,
    }
    rA = _trial("g1", "g1/trial_01.csv", ba=0.9, per_class=pcA, transitions={}, buckets=_bucket({}),
                scored_rows=100, overall_accuracy=0.9, nontechnical=nontechA, total_pred_transitions=3)
    rB = _trial("g1", "g1/trial_02.csv", ba=0.8, per_class=pcB, transitions={}, buckets=_bucket({}),
                scored_rows=50, overall_accuracy=0.8, nontechnical=nontechB, total_pred_transitions=2)
    result = CorpusResult(records=[rA, rB], failures=[])
    nagg = aggregate_nontechnical(result)

    chk("frame_error_rate is scored_rows-weighted across trials (1 - 130/150)",
        abs(nagg["frame_error_rate"] - (1 - 130/150)) < 1e-4, nagg["frame_error_rate"])
    chk("total_frames sums scored_rows across trials", nagg["total_frames"] == 150)

    ts = nagg["sub_scores"]["transition_mishandled"]
    chk("transition_mishandled count/denominator sum exactly across trials",
        ts["count"] == 1 and ts["denominator"] == 3, ts)

    pairs_by_key = {(p["from_state"], p["to_state"]): p for p in nagg["transition_pairs"]}
    p = pairs_by_key[("STANDING", "WALKING")]
    chk("transition_pairs pools total/missed/late across trials (total 2+1=3, missed 1+0=1)",
        p["total"] == 3 and p["missed"] == 1 and p["late"] == 0 and p["on_time"] == 2, p)
    chk("transition_pairs missed+late+on_time == total after pooling (no drift vs. per-trial)",
        p["missed"] + p["late"] + p["on_time"] == p["total"], p)

    ft_by_key = {(f["from_state"], f["to_state"]): f for f in nagg["false_transitions"]}
    ft = ft_by_key[("STANDING", "CLASS_99")]
    chk("false_transitions denominator sums total_pred_transitions from every trial (3+2=5), "
        "including the trial with no entry for this pair",
        ft["count"] == 1 and ft["denominator"] == 5, ft)
    chk("false_transitions entry keeps translated=False for a CLASS_n state", ft["translated"] is False)

    ss_by_key = {(s["bucket_id"], s["from_state"], s["to_state"]): s for s in nagg["steady_state_errors"]}
    wss = ss_by_key[(WRONG_STEADY_STATE, "STANDING", "WALKING")]
    chk("steady_state_errors denominator is pooled gt support_rows (60+30=90), not summed entry denominators",
        wss["count"] == 7 and wss["denominator"] == 90, wss)


# test that aggregation doesn't crash on an empty corpus
def test_degenerate():
    chk("empty corpus doesn't crash", aggregate(CorpusResult([], []))["coverage"]["n_trials_ok"] == 0)
    chk("empty corpus nontechnical aggregation doesn't crash",
        aggregate_nontechnical(CorpusResult([], []))["total_frames"] == 0)

# test that the report printer renders the aggregate without crashing (output suppressed)
def test_print(agg):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            print_fullreport(agg)
        chk("print_fullreport renders without crashing", True)
    except Exception as e:
        chk("print_fullreport renders without crashing", False, repr(e))

if __name__ == "__main__":
    print("FULLREPORT SELFTEST\n")
    test_discovery()
    test_discovery_excludes_keep_dir()
    agg = test_aggregation()
    cf_result, cf_agg = test_confusion_findings_and_worst_trials()
    test_csv_dir_outputs(cf_result)
    test_nontechnical_aggregation()
    test_degenerate()
    test_print(agg)
    test_print(cf_agg)
    print(f"\n{sum(results)}/{len(results)} checks passed.")
    raise SystemExit(0 if all(results) else 1)
