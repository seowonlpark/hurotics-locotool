#!/usr/bin/env python3
"""
runcorpus.py

Run the locoeval evaluation pipeline over an entire corpus of CSVs and
produce a single comprehensive overview of the algorithm's behavior.

python runcorpus.py <input_root> <gt_root> <algorithm.exe>
                    [--timeout SEC] [--keep-dir DIR] [--report-dir DIR]
                    [--late-threshold-ms MS]
                    [--json out.json] [--json-nontechnical out.json] [--csv per_trial.csv]
                    [--csv-dir breakdowns/] [--quiet]
"""
import argparse
import os
import sys

from locoeval.fullreport import (discover_trials, run_corpus, aggregate, aggregate_nontechnical,
                                 print_fullreport, write_outputs)
from locoeval.nontech import DEFAULT_LATE_THRESHOLD_MS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_root", help="folder of CSVs to feed to the algorithm")
    ap.add_argument("gt_root", help="folder of ground-truth CSVs, matched to input_root by relative path")
    ap.add_argument("exe")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--keep-dir", default=None) # put folder name
    ap.add_argument("--report-dir", default=None, metavar="DIR",
                     help="write each trial's report_technical_<trial>.json / "
                          "report_nontechnical_<trial>.json into DIR (nested by group_id, "
                          "like --keep-dir)")
    ap.add_argument("--late-threshold-ms", type=float, default=DEFAULT_LATE_THRESHOLD_MS,
                     help=f"nontechnical LATE_TRANSITION cutoff in ms (default {DEFAULT_LATE_THRESHOLD_MS:.0f})")
    ap.add_argument("--json", default=None, metavar="FILE",
                     help="technical corpus aggregate (unchanged from before the report split)")
    ap.add_argument("--json-nontechnical", default=None, metavar="FILE",
                     help="nontechnical corpus aggregate")
    ap.add_argument("--csv", default=None, metavar="FILE")
    ap.add_argument("--csv-dir", default=None, metavar="DIR",
                     help="write confusion.csv, per_class_by_trial.csv, findings.csv, "
                          "transition_events.csv, skipped.csv into DIR")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.input_root):
        print(f"ERROR: input root directory not found: {args.input_root!r}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.gt_root):
        print(f"ERROR: ground-truth root directory not found: {args.gt_root!r}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.exe):
        print(f"ERROR: algorithm not found: {args.exe!r}", file=sys.stderr)
        sys.exit(2)

    trials, skipped = discover_trials(args.input_root, args.gt_root, exclude=args.keep_dir)
    if not trials:
        print(f"ERROR: no CSVs recognized as trials under {args.input_root!r} "
              f"(with a match under {args.gt_root!r})", file=sys.stderr)
        if skipped:
            print(f"  ({len(skipped)} csv file(s) found but skipped -- e.g. "
                  f"{skipped[0]['path']}: {skipped[0]['reason']})", file=sys.stderr)
        sys.exit(2)

    print(f"[corpus] found {len(trials)} trial(s)" +
          (f", skipped {len(skipped)} non-conforming csv(s)" if skipped else "") +
          f" — running {os.path.basename(args.exe)}")

    result = run_corpus(
        args.input_root,
        args.gt_root,
        args.exe,
        timeout = args.timeout,
        keep_dir = args.keep_dir,
        report_dir = args.report_dir,
        late_threshold_ms = args.late_threshold_ms,
        progress = not args.quiet,
    )

    if not result.records:
        print(f"ERROR: all {len(result.failures)} trial(s) failed — nothing to aggregate",
              file=sys.stderr)
        for f in result.failures:
            print(f"{f['trial_id']}: {f['error']}", file=sys.stderr)
        sys.exit(1)

    agg = aggregate(result)
    nontechnical_agg = aggregate_nontechnical(result)
    print_fullreport(agg)
    write_outputs(agg, result, json_path=args.json, csv_path=args.csv, csv_dir=args.csv_dir,
                  nontechnical_agg=nontechnical_agg, json_nontechnical_path=args.json_nontechnical)
    sys.exit(0)


if __name__ == "__main__":
    main()
