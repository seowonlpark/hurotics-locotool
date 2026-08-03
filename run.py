#!/usr/bin/env python3
"""
run.py

python run.py <ground_truth.csv> <input.csv> <algorithm.exe>

inputs:
ground_truth.csv: ground truth label per row
input.csv: IMU serving as input for algorithm
algorithm.exe: executable algorithm

can add options…
	--keep-output {prediction.csv}		save the algo prediction
	--report-dir {dir}			write report_technical_<input>.json /
						report_nontechnical_<input>.json into dir

"""
import argparse, sys, os

from locoeval import load_aligned, assemble, print_problems, write_reports
from locoeval.nontech import assemble_nontechnical, DEFAULT_LATE_THRESHOLD_MS
from locoeval.runner import run_algorithm

def main():
    # input format
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth_csv")
    ap.add_argument("input_csv")
    ap.add_argument("exe")
    ap.add_argument("--keep-output", default = None)
    ap.add_argument("--report-dir", default = None,
                     help = "write report_technical_<input>.json / report_nontechnical_<input>.json here")
    ap.add_argument("--late-threshold-ms", type = float, default = DEFAULT_LATE_THRESHOLD_MS,
                     help = f"nontechnical LATE_TRANSITION cutoff in ms (default {DEFAULT_LATE_THRESHOLD_MS:.0f})")
    args = ap.parse_args()

    # check if user wants to keep output
    out_csv = args.keep_output
    delete_output = (out_csv is None)

    try:
        print(f"[running algorithm] {os.path.basename(args.exe)} on {os.path.basename(args.input_csv)} ...")
        try:
            out_csv, stdout = run_algorithm(args.exe, args.input_csv, out_csv)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

        cleared = stdout.strip()
        if cleared:
            print(f"algo: {cleared.splitlines()[-1]}")

        # pipeline: load aligned data, assemble report, print problems, exit with alignment_ok status
        try:
            a = load_aligned(args.input_csv, out_csv, gt_csv = args.ground_truth_csv)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        rep = assemble(a)
        print_problems(rep)

        if args.report_dir:
            nontech_rep = assemble_nontechnical(a, late_threshold_ms = args.late_threshold_ms)
            base = os.path.splitext(os.path.basename(args.input_csv))[0]
            write_reports(rep, nontech_rep, args.report_dir, base)

        sys.exit(0 if a.health.alignment_ok else 3) # 0 if clean alignment, 3 if untrustworthy
    finally:
        # delete the algo's output if user didn't ask to keep it
        if delete_output and out_csv and os.path.exists(out_csv):
            os.remove(out_csv)

if __name__ == "__main__":
    main()
