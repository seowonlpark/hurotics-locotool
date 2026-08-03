"""
diagnose.py

interpretation of error

row buckets are assigned in strict precedence order so every row gets exactly one:
    (1) correct
        > (2) omission: a gt segment pred never reaches, split by what pred was doing instead:
                - swallowed: pred flanks the missed segment with the same label on both sides
                  (a real bout was fully absorbed and left no trace)
                - omission: pred flanks the missed segment with two different labels
                  (pred transitioned, but skipped over the missed class entirely)
                - edge_omission: the missed segment touches a recording boundary, so one flank
                  doesn't exist and swallowed-vs-skipped can't be determined
            > (3) flicker
                > (4) late
                > (5) early
                    > (6) steady_confusion

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import AlignedData, segments, transitions, cname
from .measure import frame_metrics

FLICKER_MAX_MS = 200.0            # pred run shorter than this, flanked by equal others => flicker
MIN_EVENTS_FOR_STATISTIC = 10     # below this, timing is tagged low_sample
LAG_MAX_MS = 1000.0               # beyond is too big of a lag to be considered a lag
WEAK_CLASS_F1 = 0.5               # per-class F1 below this triggers a weak-class finding
SUSTAINED_FRACTION = 0.5          # early/late run that also clears LAG_MAX_MS in absolute terms and
                                    # covers >= this fraction of the adjacent segment's duration is a
                                    # sustained misclassification, not detection jitter

ERROR_BUCKETS = [
    "omission",        # gt segment never reached by pred; pred flanks it with two different labels
    "swallowed",       # gt segment never reached by pred; pred flanks it with the same label both sides
    "edge_omission",   # gt segment never reached by pred; touches a recording boundary, flank missing
    "flicker",         # pred run shorter than FLICKER_MAX_MS, flanked by equal others
    "late",            # pred still shows the old label after a gt transition
    "early",           # pred already shows the new label before a gt transition
    "steady_confusion" # pred stays in a different class for the entire GT segment
]

@dataclass
class Finding:
    kind: str
    category: str
    claim: str
    evidence: dict
    severity: str
    confidence: str

@dataclass
class BucketResult:
    counts: dict
    fractions: dict
    dominant: str | None
    total_error_rows: int
    correct_rows: int

@dataclass
class TransitionResult:
    frm: str
    to: str
    offset_ms: float | None
    spans_gap: bool
    gt_idx: int
    full_segment_miss: bool
    sustained_early: bool
    sustained_late: bool
    late_end_idx: int
    early_start_idx: int
    omission_kind: str | None = None
    # exact pred-side transition index that resolves this gt transition (jj for early,
    # j for late -- NOT always == late_end_idx, which can stop early at a transient
    # intermediate class); None when the transition was missed. Used by nontech.py to
    # tell "the transition that actually happened" apart from any other pred transition
    # boundary nearby (e.g. a transient class passed through en route).
    resolved_pred_idx: int | None = None

def _label_arrays(df):
    return df["gt"].to_numpy(dtype = "float64"), df["pred"].to_numpy(dtype = "float64")

# for each GT transition into class X, read forward until pred catches up to X
def transition_timing(a: AlignedData):
    df = a.df
    n = len(df)

    if n == 0: return [] # if no input rows, no results
    gt, pred = _label_arrays(df)
    t = df["time_ms"].to_numpy(dtype = float)
    gap = df["in_gap"].fillna(False).to_numpy(dtype = bool) # bool array; stores gap info
    gtr = transitions(gt, t) # gt transitions
    results: list[TransitionResult] = []

    # extend by one so they don't truncate last
    last_end_t = t[-1] + (t[-1] - t[-2]) if n > 1 else t[-1]

    # loop over each gt transitions
    for k, tr in enumerate(gtr):
        i = tr["idx"]
        to = tr["to"]
        frm = tr["frm"]

        seg_end = gtr[k+1]["idx"] if k+1 < len(gtr) else n # end of curent gt segment (edge)
        seg_start = gtr[k-1]["idx"] if k > 0 else 0 # beginning of prev gt segment (edge)

        late_end = i # start at transition

        # keep moving while prediction stays in class
        while late_end < seg_end and pred[late_end] == frm: late_end += 1
 
        # find when arrive at target
        j = i
        while j < seg_end and not (pred[j] == to): j += 1
        # prediction reaches target
        if j < seg_end:
            # early prediction; pred already equals target at transition boundary
            if j == i and pred[i] == to:
                # walk backward; find when pred first entered target class
                jj = i
                while jj > seg_start and pred[jj-1] == to:
                    jj -= 1

                # offset - pos means late, neg means early
                off = t[jj] - tr["time"]

                # check if in gap
                spans_gap = bool(gap[jj:i+1].any())

                # check if same class whole time
                full_segment_miss = (jj == seg_start)

                # gt segment duration
                seg_dur_ms = t[i] - t[seg_start]

                # check if it was large, persistent early prediction
                sustained_early = bool(seg_dur_ms > 0
                                        and abs(off) >= LAG_MAX_MS
                                        and abs(off) / seg_dur_ms >= SUSTAINED_FRACTION)
                sustained_late = False  # early so not late
                early_start = jj
            # late arrival
            else:
                # offset (exclude if messy / stray)
                off = (t[j] - tr["time"]) if np.all(pred[i:j] == frm) else None

                # check if in gap
                spans_gap = bool(gap[i:j+1].any())

                full_segment_miss = False
                sustained_early = False
                early_start = i

                # if clean
                if off is not None:
                    seg_end_time = t[seg_end] if seg_end < n else last_end_t
                    seg_dur_ms = seg_end_time - t[i]
                    sustained_late = bool(seg_dur_ms > 0
                                           and abs(off) >= LAG_MAX_MS
                                           and abs(off) / seg_dur_ms >= SUSTAINED_FRACTION)
                else:
                    sustained_late = False # cannot reliably measure

            results.append(TransitionResult(cname(frm),
                                            cname(to),
                                            off,
                                            spans_gap,
                                            gt_idx = i,
                                            full_segment_miss = full_segment_miss,
                                            sustained_early = sustained_early,
                                            sustained_late = sustained_late,
                                            late_end_idx=late_end,
                                            early_start_idx=early_start,
                                            resolved_pred_idx=(jj if (j == i and pred[i] == to) else j))
                            )
        # pred never reaches target
        else:
            spans_gap = bool(gap[i:seg_end].any())

            # check classes surrounding missing segment
            left, right = _flanks(pred, i, seg_end, n)
            if left is None or right is None:
                omission_kind = "edge_omission"
            elif left == right:
                omission_kind = "swallowed"
            else:
                omission_kind = "omission"
            
            results.append(TransitionResult(cname(frm),
                                            cname(to),
                                            None,
                                            spans_gap,
                                            gt_idx = i,
                                            full_segment_miss = False,
                                            sustained_early = False,
                                            sustained_late = False,
                                            late_end_idx = late_end,
                                            early_start_idx = i,
                                            omission_kind = omission_kind)
                            )
    return results


def timing_summary(results):
    groups = {}
    for r in results:
        groups.setdefault((r.frm, r.to), []).append(r)
    out = []
    for (frm, to), rs in groups.items():
        n_match = sum(1 for r in rs if r.omission_kind is None)
        n_miss = sum(1 for r in rs if r.omission_kind is not None)
        n_messy = sum(1 for r in rs if r.omission_kind is None and r.offset_ms is None)
        n_full_seg_miss = sum(1 for r in rs if r.full_segment_miss)

        n_sustained_early = sum(1 for r in rs if r.sustained_early and not r.full_segment_miss)
        n_sustained_late = sum(1 for r in rs if r.sustained_late)

        n_omission = sum(1 for r in rs if r.omission_kind == "omission")
        n_swallowed = sum(1 for r in rs if r.omission_kind == "swallowed")
        n_edge_omission = sum(1 for r in rs if r.omission_kind == "edge_omission")

        clean_offs = [r.offset_ms for r in rs
                      if not r.spans_gap and not r.full_segment_miss
                      and not r.sustained_early and not r.sustained_late and r.offset_ms is not None]
        late_offs = [o for o in clean_offs if o > 0]
        early_offs = [o for o in clean_offs if o < 0]
        low_sample = (len(clean_offs) < MIN_EVENTS_FOR_STATISTIC)
        out.append({
            "transition": f"{frm} -> {to}",
            "events": len(rs), "matched": n_match, "missed": n_miss, "messy": n_messy,
            "missed_omission": n_omission, "missed_swallowed": n_swallowed,
            "missed_edge_omission": n_edge_omission,
            "excluded_gap": sum(1 for r in rs if r.spans_gap),
            "full_segment_miss": n_full_seg_miss,
            "sustained_early": n_sustained_early,
            "sustained_late": n_sustained_late,
            "offsets_ms": [round(o, 1) for o in clean_offs],
            "confidence": "low_sample" if low_sample else "confirmed",
            "late_offsets_ms": [round(o, 1) for o in late_offs],
            "early_offsets_ms": [round(o, 1) for o in early_offs],
            "median_late_ms": round(float(np.median(late_offs)), 1) if late_offs else None,
            "median_early_ms": round(float(np.median(early_offs)), 1) if early_offs else None,
            "late_confidence": "low_sample" if len(late_offs) < MIN_EVENTS_FOR_STATISTIC else "confirmed",
            "early_confidence": "low_sample" if len(early_offs) < MIN_EVENTS_FOR_STATISTIC else "confirmed",
        })
    return out

# values just outside [s, e), or None past either edge of the array
def _flanks(arr, s, e, n):
    return (arr[s-1] if s > 0 else None), (arr[e] if e < n else None)

# assign `label` to the still "unclassified" rows in [s, e)
def _claim(bucket, s, e, label):
    region = bucket[s:e]
    region[region == "unclassified"] = label

def row_buckets(a: AlignedData, *, transition_results: list[TransitionResult] | None = None) -> np.ndarray:
    df = a.df
    n = len(df)

    if n == 0:
        return np.full(0, "unclassified", dtype = object)

    # get label arrays
    gt, pred = _label_arrays(df)
    t = df["time_ms"].to_numpy(dtype = float)

    # initialize bucket array
    bucket = np.full(n, "unclassified", dtype = object)
    # mark correct
    bucket[gt == pred] = "correct"

    if transition_results is None: transition_results = transition_timing(a)

    # (2) omission
    first_end = transition_results[0].gt_idx if transition_results else n # find first gt transition
    if not np.any(pred[0:first_end] == gt[0]): # pred ever matched initial?
        _claim(bucket, 0, first_end, "edge_omission")
    for k, r in enumerate(transition_results):
        if r.omission_kind is not None:
            seg_end = transition_results[k+1].gt_idx if k+1 < len(transition_results) else n
            _claim(bucket, r.gt_idx, seg_end, r.omission_kind)

    # (3) flicker — a pred run shorter than FLICKER_MAX_MS, flanked on both sides by same label
    for seg in segments(pred, t):
        if seg["dur_ms"] < FLICKER_MAX_MS:
            s, e = seg["start"], seg["end"]
            left, right = _flanks(pred, s, e, n)
            if left is not None and right is not None and left == right and left != seg["label"]:
                _claim(bucket, s, e, "flicker")

    # lag -> (4) late / (5) early
    for r in transition_results:
        _claim(bucket, r.gt_idx, r.late_end_idx, "late")
        _claim(bucket, r.early_start_idx, r.gt_idx, "early")

    # (6) steady confusion
    bucket[bucket == "unclassified"] = "steady_confusion"

    return bucket


# assign error buckets to each row in the aligned data
def bucket_errors(a: AlignedData, *, transition_results: list[TransitionResult] | None = None) -> BucketResult:
    n = len(a.df)

    # if empty dataset
    if n == 0:
        zero = {b: 0 for b in ERROR_BUCKETS}
        return BucketResult(counts = zero,
                            fractions = {b: 0.0 for b in ERROR_BUCKETS},
                            dominant = None,
                            total_error_rows = 0,
                            correct_rows = 0)

    if transition_results is None: transition_results = transition_timing(a)
    bucket = row_buckets(a, transition_results = transition_results)

    counts = {b: int(np.sum(bucket == b)) for b in ERROR_BUCKETS} # count bucket membership
    total_err = sum(counts.values()) # all but correct rows
    frac = {b: round(counts[b] / total_err, 3) if total_err else 0.0 for b in ERROR_BUCKETS}
    dominant = max(ERROR_BUCKETS, key = counts.get) if total_err else None

    return BucketResult(
        counts = counts,
        fractions = frac,
        dominant = dominant,
        total_error_rows = total_err,
        correct_rows = int(np.sum(bucket == "correct")),
    )

def _alignment_findings(a: AlignedData) -> list[Finding]:
    if a.health.alignment_ok:
        return []
    return [Finding(
                "alignment",
                "data_quality",
                "Alignment is untrustworthy; downstream numbers are unreliable",
                {"notes": a.health.notes},
                "critical",
                confidence = "confirmed")
            ]

def _bucket_findings(buckets: BucketResult) -> list[Finding]:
    # only flicker/steady_confusion are reported here
    # for omission/swallowed/edge_omission check _timing_findings

    findings: list[Finding] = []

    # dominant error source
    if buckets.dominant:
        findings.append(Finding(
            "dominant_error",
            "overview",
            f"Likeliest source of error: '{buckets.dominant}' "
            f"({buckets.fractions[buckets.dominant]*100:.0f}% of mismatched frames)",
            {"fractions": buckets.fractions},
            "info",
            confidence = "confirmed")
        )

    if buckets.counts.get("flicker", 0) > 0:
        findings.append(Finding(
            "flicker",
            "detection",
            f"Flicker: {buckets.counts['flicker']} row(s) "
            f"({buckets.fractions.get('flicker', 0)*100:.0f}%) — brief pred instability, not a "
            f"real detection failure",
            {"count": buckets.counts["flicker"], "fraction": buckets.fractions.get("flicker")},
            "info",
            confidence = "confirmed")
        )

    if buckets.counts.get("steady_confusion", 0) > 0:
        findings.append(Finding(
            "steady_confusion",
            "detection",
            f"Steady confusion: {buckets.counts['steady_confusion']} row(s) "
            f"({buckets.fractions.get('steady_confusion', 0)*100:.0f}%) — pred stuck in a "
            f"different class for the entire GT segment",
            {"count": buckets.counts["steady_confusion"],
             "fraction": buckets.fractions.get("steady_confusion")},
            "warn",
            confidence = "confirmed")
        )

    return findings

# per-class weak modes: F1 catches both under-detection and over-calling
def _weak_class_findings(fm) -> list[Finding]:
    findings: list[Finding] = []
    for name, st in fm.per_class.items():
        if st["f1"] is not None and st["f1"] < WEAK_CLASS_F1 and st["support_rows"] > 0:
            sev = "warn" if st["support_s"] < 5 else "critical"
            note = "low support — collect more data" if st["support_s"] < 5 else "genuine weakness"
            findings.append(Finding(
                "weak_class",
                "class_performance",
                f"Class {name}: F1 {st['f1']*100:.0f}% over {st['support_s']}s ({note})",
                {"class": name, **st},
                sev,
                confidence = ("low_sample" if st["support_s"] < 5 else "confirmed"))
            )
    return findings

#  per-transition check for omission/swallowed/edge_omission and late/early
def _timing_findings(tsum) -> list[Finding]:
    findings: list[Finding] = []
    for g in tsum:
        L = g["median_late_ms"]
        if L is not None:
            confidence = g["late_confidence"]
            over_max = L > LAG_MAX_MS
            tag = f" [{confidence}]" if confidence == "low_sample" else ""
            over = f" — exceeds {LAG_MAX_MS:.0f} ms threshold" if over_max else ""
            findings.append(Finding(
                "late",
                "timing",
                f"{g['transition']}: median late {L:.0f} ms over {len(g['late_offsets_ms'])} event(s){tag}{over}",
                {**dict(g), "lag_max_exceeded": over_max},
                "warn" if over_max else "info",
                confidence = confidence)
            )
        E = g["median_early_ms"]
        if E is not None:
            confidence = g["early_confidence"]
            tag = f" [{confidence}]" if confidence == "low_sample" else ""
            findings.append(Finding(
                "early",
                "timing",
                f"{g['transition']}: median early {E:+.0f} ms over {len(g['early_offsets_ms'])} event(s){tag}",
                dict(g),
                "info",
                confidence = confidence)
            )
        if g["missed"] > 0:
            parts = []
            if g["missed_omission"]:
                parts.append(f"{g['missed_omission']} omission")
            if g["missed_swallowed"]:
                parts.append(f"{g['missed_swallowed']} swallowed")
            if g["missed_edge_omission"]:
                parts.append(f"{g['missed_edge_omission']} edge")
            breakdown = ", ".join(parts) if parts else "unknown"
            findings.append(Finding(
                "missed_transition",
                "detection",
                f"{g['transition']}: {g['missed']} missed ({breakdown})",
                dict(g),
                "warn",
                confidence = ("low_sample" if g["matched"] + g["missed"] < 5 else "confirmed"))
            )

        if g["messy"] > 0:
            findings.append(Finding(
                "excluded_offset",
                "timing",
                f"{g['transition']}: {g['messy']} messy (third label before match, no offset)",
                {**dict(g), "reason": "messy"},
                "info",
                confidence = g["confidence"])
            )
        if g["excluded_gap"] > 0:
            findings.append(Finding(
                "excluded_offset",
                "timing",
                f"{g['transition']}: {g['excluded_gap']} span a gap (measurement corrupted)",
                {**dict(g), "reason": "gap"},
                "warn",
                confidence = g["confidence"])
            )
        if g["full_segment_miss"] > 0:
            findings.append(Finding(
                "excluded_offset",
                "timing",
                f"{g['transition']}: {g['full_segment_miss']} cover the whole preceding segment "
                f"(already counted as missed)",
                {**dict(g), "reason": "full_segment_miss"},
                "info",
                confidence = g["confidence"])
            )
        if g["sustained_early"] > 0:
            findings.append(Finding(
                "excluded_offset",
                "timing",
                f"{g['transition']}: {g['sustained_early']} sustained (>= "
                f"{SUSTAINED_FRACTION*100:.0f}% of segment) — misclassification, not jitter",
                {**dict(g), "reason": "sustained_early"},
                "warn",
                confidence = g["confidence"])
            )
        if g["sustained_late"] > 0:
            findings.append(Finding(
                "excluded_offset",
                "timing",
                f"{g['transition']}: {g['sustained_late']} sustained (>= "
                f"{SUSTAINED_FRACTION*100:.0f}% of segment) — misclassification, not jitter",
                {**dict(g), "reason": "sustained_late"},
                "warn",
                confidence = g["confidence"])
            )
    return findings


def build_findings(a: AlignedData, *, fm = None, tsum = None, buckets = None) -> list[Finding]:
    fm = fm if fm is not None else frame_metrics(a)
    if tsum is None or buckets is None:
        tr = transition_timing(a)  # shared so tsum/buckets can't compute it two different ways
        tsum = tsum if tsum is not None else timing_summary(tr)
        buckets = buckets if buckets is not None else bucket_errors(a, transition_results=tr)
    return (_alignment_findings(a) + _bucket_findings(buckets)
            + _weak_class_findings(fm) + _timing_findings(tsum))


@dataclass
class Diagnosis:
    buckets: BucketResult       # row-level error classification
    timing: list                # per-transition latency stats (list of dicts, see timing_summary)
    findings: list              # list[Finding], the human-readable rollup of both

def diagnose(a: AlignedData, *, fm = None) -> Diagnosis:
    tr = transition_timing(a)
    tsum = timing_summary(tr)
    buckets = bucket_errors(a, transition_results=tr)
    findings = build_findings(a, fm = fm, tsum = tsum, buckets = buckets)
    return Diagnosis(buckets = buckets, timing = tsum, findings = findings)