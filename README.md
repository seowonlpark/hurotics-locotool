# locoeval

An evaluation pipeline for a black-box IMU locomotion-classification algorithm.
Given hand-labeled ground truth and the raw IMU input, `locoeval` runs the
algorithm, aligns its predictions against the ground truth by row index,
computes frame-level and transition-level metrics, diagnoses the most likely
source of error, and (optionally) rolls all of that up into a single
corpus-wide report across many trials, however they happen to be grouped
on disk.

---

## Quick start

```bash
python selftest.py                      # unit tests on synthetic fault-injected data
python test_fullreport.py               # unit tests on corpus aggregation (no .exe needed)

python run.py <ground_truth.csv> <input.csv> <algorithm.exe>

python runcorpus.py <input_root> <gt_root> <algorithm.exe> --csv-dir breakdowns
```

`<input_root>` can be **any** folder: every `*.csv` found anywhere underneath
it (any depth, any naming) is treated as a candidate trial and kept if it has
a recognizable time column **and** a CSV exists at the same relative path
under `<gt_root>` with a recognizable label column — anything else is skipped
and reported, not fatal. A trial's containing folder (relative to
`<input_root>`) becomes its **group** for by-group breakdowns; a flat folder
with no subdirectories just collapses into a single group. `<gt_root>` must
mirror `<input_root>`'s relative folder/file layout (same trial, same
relative path) so each input CSV can be matched to its ground truth.
If the same CSVs double as both the algorithm's input and the ground
truth (each file already carries the raw signal columns and the label
column), pass the same folder for both `<input_root>` and `<gt_root>` — e.g.
this repo's `cleaned/annotated/` (older `rev<N>/csv/...` layout — each
`revN/csv` folder is picked up as its own group).

---

## Alignment model

Rows are always joined **by position (row index)** across the input, output,
and ground-truth CSVs — not by timestamp matching. Two corroborating checks
are run on top of that join and surfaced as health flags, but neither one
invalidates the join by itself:

- `time_align_ok` — input/output timestamps agree within `TIME_ALIGN_TOL_MS`
- `feature_echo` — input/output "echoed" feature columns (angle, angular
  velocity) agree within `ECHO_ALIGN_TOL`, when the algorithm happens to echo
  them back

The join **is** invalidated (raises `ValueError`) when:

- row counts differ across the three files
- any ground-truth or predicted label is NaN / unparseable

Gaps in the timeline (`dt_ms > GAP_MAX`) are flagged (`health.n_gaps`,
`health.gap_idx`) but do **not** raise an error — they're a data-quality note
that downstream transition-timing measurements exclude where relevant.

---

## CLI reference

### `run.py` — single trial

```
python run.py <ground_truth.csv> <input.csv> <algorithm.exe>
              [--keep-output prediction.csv] [--report-dir DIR] [--late-threshold-ms MS]
```

| Argument                | Description                                          |
| ----------------------- | ----------------------------------------------------- |
| `ground_truth.csv`      | per-row ground-truth label                             |
| `input.csv`              | IMU data fed to the algorithm                          |
| `algorithm.exe`          | the algorithm executable (invoked as `exe input output`) |
| `--keep-output FILE`     | save the algorithm's raw prediction CSV instead of discarding it |
| `--report-dir DIR`       | write `report_technical_<input>.json` / `report_nontechnical_<input>.json` into `DIR` — see [Output files](#output-files) |
| `--late-threshold-ms MS` | nontechnical `LATE_TRANSITION` cutoff (default `1000`) — see [Nontechnical report](#nontechnical-report-nontechpy) |

Exit codes: `0` clean alignment, `3` alignment untrustworthy (see `PROBLEMS`),
`2` bad input / algorithm failure.

### `runcorpus.py` — full corpus

```
python runcorpus.py <input_root> <gt_root> <algorithm.exe>
                    [--timeout SEC] [--keep-dir DIR] [--report-dir DIR]
                    [--late-threshold-ms MS]
                    [--json out.json] [--json-nontechnical out.json]
                    [--csv per_trial.csv] [--csv-dir breakdowns/] [--quiet]
```

Runs the algorithm once per discovered trial CSV, evaluates each, and prints
one aggregated `CORPUS OVERVIEW`. Each output option is independent and can be
combined — see [Output files](#output-files) below for exactly what each one contains.

| Argument                 | Description                                                              |
| ------------------------- | -------------------------------------------------------------------------- |
| `--keep-dir DIR`          | keep each trial's raw prediction CSV (`output_<trial>.csv`, nested by `group_id`) instead of discarding it |
| `--report-dir DIR`        | write each trial's `report_technical_<trial>.json` / `report_nontechnical_<trial>.json` (nested by `group_id`, same convention as `--keep-dir`) |
| `--late-threshold-ms MS`  | nontechnical `LATE_TRANSITION` cutoff (default `1000`)             |
| `--json FILE`             | technical corpus aggregate — **unchanged** from before the technical/nontechnical report split |
| `--json-nontechnical FILE`| nontechnical corpus aggregate, pooled across every trial           |
| `--csv FILE`              | one row per trial (summary stats)                                          |
| `--csv-dir DIR`           | per-record breakdown CSVs                                                  |

**Backward compatibility:** `--json` writes exactly what it always has (the
technical aggregate); no existing flag changed behavior. `--report-dir` and
`--json-nontechnical` opt-in — omitting them reproduces the
pre-split output exactly.

---

## Output files

### `--csv per_trial.csv` — one row per trial

The coarse summary: enough to sort/filter trials without opening anything
else. One row per successfully-evaluated trial, plus one row per failed trial
(`ok=False`, only `trial_id`/`group_id`/`file_name`/`error` populated).

| Column              | Meaning                                                              |
| -------------------- | ----------------------------------------------------------------------- |
| `trial_id`           | path of the input CSV relative to `<input_root>` — unique per trial   |
| `group_id`           | containing folder of that CSV relative to `<input_root>` (`""` if ungrouped) |
| `file_name`          | source annotated CSV                                                   |
| `ok`                 | `True` if the trial ran and scored; `False` if it failed (see `error`) |
| `duration_s`         | recording length                                                        |
| `scored_rows`        | rows scored (= all rows; every row is scored, in-gap or not)           |
| `alignment_ok`       | `False` means the numbers on this row are untrustworthy — check `notes` |
| `n_in_gap`           | rows flagged as timeline gaps (data-quality note only)                  |
| `macro_f1`           | **headline accuracy** for this trial                                    |
| `balanced_accuracy`  | mean per-class recall (UNKNOWN excluded)                                |
| `overall_accuracy`   | raw fraction correct — imbalance-sensitive, reference only              |
| `dominant_bucket`    | the error bucket responsible for the most mismatched frames             |
| `total_error_rows`   | rows not scored `correct`                                               |
| `notes`              | alignment/health notes (`;`-joined), or the failure error message       |

### `--report-dir DIR` — per-trial technical + nontechnical reports

For every trial, writes two files nested under `DIR/<group_id>/`
(same nesting convention as `--keep-dir`):

- **`report_technical_<trial>.json`** — exactly `assemble()`'s per-trial
  dict (`health`, `frame_metrics`, `transition_timing`, `error_buckets`,
  `in_gap_rows`, `findings`) — the same object `run.py --report-dir` writes
  for a single trial, and the same content `print_problems()` prints from.
- **`report_nontechnical_<trial>.json`** — the nontechnical report; see
  [Nontechnical report](#nontechnical-report-nontechpy) below for its shape.

Both files use the `<trial>` basename (input CSV filename, no extension) —
consistent with `--keep-dir`'s `output_<trial>.csv` naming.

### `--json out.json` — the technical corpus aggregate

The entire `aggregate()` dict as JSON, nested exactly as described in
[Corpus overview](#corpus-overview-fullreportpy) above (`coverage`,
`error_source`, `confusion`, `latency`, `accuracy`, `by_group`, `trials`,
`findings`). Use this when you need the full pooled confusion matrix,
per-transition provenance, or the findings rollup programmatically — the
`--csv`/`--csv-dir` files are flattened views of pieces of this same structure.
This is unchanged from before the technical/nontechnical report split.

### `--json-nontechnical out.json` — the nontechnical corpus aggregate

The `aggregate_nontechnical()` dict as JSON: `frame_error_rate` +
`total_frames` pooled (scored-rows-weighted) across every trial, the
`sub_scores.transition_mishandled` sub-score, and `error_buckets` pooled per
`(bucket_id, from_state, to_state)` — see
[Nontechnical report](#nontechnical-report-nontechpy) for the pooling rules
and why summing per-trial entries naively would under-count some
denominators.

### `--csv-dir DIR` — per-record detail

Five files, each one row per atomic fact — too wide/deep for the trial-summary
CSV or a terminal table. All five share `trial_id`/`group_id` so they can be
joined back to `per_trial.csv`.

**`confusion.csv`** — one row per `(trial_id, gt, pred)` cell of that trial's
confusion matrix (includes explicit zero-count cells, so every trial
contributes a full `K×K` block even when a class never appears):

```
count,gt,group_id,pred,trial_id
2959,STANDING,annotated/rev14/csv,STANDING,annotated/rev14/csv/annotated_loco_rev14_trial_1.csv
11432,STANDING,annotated/rev14/csv,WALKING,annotated/rev14/csv/annotated_loco_rev14_trial_1.csv
832,WALKING,annotated/rev14/csv,STANDING,annotated/rev14/csv/annotated_loco_rev14_trial_1.csv
70439,WALKING,annotated/rev14/csv,WALKING,annotated/rev14/csv/annotated_loco_rev14_trial_1.csv
```
Read as: in that trial, 11,432 rows were truly `STANDING` but predicted
`WALKING`. Sum this file's `count` grouped by `(gt, pred)` across all rows to
reconstruct the corpus-pooled matrix shown earlier.

**`per_class_by_trial.csv`** — one row per `(trial_id, class)`: that class's
precision/recall/F1 and support *in that single trial*. This is the raw
material `aggregate()` pools into `accuracy.per_class` (mean/std across trials,
plus exact pooled recall/precision/F1 from summed confusion counts) — use this
file directly when you want to see *which trial* is dragging a class's number
down instead of just the corpus-wide mean:

```
class,f1,group_id,precision,recall,support_rows,support_s,trial_id
STANDING,0.5862,annotated/rev2/csv,1.0,0.4146,13175,26.35,annotated/rev2/csv/annotated_loco_rev2_trial_1.csv
```
Read as: that trial's class `STANDING` had 13,175 ground-truth rows (26.35s),
recall 41.5%, precision 100%, F1 58.6%. A blank `recall`/`f1` (e.g. a class row
with `support_rows=0`) means that class never occurred as ground truth in that
trial — precision-only or entirely absent.

**`findings.csv`** — one row per `Finding` raised in any trial (the same
objects `print_problems()` prints, flattened and tagged with which trial
raised them):

```
category,claim,confidence,group_id,kind,severity,trial_id
class_performance,Class STANDING: F1 29% over 31.27s (genuine weakness),confirmed,annotated/rev2/csv,weak_class,critical,annotated/rev2/csv/annotated_loco_rev2_trial_3.csv
```
Read as: that trial raised a `critical`-severity, `confirmed`-confidence
`weak_class` finding — `claim` is the human-readable sentence, `kind` is the
machine-readable finding type (`dominant_error`, `weak_class`, `late`, `early`,
`missed_transition`, `excluded_offset`, `flicker`, `steady_confusion`,
`alignment`), `category` groups those kinds (`overview`, `class_performance`,
`timing`, `detection`, `data_quality`). Filter this file to `severity=critical`
for a punch list of genuine, well-supported problems across the whole corpus.

**`transition_events.csv`** — one row per individual *clean* late/early
latency measurement (the raw numbers behind every median/p90 in the `LATENCY`
section and in `aggregate()["latency"]`):

```
group_id,offset_ms,side,transition,trial_id
annotated/rev2/csv,3981.0,late,STANDING -> WALKING,annotated/rev2/csv/annotated_loco_rev2_trial_2.csv
annotated/rev2/csv,-371.0,early,STANDING -> WALKING,annotated/rev2/csv/annotated_loco_rev2_trial_1.csv
```
Read as: in one trial, a `STANDING -> WALKING` transition was detected 3,981 ms
late; in another, one such transition was predicted 371 ms early. `side` is
`late` or `early` (never both — see [Frame metrics](#frame-metrics-measurepy)
above for why they're kept separate); messy, gap-spanning, `full_segment_miss`,
and `sustained_early`/`sustained_late` events are excluded from this file
because they're not clean single-event lag measurements (their counts are
still visible in `findings.csv` / `per_trial.csv`).

**`skipped.csv`** — one row per CSV that was found under `<input_root>` but
not recognized as a trial (no `time`-like column, or no matching CSV with a
recognizable `label`-like column at the same relative path under `<gt_root>`),
with a `path` (relative to `<input_root>`) and a `reason`. Only written when
at least one file was skipped.

---

## Frame metrics (`measure.py`)

`frame_metrics(aligned)` scores **every** row (in-gap or not — alignment
already guarantees no NaN gt/pred) and returns:

- `confusion` — `confusion[true_class][pred_class] = count`
- `per_class` — per class: `precision`, `recall`, `f1`, `support_rows`, `support_s`
- `overall_accuracy` — trace(confusion) / total; **imbalance-sensitive, not the headline**
- `balanced_accuracy` — mean per-class recall, `UNKNOWN` excluded
- `macro_f1` — mean per-class F1, `UNKNOWN` excluded — **the headline metric**

A class with real ground-truth support but zero true positives scores F1 = 0
(never NaN), so a fully-missed class can't silently drop out of macro-F1 or
dodge the weak-class finding. A class with no ground-truth support at all
(prediction-only) stays `None`/excluded.

### Confusion matrix — worked example

Real output from `runcorpus.py` against `loco_classifier_app.exe` over
`cleaned/annotated/` (rev2–rev14, 46 trials). A single trial's confusion
matrix (`assemble(a)["frame_metrics"]["confusion"]`, trial_id
`annotated/rev14/csv/annotated_loco_rev14_trial_1.csv`):

```
gt \ pred     STANDING    WALKING
STANDING          2959      11432
WALKING            832      70439
```

→ `recall(STANDING) = 2959/14391 = 0.2056`, `precision(STANDING) = 2959/3791 = 0.7805`
(matches `per_class_by_trial.csv` for that same `trial_id`).

Pooled corpus-wide (`aggregate(result)["confusion"]`, summed over all 46 trials
via `_confusion_stats`) — `SIT_TO_STAND`/`STAND_TO_SIT`/`SITTING` are part of
the algorithm's label schema but never actually occur (as truth or prediction)
in this corpus:

```
gt \ pred        STANDING      WALKING  SIT_TO_STAND  STAND_TO_SIT      SITTING
STANDING           119872       260848           920            53         1153
WALKING             39619      2178613             0             0            0
SIT_TO_STAND            0            0             0             0            0
STAND_TO_SIT            0            0             0             0            0
SITTING                 0            0             0             0            0
```

|  Class   | support | pooled recall | pooled precision | pooled F1 |
| -------- | ------- | -------------- | ------------------ | --------- |
| STANDING | 382,846 | 0.3131          | 0.7516             | 0.4421    |
| WALKING  | 2,218,232 | 0.9821        | 0.8931             | 0.9355    |

This is exactly the shape `write_outputs(..., csv_dir=...)` writes to
`confusion.csv` (one row per `trial_id, gt, pred, count`) — the table above
is that file pivoted and summed.

---

## Error diagnosis (`diagnose.py`)

### Row-level error buckets

Every row lands in **exactly one** bucket, assigned in strict precedence order:

```
(1) correct
    > (2) omission        — gt segment pred never reaches; pred flanks it with two different labels
        > swallowed       — same variant, but pred flanks it with the SAME label both sides
                             (a real bout was fully absorbed and left no trace)
        > edge_omission   — the missed segment touches a recording boundary,
                             so swallowed-vs-omission can't be determined
    > (3) flicker         — pred run shorter than FLICKER_MAX_MS, flanked by equal others
        > (4) late        — pred still shows the old label after a gt transition
            > (5) early   — pred already shows the new label before a gt transition
                > (6) steady_confusion — pred stuck in a different class for the whole gt segment
```

`bucket_errors()` returns counts, fractions, the dominant bucket, and
`total_error_rows` / `correct_rows` (exhaustive: `correct + total_error == n_rows`).

### Transition timing

`transition_timing()` walks forward from every ground-truth transition until
the prediction catches up, and records:

- `offset_ms` — signed lag (positive = late, negative = early), or `None` if messy
  (a third label appeared before the match) or the transition spans a gap
- `sustained_early` / `sustained_late` — the run covers ≥ `SUSTAINED_FRACTION`
  of the adjacent segment **and** the offset exceeds `LAG_MAX_MS` in absolute
  terms — a real misclassification, not detection jitter, and excluded from
  the clean latency stats
- `omission_kind` — `omission` / `swallowed` / `edge_omission` when the
  prediction never reaches the target class at all

`timing_summary()` groups these by `(from, to)` transition pair and reports
late/early medians **separately** (never averaged together, so a transition
that's sometimes very late and sometimes very early doesn't cancel out to a
falsely reassuring near-zero number), plus a `confidence` of `low_sample` when
fewer than `MIN_EVENTS_FOR_STATISTIC` clean events are behind the number.

### Findings

`build_findings()` turns the above into a flat list of `Finding(kind, category,
claim, evidence, severity, confidence)` — the human-readable rollup consumed
by `print_problems()` and by the corpus-level `findings` summary.

| Severity   | Meaning                                                                 |
| ---------- | ------------------------------------------------------------------------ |
| `info`     | the tool's read on the likely source of error, or an FYI-level observation |
| `warn`     | flagged, but either low-sample or not conclusively "bad"                 |
| `critical` | a class/transition the algorithm gets meaningfully wrong, on enough data to trust |

| Confidence   | Meaning                                                     |
| ------------ | ------------------------------------------------------------ |
| `confirmed`  | ≥ `MIN_EVENTS_FOR_STATISTIC` clean events behind the claim   |
| `low_sample` | fewer events than that — real, but don't over-read the number |

---

## Nontechnical report (`nontech.py`)

`assemble_nontechnical(a, *, late_threshold_ms=1000.0)` produces a second,
language-neutral report alongside the technical one, aimed at non-engineer
Korean-speaking QA checkers. It emits **no Korean text** — every field is a
state ID (a `LOCO_NAMES` name, or `CLASS_n` for a class outside that table)
plus numbers. Korean rendering happens downstream in the dashboard, which
maps each error-bucket entry's `(bucket_id, from_state, to_state)` to a
template (specific overrides for notable pairs, a generic per-`bucket_id`
fallback that interpolates the state names otherwise). Every metric here is
*derived* from the same computations behind the technical report
(`frame_metrics`, `transition_timing`, `row_buckets`) — nothing about the
technical numbers changes.

```json
{
  "frame_error_rate": 0.08,
  "total_frames": 50,
  "sub_scores": {
    "transition_mishandled": {
      "count": 0, "denominator": 1, "denominator_kind": "transitions",
      "fraction": 0.0, "late_threshold_ms": 1000.0
    }
  },
  "error_buckets": [
    {"bucket_id": "OTHER", "from_state": "WALKING", "to_state": "STANDING",
     "probability": 0.08, "count": 2, "denominator": 25,
     "denominator_kind": "frames", "translated": true}
  ],
  "total_pred_transitions": 3
}
```

### Composite score

`frame_error_rate = 1 - frame_metrics.overall_accuracy`, `total_frames =
frame_metrics.scored_rows` — the primary, top-level number. This is the only
field a QA checker strictly needs; everything else explains it.

### Event sub-score

`sub_scores.transition_mishandled` — subordinate to (never replacing) the
composite score. `count` = ground-truth transitions never detected (missed)
plus transitions detected later than `late_threshold_ms` (late); denominator
= every ground-truth transition in the trial (matched + missed). This is
exactly the sum of the `MISSED_TRANSITION` + `LATE_TRANSITION` bucket counts
below, pooled across all state pairs.

### Error bucket taxonomy

A fixed enum of `bucket_id`s — never add/remove/rename one without updating
the dashboard's Korean template set:

| `bucket_id`          | Meaning                                                                 | Denominator (`denominator_kind`) |
| --------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| `MISSED_TRANSITION`   | a ground-truth transition never detected                                | gt transitions for that `(from,to)` pair, matched + missed (`transitions`) |
| `LATE_TRANSITION`     | detected, but later than `late_threshold_ms` (default `1000.0`, mirrors `diagnose.LAG_MAX_MS`, independently configurable) | gt transitions for that pair that were detected at all, i.e. matched (`transitions`) |
| `FALSE_TRANSITION`    | a predicted transition with no gt counterpart                            | every predicted transition event in the trial, shared across all pairs (`transitions`) |
| `WRONG_STEADY_STATE`  | steady-state frames misclassified (the row-level `steady_confusion` bucket, broken down by `(gt, pred)`) | gt support rows for `from_state` (`frames`) |
| `OTHER`               | every other misclassified frame — the frame-level residue of `omission`/`swallowed`/`edge_omission`/`flicker`/`late`/`early` row buckets, i.e. transition-boundary effects already counted at the event level above | gt support rows for `from_state` (`frames`) |

`WRONG_STEADY_STATE` + `OTHER` frame counts always sum exactly to the
technical report's `error_buckets.total_error_rows` — every misclassified
frame lands in one or the other, so nothing is silently dropped.
`MISSED_TRANSITION` and `LATE_TRANSITION` are emitted for every `(from,to)`
pair the trial actually observed, even when the count is 0 — this keeps the
list small (bounded by the trial's distinct transition types) while making
corpus-level pooling exact (see below).

Every entry carries an explicit `count`/`denominator`/`denominator_kind`
alongside its `probability` — the ratio is never presented without the
numbers behind it. `translated: false` marks an entry where either state is
a `CLASS_n` fallback (no `LOCO_NAMES` entry), the dashboard's cue to fall
back to a generic template.

### Resolving the asymmetric label space

Some predicted classes (transient blips the algorithm passes through)
never appear in ground truth at all. `FALSE_TRANSITION` handles this: a gt
transition is "resolved" by exactly one predicted-transition boundary
(`diagnose.TransitionResult.resolved_pred_idx`); every *other* predicted
transition boundary — including a transient intermediate class passed
through en route to the real target — has no gt counterpart of its own, so
it's attributed by **the ground-truth state at that instant** (`from_state`)
and whatever class the prediction jumped to (`to_state`, which may itself be
an untranslated `CLASS_n`). E.g. a 150ms blip to an unmapped class 99 in the
middle of a `WALKING` segment surfaces as `FALSE_TRANSITION WALKING ->
CLASS_99` (`translated: false`) for the entry into the blip, and
`FALSE_TRANSITION WALKING -> WALKING` for the exit (a real predicted-transition
event with no gt counterpart, even though it happens to land back on the
correct class).

### Corpus-level pooling (`aggregate_nontechnical`)

`runcorpus.py --json-nontechnical` pools every trial's nontechnical report.
This is not a naive sum of each trial's emitted `error_buckets` entries,
because a trial that observed a `(bucket_id, from, to)` pair but had zero
occurrences of it still needs to contribute its true denominator — dropping
it would silently inflate the pooled probability. Concretely:

- `frame_error_rate` — pooled scored-rows-weighted (like `overall_accuracy`
  elsewhere), not a plain mean of per-trial rates.
- `transition_mishandled` — `count`/`denominator` summed directly from each
  trial's own sub-score (both already exact integers).
- `MISSED_TRANSITION` / `LATE_TRANSITION` — summed directly from trials'
  entries, since every trial that observed the pair emits an entry for it
  (0-count included).
- `FALSE_TRANSITION` — denominator is `total_pred_transitions` summed across
  **every** trial (not just ones that produced that exact pair).
- `WRONG_STEADY_STATE` / `OTHER` — denominator is the gt `support_rows` for
  `from_state`, pooled across every trial (the same support-row pooling
  `accuracy.per_class` already does), not summed entry-level denominators.

---

## Corpus overview (`fullreport.py`)

`discover_trials(input_root, gt_root)` recursively finds every `*.csv` under
`<input_root>` and keeps the ones with a recognizable time column that also
have a matching CSV at the same relative path under `<gt_root>` with a
recognizable label column, regardless of naming or folder depth (anything
else is reported in `skipped`, not raised). Each kept file gets a `trial_id`
(its path relative to `<input_root>`, always unique) and a `group_id` (its
containing folder relative to `<input_root>` — the natural stand-in for
whatever grouping the corpus's folder structure happens to encode, collapsing
to a single group for a flat corpus). `run_corpus()` runs the algorithm on
each input CSV and the full per-trial pipeline (`load_aligned` → `assemble` →
`assemble_nontechnical`) against its matched ground-truth CSV, catching and
recording per-trial failures instead of aborting the whole run (and, when
`report_dir` is given, writing that trial's `report_technical_*.json` /
`report_nontechnical_*.json` as it goes). `aggregate()` then rolls the
technical side up:

| Section        | Question answered                                                                   |
| -------------- | -------------------------------------------------------------------------------------- |
| `coverage`     | How many trials ran, across which groups, with how much in-gap data?                   |
| `error_source` | Is the same error bucket dominant across trials, or is it noisy/one-trial-specific?    |
| `confusion`    | Pooled confusion matrix across every trial                                             |
| `latency`      | Is detection lag (late) or premature prediction (early) consistent in magnitude?       |
| `accuracy`     | Which classes are reliable, and how much do they swing trial-to-trial (`std_recall`)?  |
| `by_group`     | Which group (folder) performs best/worst, by the macro-F1 headline?                    |
| `trials`       | The 10 worst / best individual trials by macro-F1 — where to start debugging           |
| `findings`     | All per-trial findings pooled, by severity and by kind                                 |

Two recurring guards worth knowing about:

- **provenance / concentration** — every pooled stat (a bucket's row share, a
  class's error share, a transition's late/early lag) tracks whether it's
  actually spread across the corpus or dragged around by one noisy trial
  (`CONCENTRATION_THRESHOLD = 0.5`); `print_fullreport` calls this out inline
- **`corpus_balanced_accuracy`** pools each class's recall support-weighted
  across the *whole* corpus before averaging classes — distinct from
  `balanced_accuracy.mean`, which averages each trial's already-computed
  balanced accuracy

`aggregate_nontechnical()` is a separate function (not folded into
`aggregate()`) specifically so `aggregate()`'s return value — the technical
report — stays byte-for-byte identical to before the technical/nontechnical
split; see [Nontechnical report](#nontechnical-report-nontechpy) above for
its pooling rules.

---

## Testing

- **`selftest.py`** — fault-injection unit tests on synthetic data (`faults.py`):
  known injected latency is recovered within tolerance, a swallowed short bout
  is detected as `omission`/`swallowed`, a sustained mid-segment relabel becomes
  `steady_confusion`, long early/late runs become `sustained_early`/`sustained_late`,
  clean (identical) prediction has ~zero lag and no misses, bucket assignment is
  exhaustive, a fully-missed class scores F1 = 0 (not NaN) and trips the
  weak-class finding, and a NaN label is rejected at construction.
  Also covers the nontechnical report: `LATE_TRANSITION` recovers an injected
  delay, `MISSED_TRANSITION` catches a swallowed bout (and the
  `transition_mishandled` sub-score matches), a transient pred-only class
  surfaces as `FALSE_TRANSITION` attributed to the surrounding gt state with
  `translated: false`, and `WRONG_STEADY_STATE` + `OTHER` frame counts sum
  exactly to the technical report's `total_error_rows`.
- **`test_fullreport.py`** — corpus-aggregation unit tests against hand-built
  `TrialRecord`s (no `.exe` or real data needed): trial discovery, pooled
  latency/accuracy/error-source math, exact pooled confusion-derived precision/F1,
  error concentration, worst/best trial ranking, findings rollup, and the
  `--csv-dir` breakdown files. Also covers `aggregate_nontechnical()`'s
  pooling rules — in particular that a trial observing a state pair with zero
  errors on it still contributes its true denominator, rather than being
  silently excluded.

---

## Architecture

| Module        | Responsibility                                                                    |
| ------------- | ------------------------------------------------------------------------------------ |
| `__init__`    | public API surface                                                                   |
| `core`        | CSV alignment by row index; health checks; transitions/segments helpers              |
| `measure`     | frame-level metrics: confusion matrix, per-class precision/recall/F1, accuracy headlines |
| `diagnose`    | row-level error buckets, transition timing, findings                                 |
| `nontech`     | nontechnical (language-neutral) per-trial report: composite score, event sub-score, fixed error-bucket taxonomy |
| `faults`      | synthetic ground truth + fault injection, used only by `selftest.py`                 |
| `runner`      | invokes the algorithm executable on an input CSV                                     |
| `report`      | single-trial technical report assembly (`assemble`), printing (`print_problems`), and writing both reports to disk (`write_reports`) |
| `fullreport`  | corpus-level discovery, aggregation (technical `aggregate` + nontechnical `aggregate_nontechnical`), printing, and CSV/JSON output |

---

## Locomotion classes

| Class          | Value |
| -------------- | ----- |
| `STANDING`     | 0     |
| `BENDING`      | 5     |
| `WALKING`      | 10    |
| `SIT_TO_STAND` | 20    |
| `STAND_TO_SIT` | 25    |
| `SITTING`      | 30    |
| `UNKNOWN`      | 255   |
| `CLASS_<n>`    | any label value not in the table above |

---

## Tunable constants

No runtime knobs — everything lives as module-level constants near the top of
each file.

| Constant                    | Module      | Default  | Meaning                                                                 |
| --------------------------- | ----------- | -------- | ------------------------------------------------------------------------ |
| `TIME_ALIGN_TOL_MS`         | `core`      | `10.0`   | max input/output timestamp diff (ms) to count as aligned                |
| `ECHO_ALIGN_TOL`            | `core`      | `1.0`    | max input/output echoed-feature diff to count as aligned                |
| `GAP_MAX`                   | `core`      | `15.0`   | dt (ms) above which a row starts a timeline gap                         |
| `FLICKER_MAX_MS`            | `diagnose`  | `200.0`  | pred run shorter than this, flanked by equal others, is `flicker`       |
| `MIN_EVENTS_FOR_STATISTIC`  | `diagnose`  | `10`     | below this many clean events, timing/finding confidence is `low_sample` |
| `LAG_MAX_MS`                | `diagnose`  | `1000.0` | beyond this, a lag is too large to be ordinary detection jitter         |
| `WEAK_CLASS_F1`             | `diagnose`  | `0.5`    | per-class F1 below this triggers a weak-class finding                  |
| `SUSTAINED_FRACTION`        | `diagnose`  | `0.5`    | + clears `LAG_MAX_MS` → sustained misclassification, not jitter         |
| `DEFAULT_LATE_THRESHOLD_MS` | `nontech`   | `1000.0` | nontechnical `LATE_TRANSITION` cutoff (mirrors `LAG_MAX_MS`); override per call via `late_threshold_ms`, or via `--late-threshold-ms` on either CLI |

---

## Reading a single-trial report

```
PROBLEMS
  [INFO    ] Likeliest source of error: 'swallowed' (58% of mismatched frames)
  [CRITICAL] Class STANDING: F1 41% over 146.22s (genuine weakness)
  [WARN    ] STANDING -> WALKING: median late 1935 ms over 4 event(s) — exceeds 1000 ms threshold
  [WARN    ] STANDING -> WALKING: 1 missed (1 swallowed)
  [INFO    ] STANDING -> WALKING: 7 cover the whole preceding segment (already counted as missed)
```

(taken from a real run, rev2/trial6). Each line is one `Finding`; a
`[LOW_SAMPLE]` tag appears inline when confidence isn't `confirmed`.
