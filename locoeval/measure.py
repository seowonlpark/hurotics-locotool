"""
measure.py — compute locomotion evaluation metrics

Frame agreement only: how often each row's label is right
    - confusion matrix
    - per-class
        - precision
        - recall
        - F1
    - accuracy headlines
        - overall
        - balanced
        - macro-F1

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .core import AlignedData, cname

UNKNOWN_CLASS = 255  # excluded from balanced accuracy and macro-F1 (both averages, at the same source loop)

@dataclass
class FrameMetrics:
    classes: list
    confusion: dict # confusion[true][pred] = count (square over union of classes)
    per_class: dict # name -> {precision, recall, f1, support_rows, support_s}
    overall_accuracy: float # vanity number: imbalance-sensitive, not the headline
    balanced_accuracy: float # mean per-class recall (UNKNOWN excluded)
    macro_f1: float # headline: mean per-class F1 (UNKNOWN excluded)
    scored_rows: int

def frame_metrics(a: AlignedData) -> FrameMetrics:
    df = a.df
    # Every row is scored, in_gap or not (AlignedData guarantees no NaN gt/pred label).
    gt = df["gt"].to_numpy(dtype=int)
    pr = df["pred"].to_numpy(dtype=int)
    classes = sorted(set(gt.tolist()) | set(pr.tolist()))
    idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    M = np.zeros((K, K), dtype=int)
    for g, p in zip(gt, pr):
        M[idx[g], idx[p]] += 1

    nominal_s = a.health.nominal_dt_ms / 1000.0
    per_class, recalls, f1s = {}, [], []
    for c in classes:
        i = idx[c]
        tp = M[i, i]
        fp = M[:, i].sum() - tp # predicted c, truth other
        support = M[i, :].sum() # rows whose TRUTH is c
        recall = tp / support if support else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        # tp == 0 with real GT support is a fully-missed class: F1 is 0 by definition,
        # never NaN (NaN would silently drop the worst class from macro-F1).
        # tp == 0 with no GT support (pred-only class) stays NaN -> None -> excluded.
        if tp > 0:
            f1 = 2*precision*recall/(precision+recall)
        else:
            f1 = 0.0 if support else float("nan")
        if support and c != UNKNOWN_CLASS:
            recalls.append(recall)
        if not np.isnan(f1) and c != UNKNOWN_CLASS:
            f1s.append(f1)
        per_class[cname(c)] = {
            "precision": _r(precision), "recall": _r(recall), "f1": _r(f1),
            "support_rows": int(support), "support_s": round(support * nominal_s, 2),
        }

    overall = float(np.trace(M) / M.sum()) if M.sum() else float("nan")
    balanced = float(np.nanmean(recalls)) if recalls else float("nan")
    macro_f1 = float(np.nanmean(f1s)) if f1s else float("nan")
    confusion = {cname(t): {cname(p): int(M[idx[t], idx[p]]) for p in classes} for t in classes}
    return FrameMetrics(
        [cname(c) for c in classes], 
        confusion, 
        per_class,
        _r(overall), 
        _r(balanced), 
        _r(macro_f1), 
        int(len(df))
    )

def _r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)
