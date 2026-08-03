"""
faults.py

library for selftest.py
    synthetic data creation
    delay transitions
    swallow short bouts
    
"""
from __future__ import annotations
import numpy as np

from .core import transitions, segments

# simple synthetic ground-truth locomotion stream
def make_synthetic(dt: float):
    plan = [(0, 4000), (10, 6000), (0, 3000), (5, 5000), (10, 7000), (0, 4000)]  # (label, dur)
    gt = []
    for label, dur in plan:
        gt += [label] * int(dur / dt)
    gt = np.array(gt, dtype=int)
    t = np.arange(len(gt)) * dt
    return gt, t

# make prediction; same as ground truth, but every transition arrives `delay_ms` late
def delay_transitions(gt: np.ndarray, t: np.ndarray, delay_ms: float) -> np.ndarray:
    dt = float(np.median(np.diff(t)))
    shift = int(round(delay_ms / dt))
    pred = gt.copy()
    for tr in transitions(gt, t):
        i = tr["idx"]
        pred[i:i+shift] = tr["frm"]
    return pred

# delete any short bouts of the given label from gt, replacing them with the previous label
def swallow_short_bouts(gt: np.ndarray, t: np.ndarray, label: int, max_dur_ms: float) -> np.ndarray:
    pred = gt.copy()
    for seg in segments(gt, t):
        if seg["label"] == label and seg["dur_ms"] < max_dur_ms:
            s, e = seg["start"], seg["end"]
            prev = pred[s-1] if s > 0 else (pred[e] if e < len(pred) else label)
            pred[s:e] = prev
    return pred
