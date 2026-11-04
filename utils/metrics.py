"""
Evaluation metrics for nodule classification (numpy only).
Report these on a HELD-OUT, patient-disjoint test set, never on data used for
training, checkpoint selection or calibration.
"""
from typing import Callable, Optional, Sequence, Tuple
import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_s = a[sorter]
    obs = np.r_[True, a_s[1:] != a_s[:-1]]
    dense = np.cumsum(obs)[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, specificity: float = 0.90) -> float:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    thr = np.quantile(p[y == 0], specificity)
    return float((p[y == 1] > thr).mean())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def grouped_bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    groups: Sequence,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """95% CI by resampling PATIENTS (not nodules), since nodules of one patient are correlated."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.RandomState(seed)
    ids = np.unique(groups)
    by_id = {g: np.nonzero(groups == g)[0] for g in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([by_id[g] for g in pick])
        v = metric(y[idx], p[idx])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
