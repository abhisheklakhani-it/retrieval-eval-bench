"""Is system A really better than system B, or is the difference luck? Paired tests over the same queries."""

import numpy as np


def paired_randomization_test(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for mean(a - b) != 0 (sign-flip permutation test, standard in IR evaluation).

    If A and B were equally good, each per-query difference would be equally likely to be positive
    or negative. We flip signs at random many times and count how often the mean difference is at
    least as large as the one we observed.
    """
    diffs = np.asarray(a) - np.asarray(b)
    observed = abs(diffs.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(diffs)))
    extreme = (np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-12).sum()
    return float((extreme + 1) / (n + 1))


def bootstrap_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for mean(a - b), resampling queries with replacement."""
    diffs = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
