from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Cliff's delta ──────────────────────────────────────────────────────────────

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta: d = (P(X>Y) - P(X<Y)), range [-1, 1].

    Positive d means group X tends to score higher than group Y.
    """
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    n, m = len(a), len(b)
    greater = float(np.sum(a[:, None] > b[None, :]))
    less    = float(np.sum(a[:, None] < b[None, :]))
    return (greater - less) / (n * m)


def cliffs_delta_ci(
    x: list[float],
    y: list[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap percentile CI for Cliff's delta.

    Returns (lower, upper) at the requested confidence level.
    Falls back to (nan, nan) if either sample has zero variance (tied ranks
    make bootstrap CIs uninformative — the point estimate is still valid).
    """
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if len(set(a)) == 1 and len(set(b)) == 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        greater = float(np.sum(sa[:, None] > sb[None, :]))
        less    = float(np.sum(sa[:, None] < sb[None, :]))
        boot[i] = (greater - less) / (len(sa) * len(sb))
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return (round(lo, 4), round(hi, 4))


_EFFECT_THRESHOLDS = [
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
    (float("inf"), "large"),
]


def effect_label(d: float) -> str:
    for threshold, label in _EFFECT_THRESHOLDS:
        if abs(d) < threshold:
            return label
    return "large"


# ── Mann-Whitney U ─────────────────────────────────────────────────────────────

def mann_whitney_p(x: list[float], y: list[float]) -> float:
    """Two-sided Mann-Whitney U p-value. Returns 1.0 for degenerate inputs."""
    if len(x) < 2 or len(y) < 2:
        return 1.0
    if len(set(x) | set(y)) == 1:  # all values identical
        return 1.0
    result = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(result.pvalue)


# ── Pairwise tests ─────────────────────────────────────────────────────────────

def run_pairwise_tests(
    df: pd.DataFrame,
    bootstrap_ci: bool = False,
    score_col: str = "score",
    n_tests_override: Optional[int] = None,
) -> pd.DataFrame:
    """For each model × condition-pair, compute Cliff's d, p-value, and effect label.

    Bonferroni threshold is computed from measurable models only (models where
    score_col has >1 unique value), so uniform-scoring models don't inflate the
    family and unnecessarily penalise the models that actually vary.

    A Kruskal-Wallis omnibus test is run per model first. Pairwise significance
    is gated on the KW result: a pair is only flagged significant when both the
    omnibus KW p < 0.05 AND the Bonferroni-adjusted pairwise p-value is met.
    The raw KW p-value is retained in the `kw_pvalue` column for inspection.

    Args:
        bootstrap_ci: If True, add `cliffs_d_lo` and `cliffs_d_hi` 95% CI columns
                      via bootstrap resampling (n_boot=2000). Adds ~1s per 30 pairs.
        score_col: Column to use as the score (default "score"; pass a dimension
                   column like "score_technical" or "score_composite" for sub-dimension
                   or composite-score analysis).
        n_tests_override: If set, use this as the Bonferroni denominator instead of
                          computing it from the current DataFrame alone.
    """
    conditions = list(df["condition_id"].unique())
    pairs = list(itertools.combinations(conditions, 2))
    models = list(df["model_id"].unique())

    # #6: only count models that actually vary — uniform scorers have p=1 by
    # construction and don't consume real Type I error budget.
    if n_tests_override is not None:
        n_tests = n_tests_override
    else:
        measurable = [
            m for m in models
            if df[(df["model_id"] == m) & df[score_col].notna()][score_col].nunique() > 1
        ]
        n_tests = len(measurable) * len(pairs) if measurable else len(models) * len(pairs)
    alpha = 0.05 / n_tests if n_tests > 0 else 0.05

    records = []
    for model_id in models:
        mdf = df[(df["model_id"] == model_id) & df[score_col].notna()]

        # #7: Kruskal-Wallis omnibus — gate pairwise tests on this
        kw_groups = [
            mdf[mdf["condition_id"] == c][score_col].dropna().tolist()
            for c in conditions
            if len(mdf[mdf["condition_id"] == c][score_col].dropna()) >= 2
        ]
        all_same = len({v for g in kw_groups for v in g}) <= 1
        if len(kw_groups) >= 2 and not all_same:
            kw_stat, kw_p = scipy_stats.kruskal(*kw_groups)
            kw_p = float(kw_p)
        else:
            kw_p = 1.0
        kw_significant = kw_p < 0.05

        for cond_a, cond_b in pairs:
            x = mdf[mdf["condition_id"] == cond_a][score_col].tolist()
            y = mdf[mdf["condition_id"] == cond_b][score_col].tolist()
            if len(x) < 3 or len(y) < 3:
                continue
            d = cliffs_delta(x, y)
            p = mann_whitney_p(x, y)
            row: dict = {
                "model_id":    model_id,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n_a":         len(x),
                "n_b":         len(y),
                "cliffs_d":    round(d, 4),
                "p_value":     round(p, 6),
                "kw_pvalue":   round(kw_p, 6),
                "significant": kw_significant and (p < alpha),
                "effect":      effect_label(d),
                "alpha":       round(alpha, 6),
            }
            if bootstrap_ci:
                lo, hi = cliffs_delta_ci(x, y)
                row["cliffs_d_lo"] = lo
                row["cliffs_d_hi"] = hi
            records.append(row)

    return pd.DataFrame(records)


def run_mean_comparison_tests(
    df: pd.DataFrame,
    score_col: str = "score_composite",
) -> pd.DataFrame:
    """For each model × condition-pair, compare group means with Welch's t-test.

    Returns all C(6,2)=15 pairs per model. No Bonferroni correction, no KW gate.
    Significance threshold: p < 0.05 (two-sided).

    Columns returned:
        model_id, condition_a, condition_b,
        n_a, n_b, mean_a, mean_b, mean_diff,
        t_stat, p_value, significant
    """
    conditions = list(df["condition_id"].unique())
    pairs = list(itertools.combinations(conditions, 2))
    models = list(df["model_id"].unique())

    records = []
    for model_id in models:
        mdf = df[(df["model_id"] == model_id) & df[score_col].notna()]
        for cond_a, cond_b in pairs:
            x = mdf[mdf["condition_id"] == cond_a][score_col].dropna().values.astype(float)
            y = mdf[mdf["condition_id"] == cond_b][score_col].dropna().values.astype(float)
            if len(x) < 2 or len(y) < 2:
                continue
            mean_a = float(np.mean(x))
            mean_b = float(np.mean(y))
            if np.std(x) == 0 and np.std(y) == 0:
                t_stat, p_val = 0.0, 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    t_result = scipy_stats.ttest_ind(x, y, equal_var=False)
                t_stat = float(t_result.statistic)
                p_val = float(t_result.pvalue)
            records.append({
                "model_id":    model_id,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "n_a":         len(x),
                "n_b":         len(y),
                "mean_a":      round(mean_a, 3),
                "mean_b":      round(mean_b, 3),
                "mean_diff":   round(mean_a - mean_b, 3),
                "t_stat":      round(t_stat, 3),
                "p_value":     round(p_val, 4),
                "significant": p_val < 0.05,
            })

    return pd.DataFrame(records)


def run_pairwise_tests_by_dimension(
    df: pd.DataFrame,
    bootstrap_ci: bool = False,
) -> pd.DataFrame:
    """Run pairwise tests for each dimension column plus the holistic score.

    The Bonferroni budget is shared across ALL (dimensions × models × pairs),
    preserving family-wise error rate control jointly across dimensions.

    Returns a DataFrame with the same columns as run_pairwise_tests plus a
    leading 'dimension' column identifying which score column was tested.
    """
    from bias_framework.storage import DIMENSION_COLS
    all_cols = ["score"] + list(DIMENSION_COLS)

    conditions = list(df["condition_id"].unique())
    models = list(df["model_id"].unique())
    n_pairs = len(list(itertools.combinations(conditions, 2)))
    valid_cols = [c for c in all_cols if c in df.columns and df[c].notna().any()]
    total_n_tests = len(models) * n_pairs * len(valid_cols)

    frames = []
    for col in valid_cols:
        frame = run_pairwise_tests(
            df,
            bootstrap_ci=bootstrap_ci,
            score_col=col,
            n_tests_override=total_n_tests,
        )
        frame.insert(1, "dimension", col)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Summary stats per (model, condition) ──────────────────────────────────────

def compute_summary_stats(
    df: pd.DataFrame,
    score_col: str = "score_composite",
) -> pd.DataFrame:
    """Mean, SD, median, n, refusal_rate per (model_id, condition_id)."""
    valid = df[df[score_col].notna()].copy()
    total_rows = df.groupby(["model_id", "condition_id"]).size().rename("n_total")
    agg = (
        valid
        .groupby(["model_id", "condition_id"])[score_col]
        .agg(mean="mean", std="std", median="median", n_valid="count")
        .round(3)
    )
    result = agg.join(total_rows, how="left")
    result["refusal_rate"] = (
        (result["n_total"] - result["n_valid"]) / result["n_total"]
    ).round(3)
    return result.reset_index()


# ── Bias summary per model ─────────────────────────────────────────────────────

_ETHNIC_PAIRS = [
    ("chinese_male",   "malay_male"),
    ("chinese_female", "malay_female"),
    ("chinese_male",   "indian_male"),
    ("chinese_female", "indian_female"),
    ("malay_male",     "indian_male"),
    ("malay_female",   "indian_female"),
]

_GENDER_PAIRS = [
    ("chinese_male",  "chinese_female"),
    ("malay_male",    "malay_female"),
    ("indian_male",   "indian_female"),
]


def compute_bias_summary(
    df: pd.DataFrame,
    pairwise: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """One row per model: ethnic bias d, gender bias d, worst intersectional cell.

    Adds measurability metadata:
      - score_std: within-model score standard deviation
      - n_unique_scores: number of distinct score values the model emits
      - is_unmeasurable: True when n_unique_scores <= 1 (all trials identical)

    When is_unmeasurable=True:
      - ethnic_bias_d / gender_bias_d are NaN, NOT zero. Zero would be
        mathematically forced by tied Mann-Whitney and conflated with true equity.
        NaN explicitly marks "cannot measure" as distinct from "measured zero bias".
      - ethnic_effect / gender_effect = "unmeasurable" (not "negligible").
      - Bootstrap CIs are also NaN.

    ethnic_bias_d_lo/hi and gender_bias_d_lo/hi are 95% bootstrap CIs on the
    pooled mean |Cliff's δ| across the relevant pair sets. Only populated for
    measurable models.
    """
    from config.settings import DEMOGRAPHIC_CONDITIONS, MODELS

    model_meta = {m["id"]: m for m in MODELS}
    records = []

    for model_id in df["model_id"].unique():
        mdf = df[(df["model_id"] == model_id) & df[score_col].notna()]
        mpw = pairwise[pairwise["model_id"] == model_id]

        score_std = float(mdf[score_col].std(ddof=1)) if len(mdf) > 1 else 0.0
        n_unique = int(mdf[score_col].nunique())
        is_unmeasurable = n_unique <= 1

        if is_unmeasurable:
            ethnic_d = float("nan")
            ethnic_d_lo, ethnic_d_hi = float("nan"), float("nan")
            ethnic_eff = "unmeasurable"
            gender_d = float("nan")
            gender_d_lo, gender_d_hi = float("nan"), float("nan")
            gender_eff = "unmeasurable"
            worst_ethnic_pair, worst_ethnic_d = "", float("nan")
        else:
            ethnic_d, ethnic_d_lo, ethnic_d_hi = _pool_pairs_with_ci(mdf, _ETHNIC_PAIRS, score_col=score_col)
            ethnic_eff = effect_label(ethnic_d)
            gender_d, gender_d_lo, gender_d_hi = _pool_pairs_with_ci(mdf, _GENDER_PAIRS, score_col=score_col)
            gender_eff = effect_label(gender_d)
            worst_ethnic_pair, worst_ethnic_d = _worst_pair(mdf, _ETHNIC_PAIRS, score_col=score_col)

        grand_mean = mdf[score_col].mean()
        cell_means = mdf.groupby("condition_id")[score_col].mean()
        worst_cell = (cell_means - grand_mean).abs().idxmax() if not cell_means.empty else ""
        worst_residual = round(float((cell_means - grand_mean).abs().max()), 3) if not cell_means.empty else None

        n_sig = int(mpw["significant"].sum()) if not mpw.empty else 0
        meta = model_meta.get(model_id, {})

        records.append({
            "model_id":             model_id,
            "provider":             meta.get("provider", ""),
            "tier":                 meta.get("tier", ""),
            "score_std":            round(score_std, 3),
            "n_unique_scores":      n_unique,
            "is_unmeasurable":      is_unmeasurable,
            "ethnic_bias_d":        round(ethnic_d, 4) if not np.isnan(ethnic_d) else float("nan"),
            "ethnic_bias_d_lo":     ethnic_d_lo,
            "ethnic_bias_d_hi":     ethnic_d_hi,
            "ethnic_effect":        ethnic_eff,
            "worst_ethnic_pair":    worst_ethnic_pair,
            "worst_ethnic_pair_d":  round(worst_ethnic_d, 4) if not np.isnan(worst_ethnic_d) else float("nan"),
            "gender_bias_d":        round(gender_d, 4) if not np.isnan(gender_d) else float("nan"),
            "gender_bias_d_lo":     gender_d_lo,
            "gender_bias_d_hi":     gender_d_hi,
            "gender_effect":        gender_eff,
            "worst_cell":           worst_cell,
            "worst_residual":       worst_residual,
            "n_significant":        n_sig,
        })

    return pd.DataFrame(records).sort_values("ethnic_bias_d", ascending=False, na_position="last")


def _pool_pairs(
    mdf: pd.DataFrame,
    pairs: list[tuple[str, str]],
    score_col: str = "score",
) -> float:
    """Mean Cliff's |δ| across a list of condition pairs."""
    deltas = []
    for cond_a, cond_b in pairs:
        x = mdf[mdf["condition_id"] == cond_a][score_col].tolist()
        y = mdf[mdf["condition_id"] == cond_b][score_col].tolist()
        if x and y:
            deltas.append(abs(cliffs_delta(x, y)))
    return float(np.mean(deltas)) if deltas else 0.0


def _pool_pairs_with_ci(
    mdf: pd.DataFrame,
    pairs: list[tuple[str, str]],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    score_col: str = "score",
) -> tuple[float, float, float]:
    """Mean |Cliff's δ| across condition pairs, with 95% bootstrap percentile CI.

    Returns (mean_abs_d, ci_lo, ci_hi).
    Returns (nan, nan, nan) if no valid pairs.
    Returns (mean, nan, nan) if all samples are tied (bootstrap uninformative).
    """
    data: list[tuple[np.ndarray, np.ndarray]] = []
    for cond_a, cond_b in pairs:
        x = mdf[mdf["condition_id"] == cond_a][score_col].dropna().values.astype(float)
        y = mdf[mdf["condition_id"] == cond_b][score_col].dropna().values.astype(float)
        if len(x) and len(y):
            data.append((x, y))

    if not data:
        return float("nan"), float("nan"), float("nan")

    def _abs_d(a: np.ndarray, b: np.ndarray) -> float:
        g = float(np.sum(a[:, None] > b[None, :]))
        lt = float(np.sum(a[:, None] < b[None, :]))
        return abs((g - lt) / (len(a) * len(b)))

    mean_d = float(np.mean([_abs_d(a, b) for a, b in data]))

    if all(len(np.unique(a)) == 1 and len(np.unique(b)) == 1 for a, b in data):
        return mean_d, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = float(np.mean([
            _abs_d(rng.choice(a, size=len(a), replace=True),
                   rng.choice(b, size=len(b), replace=True))
            for a, b in data
        ]))

    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return mean_d, round(lo, 4), round(hi, 4)


def _worst_pair(
    mdf: pd.DataFrame,
    pairs: list[tuple[str, str]],
    score_col: str = "score",
) -> tuple[str, float]:
    """Return (label, |δ|) for the single highest-|δ| pair in the list."""
    best_label, best_d = "", 0.0
    for cond_a, cond_b in pairs:
        x = mdf[mdf["condition_id"] == cond_a][score_col].tolist()
        y = mdf[mdf["condition_id"] == cond_b][score_col].tolist()
        if x and y:
            d = abs(cliffs_delta(x, y))
            if d > best_d:
                best_d = d
                best_label = f"{cond_a} vs {cond_b}"
    return best_label, best_d


# ── Dimension-level bias summary ──────────────────────────────────────────────

def compute_bias_summary_by_dimension(
    df: pd.DataFrame,
    pairwise_by_dim: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (model_id, dimension): pooled mean |Cliff's δ| for ethnic and gender contrasts.

    Consumes the output of run_pairwise_tests_by_dimension. Measurability is
    assessed per (model, dimension): a dimension is unmeasurable when all scores
    for that model are identical in that column.
    """
    from config.settings import MODELS
    model_meta = {m["id"]: m for m in MODELS}

    ethnic_frozensets = {frozenset(p) for p in _ETHNIC_PAIRS}
    gender_frozensets = {frozenset(p) for p in _GENDER_PAIRS}

    def _pool_from_pairwise(rows: pd.DataFrame, pair_set: set) -> float:
        mask = rows.apply(
            lambda r: frozenset([r["condition_a"], r["condition_b"]]) in pair_set,
            axis=1,
        )
        sub = rows[mask]
        return float(sub["cliffs_d"].abs().mean()) if len(sub) else float("nan")

    records = []
    for model_id in pairwise_by_dim["model_id"].unique():
        for dimension in pairwise_by_dim["dimension"].unique():
            mpw = pairwise_by_dim[
                (pairwise_by_dim["model_id"] == model_id) &
                (pairwise_by_dim["dimension"] == dimension)
            ]

            if dimension in df.columns:
                mdf_dim = df[(df["model_id"] == model_id) & df[dimension].notna()]
                n_unique = int(mdf_dim[dimension].nunique())
                is_unmeasurable = n_unique <= 1
            else:
                is_unmeasurable = True

            ethnic_d = _pool_from_pairwise(mpw, ethnic_frozensets)
            gender_d = _pool_from_pairwise(mpw, gender_frozensets)
            n_sig = int(mpw["significant"].sum()) if not mpw.empty else 0
            meta = model_meta.get(model_id, {})

            def _eff(d: float) -> str:
                if is_unmeasurable:
                    return "unmeasurable"
                if np.isnan(d):
                    return "n/a"
                return effect_label(d)

            records.append({
                "model_id":      model_id,
                "provider":      meta.get("provider", ""),
                "tier":          meta.get("tier", ""),
                "dimension":     dimension,
                "ethnic_bias_d": round(ethnic_d, 4) if not np.isnan(ethnic_d) else float("nan"),
                "ethnic_effect": _eff(ethnic_d),
                "gender_bias_d": round(gender_d, 4) if not np.isnan(gender_d) else float("nan"),
                "gender_effect": _eff(gender_d),
                "n_significant": n_sig,
                "is_unmeasurable": is_unmeasurable,
            })

    return pd.DataFrame(records)
