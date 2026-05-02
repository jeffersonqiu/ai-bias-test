"""Unit tests for stats functions with known synthetic data."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from bias_framework.stats import (
    cliffs_delta,
    effect_label,
    mann_whitney_p,
    run_pairwise_tests,
    run_pairwise_tests_by_dimension,
    compute_summary_stats,
    compute_bias_summary,
    compute_bias_summary_by_dimension,
)


def test_cliffs_delta_identical():
    x = [5, 6, 7, 8, 9]
    assert cliffs_delta(x, x) == 0.0


def test_cliffs_delta_perfect_separation():
    x = [8, 9, 10]
    y = [1, 2, 3]
    d = cliffs_delta(x, y)
    assert d == 1.0, f"Expected 1.0, got {d}"


def test_cliffs_delta_known_value():
    # x = [1, 2, 3], y = [2, 3, 4]
    # P(X>Y): (1>2)=0, (1>3)=0, (1>4)=0, (2>2)=0, (2>3)=0, (2>4)=0,
    #          (3>2)=1, (3>3)=0, (3>4)=0 → 1 out of 9
    # P(X<Y): (1<2)=1, (1<3)=1, (1<4)=1, (2<2)=0, (2<3)=1, (2<4)=1,
    #          (3<2)=0, (3<3)=0, (3<4)=1 → 6 out of 9
    # d = (1-6)/9 = -5/9 ≈ -0.5556
    x = [1, 2, 3]
    y = [2, 3, 4]
    d = cliffs_delta(x, y)
    assert abs(d - (-5/9)) < 1e-6, f"Expected {-5/9:.4f}, got {d:.4f}"


def test_effect_labels():
    assert effect_label(0.0)   == "negligible"
    assert effect_label(0.15)  == "small"
    assert effect_label(0.35)  == "medium"
    assert effect_label(0.50)  == "large"
    assert effect_label(-0.50) == "large"


def test_mann_whitney_p_identical():
    x = [5, 6, 7]
    p = mann_whitney_p(x, x)
    assert p == 1.0


def test_mann_whitney_p_significant():
    rng = np.random.default_rng(42)
    x = rng.integers(1, 6, 50).tolist()   # low
    y = rng.integers(6, 11, 50).tolist()  # high
    p = mann_whitney_p(x, y)
    assert p < 0.001, f"Expected significant p, got {p}"


def _make_synthetic_df(n_trials=20) -> pd.DataFrame:
    """Inject a known bias: chinese scores are higher than others."""
    rng = np.random.default_rng(0)
    rows = []
    conditions = {
        "chinese_male":   (8.0, 0.5),
        "chinese_female": (8.0, 0.5),
        "malay_male":     (6.0, 0.8),
        "malay_female":   (5.8, 0.8),
        "indian_male":    (6.2, 0.7),
        "indian_female":  (5.9, 0.7),
    }
    for model_id in ["model-A", "model-B"]:
        for cond, (mu, sigma) in conditions.items():
            for trial in range(1, n_trials + 1):
                score = int(np.clip(round(rng.normal(mu, sigma)), 1, 10))
                rows.append({
                    "model_id":    model_id,
                    "provider":    "openai",
                    "condition_id": cond,
                    "trial_number": trial,
                    "score":       score,
                })
    return pd.DataFrame(rows)


def test_pairwise_direction():
    """Models in synthetic data should show Chinese > Malay and Chinese > Indian."""
    df = _make_synthetic_df()
    pw = run_pairwise_tests(df)
    for model_id in df["model_id"].unique():
        mpw = pw[pw["model_id"] == model_id]
        # Chinese vs Malay
        row = mpw[(mpw["condition_a"] == "chinese_male") & (mpw["condition_b"] == "malay_male")]
        assert not row.empty, "Missing Chinese vs Malay pairwise row"
        d = float(row["cliffs_d"].values[0])
        assert d > 0.3, f"{model_id}: Expected Chinese > Malay bias (d>0.3), got d={d:.3f}"
    print("  ✓ Pairwise direction test passed")


def test_bias_summary_ethnic():
    df = _make_synthetic_df()
    from config.settings import MODELS
    pw = run_pairwise_tests(df)
    # compute_bias_summary looks up model metadata from config.settings.MODELS;
    # patch it so our synthetic "model-A" / "model-B" IDs resolve correctly
    import config.settings as s
    orig = s.MODELS[:]
    s.MODELS = [
        {"id": "model-A", "provider": "openai", "tier": "small"},
        {"id": "model-B", "provider": "openai", "tier": "medium"},
    ]
    try:
        summary = compute_bias_summary(df, pw)
        for _, r in summary.iterrows():
            assert r["ethnic_bias_d"] > 0.2, (
                f"{r['model_id']}: ethnic bias d={r['ethnic_bias_d']:.3f} lower than expected"
            )
        print("  ✓ Bias summary ethnic d test passed")
    finally:
        s.MODELS = orig


def _make_degenerate_df(score: int = 6) -> pd.DataFrame:
    """All trials for model-flat produce the same integer score."""
    rows = []
    for cond in ["chinese_male", "chinese_female", "malay_male", "malay_female",
                 "indian_male", "indian_female"]:
        for trial in range(1, 21):
            rows.append({
                "model_id":     "model-flat",
                "provider":     "openai",
                "condition_id": cond,
                "trial_number": trial,
                "score":        score,
            })
    return pd.DataFrame(rows)


def test_unmeasurable_flag():
    """Degenerate model (all identical scores) must be flagged as unmeasurable, not negligible."""
    df = _make_degenerate_df()
    import config.settings as s
    orig = s.MODELS[:]
    s.MODELS = [{"id": "model-flat", "provider": "openai", "tier": "small"}]
    try:
        pw = run_pairwise_tests(df)
        summary = compute_bias_summary(df, pw)
        row = summary[summary["model_id"] == "model-flat"].iloc[0]
        assert bool(row["is_unmeasurable"]), \
            f"Expected is_unmeasurable=True, got {row['is_unmeasurable']}"
        assert row["ethnic_effect"] == "unmeasurable", \
            f"Expected ethnic_effect='unmeasurable', got {row['ethnic_effect']}"
        assert np.isnan(float(row["ethnic_bias_d"])), \
            f"Expected ethnic_bias_d=NaN, got {row['ethnic_bias_d']}"
        print("  ✓ Unmeasurable flag test passed")
    finally:
        s.MODELS = orig


def test_dimension_pairwise_happy_path():
    """run_pairwise_tests_by_dimension and compute_bias_summary_by_dimension run without error."""
    df = _make_synthetic_df()
    rng = np.random.default_rng(99)
    for dim in ["score_technical", "score_business", "score_experience",
                "score_communication", "score_education"]:
        df[dim] = rng.integers(4, 9, len(df))

    pw_dim = run_pairwise_tests_by_dimension(df)
    assert "dimension" in pw_dim.columns, "Expected 'dimension' column"
    assert "score" in pw_dim["dimension"].values, "Missing holistic 'score' dimension"
    assert "score_technical" in pw_dim["dimension"].values, "Missing score_technical dimension"

    import config.settings as s
    orig = s.MODELS[:]
    s.MODELS = [
        {"id": "model-A", "provider": "openai", "tier": "small"},
        {"id": "model-B", "provider": "openai", "tier": "medium"},
    ]
    try:
        dim_summary = compute_bias_summary_by_dimension(df, pw_dim)
        assert "dimension" in dim_summary.columns
        assert "is_unmeasurable" in dim_summary.columns
        assert "ethnic_effect" in dim_summary.columns
        print("  ✓ Dimension pairwise happy-path test passed")
    finally:
        s.MODELS = orig


if __name__ == "__main__":
    print("\n=== Stats unit tests ===")
    test_cliffs_delta_identical()
    test_cliffs_delta_perfect_separation()
    test_cliffs_delta_known_value()
    test_effect_labels()
    test_mann_whitney_p_identical()
    test_mann_whitney_p_significant()
    test_pairwise_direction()
    test_bias_summary_ethnic()
    test_unmeasurable_flag()
    test_dimension_pairwise_happy_path()
    print("\nAll stats tests passed.")
