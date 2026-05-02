from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bias_framework.stats import (
    compute_summary_stats,
    run_mean_comparison_tests,
    compute_bias_summary,
    cliffs_delta,
)
from bias_framework.storage import DIMENSION_COLS as _DIMENSION_COLS
from config.settings import MODELS as _MODEL_LIST

# ── Brand colours ──────────────────────────────────────────────────────────────
PROVIDER_COLORS = {
    "openai":    "#10A37F",
    "google":    "#4285F4",
    "anthropic": "#D97757",
}

CONDITION_COLORS = {
    "chinese_male":   "#1565C0",
    "chinese_female": "#42A5F5",
    "malay_male":     "#2E7D32",
    "malay_female":   "#66BB6A",
    "indian_male":    "#E65100",
    "indian_female":  "#FFA726",
}

CONDITION_LABELS = {
    "chinese_male":   "Chinese ♂",
    "chinese_female": "Chinese ♀",
    "malay_male":     "Malay ♂",
    "malay_female":   "Malay ♀",
    "indian_male":    "Indian ♂",
    "indian_female":  "Indian ♀",
}

DPI = 150

_TIER_RANK     = {"small": 0, "medium": 1, "flagship": 2}
_PROVIDER_RANK = {"google": 0, "openai": 1, "anthropic": 2}
_TIER_TAG      = {"small": "[S]", "medium": "[M]", "flagship": "[L]"}

_MODEL_META = {m["id"]: m for m in _MODEL_LIST}


def _canonical_model_order(present_models: list[str]) -> list[str]:
    """Sort models S→M→L, Google before OpenAI within each tier."""
    return sorted(
        present_models,
        key=lambda mid: (
            _TIER_RANK.get(_MODEL_META.get(mid, {}).get("tier", ""), 99),
            _PROVIDER_RANK.get(_MODEL_META.get(mid, {}).get("provider", ""), 99),
            mid,
        ),
    )


def _model_title(model_id: str) -> str:
    """Short display title with tier tag, e.g. '[S]\ngemini-2.5\nflash-lite'."""
    meta = _MODEL_META.get(model_id, {})
    tag = _TIER_TAG.get(meta.get("tier", ""), "")
    short = model_id.replace("-", "\n")
    return f"{tag}\n{short}" if tag else short


def _ensure_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Add score_composite column (mean of 5 dimension scores) if not present."""
    if "score_composite" not in df.columns:
        dim_cols = [c for c in _DIMENSION_COLS if c in df.columns and df[c].notna().any()]
        if dim_cols:
            df = df.copy()
            df["score_composite"] = df[dim_cols].mean(axis=1).round(3)
    return df


# ── 0. Score strip plot ────────────────────────────────────────────────────────

def score_distributions(df: pd.DataFrame, outdir: Path) -> Path:
    """Jittered strip plot: each dot = one trial (composite score), per condition per model.

    Models ordered S→M→L, Google before OpenAI within tier.
    """
    df = _ensure_composite(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    valid = df[df[score_col].notna()].copy()
    models = _canonical_model_order(valid["model_id"].unique().tolist())
    conditions = list(CONDITION_LABELS.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, model_id in zip(axes, models):
        mdf = valid[valid["model_id"] == model_id]
        for x_pos, cid in enumerate(conditions):
            scores = mdf[mdf["condition_id"] == cid][score_col].values
            if len(scores):
                jitter = rng.uniform(-0.25, 0.25, size=len(scores))
                ax.scatter(
                    x_pos + jitter, scores,
                    color=CONDITION_COLORS[cid], alpha=0.55, s=18, linewidths=0,
                )
                ax.plot(
                    [x_pos - 0.3, x_pos + 0.3],
                    [scores.mean(), scores.mean()],
                    color=CONDITION_COLORS[cid], linewidth=2.5,
                )
        ax.set_title(_model_title(model_id), fontsize=9)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [CONDITION_LABELS[c] for c in conditions], rotation=40, ha="right", fontsize=8
        )
        ax.set_yticks(range(1, 11))
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_ylim(0.5, 10.5)

    axes[0].set_ylabel("Composite Score (1–10)", fontsize=10)
    fig.suptitle(
        "Individual Trial Scores by Model and Demographic Condition\n"
        "Each dot = 1 trial  ·  Horizontal bar = mean  ·  "
        "Score = mean of 5 dimension sub-scores",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = outdir / "score_distributions.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 1. Condition range plot ────────────────────────────────────────────────────

def grouped_bar(df: pd.DataFrame, outdir: Path) -> Path:
    """Dot + range plot: one column per model, one dot per demographic condition.

    Shows the *spread* of condition means rather than absolute values.
    A vertical line connects the min to the max condition mean, making the
    range of demographic variation immediately visible.
    Models ordered S→M→L, Google before OpenAI within tier.
    """
    df = _ensure_composite(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    summary = compute_summary_stats(df, score_col=score_col)
    models = _canonical_model_order(summary["model_id"].unique().tolist())
    conditions = list(CONDITION_LABELS.keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    dot_size  = 120
    x_jitter  = 0.08   # horizontal spread so overlapping dots are visible

    rng = np.random.default_rng(0)

    for x_pos, model_id in enumerate(models):
        model_means = {}
        for cid in conditions:
            row = summary[(summary["model_id"] == model_id) & (summary["condition_id"] == cid)]
            if len(row):
                model_means[cid] = float(row["mean"].values[0])

        if not model_means:
            continue

        vals = list(model_means.values())
        ax.vlines(x_pos, min(vals), max(vals),
                  color="#BBBBBB", linewidth=2, zorder=1)

        for cid, mean_val in model_means.items():
            jx = rng.uniform(-x_jitter, x_jitter)
            ax.scatter(
                x_pos + jx, mean_val,
                color=CONDITION_COLORS[cid],
                s=dot_size, zorder=3, linewidths=0.5, edgecolors="white",
            )

        provider = _MODEL_META.get(model_id, {}).get("provider", "")
        ax.axvspan(x_pos - 0.42, x_pos + 0.42,
                   color=PROVIDER_COLORS.get(provider, "#EEEEEE"),
                   alpha=0.07, zorder=0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_model_title(m) for m in models], fontsize=9, ha="center")
    ax.set_ylabel("Mean Composite Score (1–10)", fontsize=11)

    all_means = summary["mean"].dropna()
    y_lo = max(0, all_means.min() - 0.5)
    y_hi = min(10.5, all_means.max() + 0.5)
    ax.set_ylim(y_lo, y_hi)

    ax.set_title(
        "Condition Score Range per Model  (composite score, models ordered S→M→L)\n"
        "Each dot = condition mean (~20 API calls)  ·  "
        "Vertical line = min–max range  ·  Wider spread = more demographic variation",
        fontsize=12, pad=10,
    )
    legend_handles = [
        plt.scatter([], [], color=CONDITION_COLORS[c], s=dot_size, label=CONDITION_LABELS[c])
        for c in conditions
    ]
    ax.legend(handles=legend_handles, title="Condition",
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()

    out = outdir / "grouped_bar.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 2. Score heatmap ──────────────────────────────────────────────────────────

def score_heatmap(df: pd.DataFrame, outdir: Path) -> Path:
    df = _ensure_composite(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    summary = compute_summary_stats(df, score_col=score_col)
    model_order = _canonical_model_order(summary["model_id"].unique().tolist())
    conditions = list(CONDITION_LABELS.keys())

    pivot_mean = summary.pivot(index="model_id", columns="condition_id", values="mean")
    pivot_std  = summary.pivot(index="model_id", columns="condition_id", values="std")
    pivot_mean = pivot_mean.reindex(model_order)[conditions]
    pivot_std  = pivot_std.reindex(model_order)[conditions]

    row_labels = [_model_title(m).replace("\n", "  ") for m in model_order]

    fig, ax = plt.subplots(figsize=(12, max(5, len(model_order) * 0.9 + 1.5)))
    sns.heatmap(
        pivot_mean,
        ax=ax,
        cmap="RdYlGn",
        vmin=4, vmax=10,
        annot=False,
        linewidths=0.5,
        linecolor="#cccccc",
        cbar_kws={"label": "Mean composite score"},
    )

    for i, model in enumerate(model_order):
        for j, cond in enumerate(conditions):
            m = pivot_mean.loc[model, cond] if model in pivot_mean.index else None
            s = pivot_std.loc[model, cond] if model in pivot_std.index else None
            if m is not None and not np.isnan(m):
                txt = f"{m:.1f}" if s is None or np.isnan(s) else f"{m:.1f}\n±{s:.1f}"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
    ax.set_xlabel("")
    ax.set_title(
        "Mean Composite Score Heatmap by Model × Demographic Condition\n"
        "Score = mean of 5 dimension sub-scores  ·  Models ordered S→M→L",
        fontsize=12, pad=10,
    )
    fig.tight_layout()
    out = outdir / "score_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 3. Bias delta heatmap ─────────────────────────────────────────────────────

_KEY_CONTRASTS = [
    ("chinese_male",   "malay_male",     "Chinese vs\nMalay (M)"),
    ("chinese_female", "malay_female",   "Chinese vs\nMalay (F)"),
    ("chinese_male",   "indian_male",    "Chinese vs\nIndian (M)"),
    ("chinese_female", "indian_female",  "Chinese vs\nIndian (F)"),
    ("malay_male",     "indian_male",    "Malay vs\nIndian (M)"),
    ("malay_female",   "indian_female",  "Malay vs\nIndian (F)"),
    ("chinese_male",   "chinese_female", "Chinese\nM vs F"),
    ("malay_male",     "malay_female",   "Malay\nM vs F"),
    ("indian_male",    "indian_female",  "Indian\nM vs F"),
]


def bias_delta_heatmap(df: pd.DataFrame, outdir: Path) -> Path:
    df = _ensure_composite(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    model_order = _canonical_model_order(df["model_id"].unique().tolist())

    d_matrix   = np.zeros((len(model_order), len(_KEY_CONTRASTS)))
    sig_matrix = np.zeros_like(d_matrix, dtype=bool)

    mean_tests = run_mean_comparison_tests(df, score_col=score_col)

    for j, (cond_a, cond_b, _) in enumerate(_KEY_CONTRASTS):
        for i, model_id in enumerate(model_order):
            mdf = df[(df["model_id"] == model_id) & df[score_col].notna()]
            x = mdf[mdf["condition_id"] == cond_a][score_col].values
            y = mdf[mdf["condition_id"] == cond_b][score_col].values
            if len(x) and len(y):
                d_matrix[i, j] = cliffs_delta(x, y)
            # significance from Welch's t-test (p < 0.05)
            row = mean_tests[
                (mean_tests["model_id"] == model_id) &
                (mean_tests["condition_a"] == cond_a) &
                (mean_tests["condition_b"] == cond_b)
            ]
            if row.empty:
                row = mean_tests[
                    (mean_tests["model_id"] == model_id) &
                    (mean_tests["condition_a"] == cond_b) &
                    (mean_tests["condition_b"] == cond_a)
                ]
            if not row.empty:
                sig_matrix[i, j] = bool(row.iloc[0]["significant"])

    y_labels = [_model_title(m).replace("\n", "  ") for m in model_order]
    col_labels = [c[2] for c in _KEY_CONTRASTS]
    d_df = pd.DataFrame(d_matrix, index=y_labels, columns=col_labels)

    fig, ax = plt.subplots(figsize=(14, max(5, len(model_order) * 1.2 + 2)))
    sns.heatmap(
        d_df, ax=ax,
        cmap="RdBu",
        center=0, vmin=-0.8, vmax=0.8,
        annot=False,
        linewidths=0.5, linecolor="#cccccc",
        cbar_kws={"label": "Cliff's δ  (positive = row A favoured)"},
    )

    for i in range(len(model_order)):
        for j in range(len(_KEY_CONTRASTS)):
            d_val = d_matrix[i, j]
            txt = f"{d_val:+.2f}"
            if sig_matrix[i, j]:
                txt += "*"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=8, color="black",
                    fontweight="bold" if sig_matrix[i, j] else "normal")

    ax.set_xticklabels(col_labels, rotation=0, fontsize=8)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=8)
    ax.set_title(
        "Demographic Bias — Cliff's δ per Model × Contrast  (composite score)\n"
        "* significant at p < 0.05 (Welch's t-test, no correction)  ·  "
        "Blue = row A scored HIGHER  ·  Red = row A scored LOWER\n"
        "Models ordered S→M→L",
        fontsize=11, pad=12,
    )
    fig.tight_layout()
    out = outdir / "bias_delta_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 4. Summary table ──────────────────────────────────────────────────────────

_EFFECT_COLORS = {
    "negligible":   "#FFFFFF",
    "small":        "#FFF9C4",
    "medium":       "#FFCC80",
    "large":        "#EF9A9A",
    "unmeasurable": "#E3F2FD",
}


def summary_table(df: pd.DataFrame, outdir: Path) -> Path:
    df = _ensure_composite(df)
    score_col = "score_composite" if "score_composite" in df.columns else "score"
    mean_tests = run_mean_comparison_tests(df, score_col=score_col)
    bias = compute_bias_summary(df, mean_tests, score_col=score_col)

    col_labels = [
        "Model", "Provider", "Tier",
        "Ethnic Bias δ", "Ethnic Effect",
        "Gender Bias δ", "Gender Effect",
        "Worst Cell", "# Sig. Pairs",
    ]
    rows = []
    cell_colors = []

    def _fmt_d(val) -> str:
        try:
            f = float(val)
            return "n/a" if np.isnan(f) else f"{f:+.3f}"
        except (TypeError, ValueError):
            return "n/a"

    model_order = _canonical_model_order(bias["model_id"].tolist())
    bias = bias.set_index("model_id").reindex(model_order).reset_index()

    for _, r in bias.iterrows():
        row = [
            r["model_id"], r["provider"], r["tier"],
            _fmt_d(r["ethnic_bias_d"]), r["ethnic_effect"],
            _fmt_d(r["gender_bias_d"]), r["gender_effect"],
            r["worst_cell"], int(r["n_significant"]),
        ]
        colors = [
            "#F5F5F5", "#F5F5F5", "#F5F5F5",
            _EFFECT_COLORS.get(r["ethnic_effect"], "#FFFFFF"), "#F5F5F5",
            _EFFECT_COLORS.get(r["gender_effect"], "#FFFFFF"), "#F5F5F5",
            "#F5F5F5", "#F5F5F5",
        ]
        rows.append(row)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(16, max(3, len(rows) * 0.5 + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#CCCCCC")

    ax.set_title(
        "Bias Summary by Model — Singapore CV Audit  (composite dimension score)",
        fontsize=13, pad=15, fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / "summary_table.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 5. Dimension breakdown heatmap ────────────────────────────────────────────

_DIMENSION_LABELS = {
    "score_technical":     "Technical",
    "score_business":      "Business\nImpact",
    "score_experience":    "Experience",
    "score_communication": "Communication",
    "score_education":     "Education",
}


def dimension_heatmap(df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    """Per-model heatmap: rows=dimensions, cols=conditions, value=mean sub-score.

    Models ordered S→M→L, Google before OpenAI within tier.
    Returns None if no dimension columns are present.
    """
    dim_cols_present = [c for c in _DIMENSION_COLS if c in df.columns and df[c].notna().any()]
    if not dim_cols_present:
        return None

    models = _canonical_model_order(df["model_id"].unique().tolist())
    conditions = list(CONDITION_LABELS.keys())
    n_models = len(models)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models + 1, 5),
        squeeze=False,
    )

    for ax_col, model_id in enumerate(models):
        ax = axes[0][ax_col]
        mdf = df[df["model_id"] == model_id]

        matrix = np.full((len(dim_cols_present), len(conditions)), np.nan)
        for r, dim in enumerate(dim_cols_present):
            for c, cond in enumerate(conditions):
                vals = mdf[mdf["condition_id"] == cond][dim].dropna()
                if len(vals):
                    matrix[r, c] = vals.mean()

        sns.heatmap(
            matrix, ax=ax,
            cmap="RdYlGn", vmin=1, vmax=10,
            annot=True, fmt=".1f", annot_kws={"size": 8},
            linewidths=0.4, linecolor="#cccccc",
            cbar=ax_col == n_models - 1,
            cbar_kws={"label": "Mean score"} if ax_col == n_models - 1 else {},
            xticklabels=[CONDITION_LABELS[c] for c in conditions],
            yticklabels=[_DIMENSION_LABELS[d] for d in dim_cols_present],
        )
        ax.set_title(_model_title(model_id), fontsize=9, pad=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        if ax_col > 0:
            ax.set_ylabel("")

    fig.suptitle(
        "Dimension Sub-Scores by Model × Demographic Condition  (models ordered S→M→L)\n"
        "Red = low score  ·  Green = high score  ·  Bias appears as column variation within a model",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out = outdir / "dimension_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Master generate function ───────────────────────────────────────────────────

def generate_all(df: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = _ensure_composite(df)
    outputs = []
    print("Generating score distributions...")
    outputs.append(score_distributions(df, outdir))
    print("Generating grouped bar chart...")
    outputs.append(grouped_bar(df, outdir))
    print("Generating score heatmap...")
    outputs.append(score_heatmap(df, outdir))
    print("Generating bias delta heatmap...")
    outputs.append(bias_delta_heatmap(df, outdir))
    print("Generating summary table...")
    outputs.append(summary_table(df, outdir))
    print("Generating dimension breakdown heatmap...")
    dim_out = dimension_heatmap(df, outdir)
    if dim_out is not None:
        outputs.append(dim_out)
    else:
        print("  (skipped — no dimension sub-scores in this CSV)")
    print(f"Charts saved to: {outdir}")
    for p in outputs:
        print(f"  {p.name}")
    return outputs
