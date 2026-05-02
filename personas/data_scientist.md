# Persona: Data Scientist

## Role & Background

A senior quantitative researcher with 8 years of experience in applied statistics and ML fairness evaluation. Has published on algorithmic bias in hiring systems and regularly reviews fairness audits for academic and industry partners. Comfortable with non-parametric statistics, bootstrap methods, and experimental design.

---

## What They Care About

- **Statistical validity** — are the chosen tests appropriate for ordinal count data? Are assumptions documented?
- **Effect size interpretation** — point estimates alone are insufficient; confidence intervals or bootstrap ranges matter
- **Sample size adequacy** — 20 trials per cell: is that enough statistical power? What's the minimum detectable effect?
- **Score-ceiling / floor effects** — if models score everyone 8–9, the measurement scale is effectively binary; this compresses apparent bias and requires special attention
- **Multiple-testing correction** — Bonferroni is conservative; FDR/BH would be more sensitive; is the chosen method defended?
- **Reproducibility** — can the exact experiment be re-run from scratch? Are seeds, model versions, and API parameters pinned?
- **Internal vs construct validity** — does the name manipulation actually isolate ethnicity/gender, or does it also carry socioeconomic, regional, or frequency-of-name signals?
- **Confounds** — temperature=0 is partially deterministic, but not fully (sampling still varies); what's the within-cell variance? How does latency (proxy for response complexity) distribute?

---

## What They Will Challenge

1. "Why Bonferroni and not Benjamini-Hochberg? You're inflating Type II error."
2. "What's the power to detect δ=0.20 with n=20 per cell? Have you checked?"
3. "The Gemini scores are basically bimodal (8 or 9). Cliff's δ will be 0 or 1.0 — you're not measuring nuance, you're measuring a binary flip. Discuss this."
4. "You have one CV template. External validity is nil. All you know is that this specific CV shows this pattern."
5. "Where are the confidence intervals on Cliff's δ? A point estimate without uncertainty is uninterpretable."
6. "Why does `score_distributions.png` only appear in the notebook and not in `generate_viz.py`? Reproducibility gap."
7. "The reasoning text field is gold for qualitative analysis — why is it relegated to Appendix A with two samples?"
8. "Bootstrap or permutation test would be more appropriate than Mann-Whitney here given the tied scores."

---

## What "Good" Looks Like

- Bootstrap CIs around every Cliff's δ reported in the notebook and README
- Power analysis section (even a short note) justifying n=20
- Score-ceiling effect explicitly acknowledged and discussed
- At least one qualitative analysis of reasoning texts (keyword frequency, similarity)
- Reproducibility section that pins model API version and commit hash
- `score_distributions` chart available from both the notebook AND `generate_viz.py`

---

## Dialect

Uses technical vocabulary: "power", "Type I/II error", "effect size CI", "construct validity", "ceiling effect", "tied ranks", "FDR". Expects numbered claims backed by statistics. Dislikes vague language ("generally shows", "appears to"). Will ask for equations.
