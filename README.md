# Singapore LLM CV Bias Audit

Six AI models evaluated **720 identical CVs**. The only thing that changed was the candidate's name. We found systematic demographic bias — and proved that removing the name eliminates it entirely.

---

## The Result in One Table

| Model | Tier | Sig. pairs (of 15) | Verdict |
|---|---|---|---|
| Gemini 2.5 Flash | Medium | **14/15** | 🔴 Large bias |
| Gemini 3.1 Pro Preview | Flagship | **11/15** | 🔴 Large bias (masked by uniform holistic scores) |
| Gemini 2.5 Flash Lite | Small | **12/15** | 🔴 Large bias (masked by uniform holistic scores) |
| GPT-5.4-mini | Medium | **9/15** | 🔴 Large bias |
| GPT-5.4-nano | Small | **11/15** | 🟡 Moderate bias (Education-concentrated) |
| GPT-5.5 | Flagship | **2/15** | 🟢 Negligible (likely false positives) |

*"Sig. pairs" = condition pairs with statistically different mean composite scores, Welch's t-test p < 0.05, no correction. 20 trials per condition per model.*

---

## The Proof: Remove the Name, Bias Disappears

We ran the identical experiment a second time with every name replaced by `[CANDIDATE]`. Result:

| | Named CVs | Blind CVs (PII removed) |
|---|---|---|
| Significant condition pairs | **59 / 90** | **3 / 90** |
| Education bias | 29 pairs | 0 |
| Business Impact bias | 24 pairs | 0 |
| Experience bias | 15 pairs | 0 |
| Communication bias | 8 pairs | 0 |
| Technical bias | 5 pairs | 0 |

**The name alone drives all observable bias. Redact it before scoring.**

---

## Methodology

- **Counterfactual design** (Bertrand & Mullainathan 2004): one Singapore data analyst CV, six demographic conditions — Chinese/Malay/Indian × Male/Female. Only the name changes.
- **Composite score**: mean of 5 sub-dimension scores (Technical, Business Impact, Experience, Communication, Education). Holistic scores are not used — several models returned uniform holistic scores that mask real sub-dimension variation.
- **Statistics**: Welch's t-test, p < 0.05, all 15 C(6,2) condition pairs per model.
- **Temperature = 0**, 20 trials per cell.

---

## Most Biased Dimension

| Dimension | Sig. pairs across 6 models (max 90) |
|---|---|
| Education | **29** |
| Business Impact | **24** |
| Experience | **15** |
| Communication | 8 |
| Technical | 5 |

Models rate identical education credentials and business impact differently based on the candidate's name. Technical skills are scored most consistently.

---

## How to Run

```bash
# Setup
cp .env.template .env   # fill in OPENAI_API_KEY, GOOGLE_API_KEY
uv sync

# Dry-run (cost estimate only)
uv run python run_experiment.py --dry-run

# Named run — ~$3 for all 6 models
uv run python run_experiment.py --mode full

# Blind run (PII-removed control) — same cost
uv run python run_experiment.py --mode full --blind

# Charts
uv run python generate_viz.py results/raw/<file>.csv --out results/charts/

# Notebook
uv run --extra notebook jupyter nbconvert --to notebook --execute --inplace \
  analysis/singapore_llm_bias_analysis.ipynb
```

---

## Repository Layout

```
bias_framework/       # CV template, rubric, async experiment runner, stats, viz
├── clients/          # One LLM client per provider (OpenAI, Google, Anthropic)
config/settings.py    # Models, conditions, pricing, concurrency
run_experiment.py     # CLI entry point (--mode, --blind, --resume, --dry-run)
generate_viz.py       # Chart generation from any results CSV
analysis/             # Executed research notebook + critique reports
personas/             # Data Scientist, RAI Officer, Product Lead, HR — review lenses
tests/                # CV render, stats math, client mocks
results/raw/          # Timestamped CSVs (gitignored)
results/charts/       # PNG charts (gitignored)
```

---

## Key Findings

1. **Google models are biased at every tier.** Flash Lite (small), Flash (medium), and 3.1 Pro (flagship) all show large bias. Tier does not predict fairness within the Google lineup.
2. **Holistic scores can hide bias.** Three Google models returned identical holistic scores across all conditions. Sub-dimension composite scores reveal the real picture.
3. **GPT-5.5 is the exception.** The only model with no meaningful demographic bias. 2/15 pairs barely significant — consistent with the false-positive floor at uncorrected p < 0.05.
4. **Education is the most sensitive dimension.** 29/90 condition pairs are statistically significant for Education scores — models appear to rate identical credentials differently based on name-signalled background.
5. **PII redaction works.** Replacing the name with `[CANDIDATE]` reduces significant pairs from 59 to 3 (95% reduction), with 100% reduction across all five sub-dimensions.

---

## Implications for Singapore Hiring

Under the **Fair Consideration Framework**, using an AI tool that produces demographically differential outcomes in shortlisting may constitute a violation — regardless of intent.

**Practical fix:** redact candidate names before any AI scoring step. This study proves it works.

---

## Citations

- Bertrand & Mullainathan (2004). *Are Emily and Greg More Employable?* AER 94(4).
- Full analysis: `analysis/singapore_llm_bias_analysis.ipynb`
