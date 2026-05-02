# CLAUDE.md — ai-bias-test

## What this project is

An LLM demographic bias audit framework using the counterfactual CV methodology (Bertrand & Mullainathan 2004). Identical CVs for a Singapore data analyst role are scored by multiple LLMs; only the candidate's name changes across 6 demographic conditions (Chinese/Malay/Indian × Male/Female). Statistical analysis detects whether models assign systematically different scores based on name-signalled demographic group.

---

## Architecture map

```
bias_framework/
├── cv_template.py        # CV_TEMPLATE (str) + render_cv(condition_id) → str
├── job_posting.py        # JOB_POSTING (str) — the Singapore data analyst JD
├── rubric.py             # SYSTEM_PROMPT + build_user_message(cv, job_posting) → str
├── experiment.py         # async run() — orchestrates all models × conditions × trials
├── storage.py            # append_result(), load_results(), load_completed_keys()
├── stats.py              # cliffs_delta, mann_whitney_p, run_pairwise_tests, compute_bias_summary
├── visualization.py      # 6 chart-generating functions + generate_all()
└── clients/
    ├── base.py           # ScoreResult dataclass, LLMClient ABC, _extract_dim, _safe_parse_json
    ├── openai_client.py  # JSON-schema mode (response_format)
    ├── gemini_client.py  # response_schema via ThreadPoolExecutor bridge (SDK is sync)
    └── anthropic_client.py  # tool-use forcing (submit_cv_score tool)
config/
├── settings.py           # MODELS, DEMOGRAPHIC_CONDITIONS, concurrency, pricing, paths
analysis/
├── singapore_llm_bias_analysis.ipynb  # executed research notebook
└── critique/             # skill-generated critique reports
personas/
├── data_scientist.md
├── responsible_ai_officer.md
├── product_lead.md
└── hr_business_user.md
results/
├── raw/                  # timestamped CSVs from run_experiment.py
└── charts/               # PNGs from generate_viz.py / notebook
```

Data flow: `run_experiment.py → experiment.run() → clients/*.score_cv() → storage.append_result() → CSV`
Analysis flow: `generate_viz.py <csv> / notebook → storage.load_results() → stats.* → visualization.* → PNGs`

---

## Run commands

```bash
# Setup
cp .env.template .env        # fill in API keys
uv sync                       # install deps (includes notebook extras)
uv sync --extra notebook      # if notebook deps not yet installed

# Experiment
uv run python run_experiment.py --dry-run
uv run python run_experiment.py --mode smoke          # small models only
uv run python run_experiment.py --mode full           # all active models
uv run python run_experiment.py --mode full --resume results/raw/<file>.csv
uv run python run_experiment.py --mode full --blind           # PII-removed control run → results_blind_*.csv

# Visualisations (CLI)
uv run python generate_viz.py results/raw/<file>.csv --out results/charts/

# Tests
uv run python tests/test_cv_render.py
uv run python tests/test_stats.py
uv run python tests/test_clients_mock.py

# Notebook (execute in-place, saves outputs to .ipynb)
uv run --extra notebook jupyter nbconvert --to notebook --execute --inplace \
  analysis/singapore_llm_bias_analysis.ipynb
```

---

## Key invariants

- **Name is the only variable** — every other CV field (education, experience, skills, bullet points) is identical across all 6 conditions.
- **Temperature = 0** — hardcoded in each client; partial determinism by design. Minor variance from token sampling is expected and measured.
- **20 trials per (model × condition) cell** — each row in the CSV is one trial.
- **Multi-dimensional rubric** — 5 dimensions scored first (Technical, Business Impact, Experience, Communication, Education), then the holistic overall score. The anti-anchoring order matters; do not change it mid-experiment.
- **Paid-tier API keys assumed** — all concurrency settings are calibrated for paid tiers (5 concurrent for OpenAI and Google).
- **Canonical CSV** — always `sorted(glob("results/raw/results_*.csv"))[-1]`. Never delete the latest CSV.

---

## Data schema

CSV columns (in order):
```
model_id, provider, condition_id, trial_number,
score, score_technical, score_business, score_experience, score_communication, score_education,
reasoning, raw_response, latency_ms, timestamp, error
```

- `score` is `None` (blank) on error/refusal rows
- `raw_response` is truncated to 2000 chars
- All dimension scores (`score_*`) will be `None` on error rows

---

## What's currently active (as of May 2026)

| Provider | Model | Tier | Status |
|---|---|---|---|
| OpenAI | `gpt-5.4-nano` | small | ✅ Active |
| OpenAI | `gpt-5.4-mini` | medium | ✅ Active |
| Google | `gemini-2.5-flash-lite` | small | ✅ Active |
| Google | `gemini-2.5-flash` | medium | ✅ Active |
| Anthropic | `claude-haiku-4-5-20251001` | small | 🔇 Commented out |
| Anthropic | `claude-sonnet-4-6` | medium | 🔇 Commented out |
| OpenAI | `gpt-5.5` | flagship | 🔇 Commented out (Phase 2) |
| Google | `gemini-2.5-pro` | flagship | 🔇 Commented out (Phase 2) |
| Anthropic | `claude-opus-4-7` | flagship | 🔇 Commented out (Phase 2) |

To enable a model, uncomment its row in `config/settings.py → MODELS`.

---

## Retry / backoff logic

`experiment._with_retry()` retries up to `RETRY_ATTEMPTS=3` on retryable errors.

- **Rate-limit 429**: base delay 20s, exponential backoff (20s, 40s)
- **Other retryable** (timeout, connection, 503): base delay 1s, exponential (1s, 2s)
- Non-retryable errors (auth, billing, refusal): no retry, error saved to CSV

---

## Testing

| File | Covers |
|---|---|
| `tests/test_cv_render.py` | 6 CV variants render correctly; name is the only diff; token budget OK |
| `tests/test_stats.py` | Cliff's δ math, effect labels, Mann-Whitney, pairwise direction, bias summary |
| `tests/test_clients_mock.py` | `_safe_parse_json` edge cases; OpenAI / Anthropic / Gemini happy-path mocks; Anthropic refusal path |

**Not covered:** retry/backoff logic, CSV write semantics, resume behaviour, chart generation.
All tests run as plain scripts: `python tests/<name>.py`. They also work with `pytest`.

---

## Don'ts

- **Do not merge CSVs from different rubric versions** — old CSV (before multi-dim scoring) has no `score_*` columns; it cannot be combined with current-schema CSVs.
- **Do not change the rubric mid-experiment** — alters what's being measured; start a fresh CSV if you change `SYSTEM_PROMPT`.
- **Do not commit `.env`** — it's in `.gitignore`; use `.env.template` as the reference.
- **Do not hard-code chart output paths** in new analysis code — use `RESULTS_RAW_DIR` from `config.settings` or pass `outdir` as a parameter.
- **Do not add `TEMPERATURE` to settings** — it was removed because each client hard-codes 0; adding it back creates a false impression it's wired up.
