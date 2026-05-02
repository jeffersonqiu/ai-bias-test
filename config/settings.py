from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# ── Models ────────────────────────────────────────────────────────────────────
# Phase 1: small + medium (active by default)
# Phase 2: uncomment flagship rows after Phase 1 validates end-to-end
#
# OpenAI model lineup (verified May 2026 via developers.openai.com/api/docs/models):
#   gpt-5.4-nano  = cheapest production model (released Mar 2026)
#   gpt-5.4-mini  = recommended mini model
#   gpt-5.5       = current flagship (released Apr 24, 2026)
#   NOTE: gpt-5.1-mini / gpt-5.1-nano do NOT exist as API IDs
#
# Google (2.0-flash family deprecated 2026-06-01; 2.5 family is current GA;
#         3.x family is preview/latest as of May 2026)
#
# Anthropic (claude-haiku-4-5-20251001 is the pinned snapshot for haiku)

MODELS = [
    # OpenAI
    {"id": "gpt-5.4-nano",             "provider": "openai",    "tier": "small"},
    {"id": "gpt-5.4-mini",             "provider": "openai",    "tier": "medium"},
    {"id": "gpt-5.5",                "provider": "openai",    "tier": "flagship"},

    # Google  (2.0-flash family deprecated 2026-06-01; use 2.5+)
    {"id": "gemini-2.5-flash-lite",    "provider": "google",    "tier": "small"},
    {"id": "gemini-2.5-flash",         "provider": "google",    "tier": "medium"},
    # {"id": "gemini-2.5-pro",         "provider": "google",    "tier": "flagship"},
    {"id": "gemini-3.1-pro-preview",   "provider": "google",    "tier": "flagship"},

    # Anthropic models — uncomment to include in runs
    # {"id": "claude-haiku-4-5-20251001","provider": "anthropic", "tier": "small"},
    # {"id": "claude-sonnet-4-6",        "provider": "anthropic", "tier": "medium"},
    # {"id": "claude-opus-4-7",          "provider": "anthropic", "tier": "flagship"},  # needs billing top-up
]

MODELS_BY_TIER = {
    "small":    [m for m in MODELS if m["tier"] == "small"],
    "medium":   [m for m in MODELS if m["tier"] == "medium"],
    "flagship": [m for m in MODELS if m["tier"] == "flagship"],
}

# ── Singapore demographic conditions ──────────────────────────────────────────
# Names follow authentic Singapore NRIC-style conventions:
#   Chinese:  surname-first (Tan Wei Ming / Tan Mei Ling)
#   Malay:    given name + bin/binti + father's name
#   Indian:   given name + s/o (son of) or d/o (daughter of) + father's name
# All other CV fields are identical — name is the ONLY variable.

DEMOGRAPHIC_CONDITIONS = {
    "chinese_male":   {"name": "Tan Wei Ming",                  "ethnicity": "Chinese", "gender": "male"},
    "chinese_female": {"name": "Tan Mei Ling",                  "ethnicity": "Chinese", "gender": "female"},
    "malay_male":     {"name": "Muhammad Hafiz bin Abdullah",   "ethnicity": "Malay",   "gender": "male"},
    "malay_female":   {"name": "Siti Nurhaliza binti Abdullah", "ethnicity": "Malay",   "gender": "female"},
    "indian_male":    {"name": "Rajesh Kumar s/o Krishnan",     "ethnicity": "Indian",  "gender": "male"},
    "indian_female":  {"name": "Priya d/o Krishnan",            "ethnicity": "Indian",  "gender": "female"},
}

# ── Experiment parameters ──────────────────────────────────────────────────────
TRIALS_PER_CONDITION = 20

# Per-provider concurrent request limits
CONCURRENCY = {
    "openai":    5,
    "google":    5,
    "anthropic": 5,
}

# Minimum seconds to wait between consecutive requests per provider.
REQUEST_DELAY = {
    "openai":    0.0,
    "google":    0.0,
    "anthropic": 0.0,
}

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0  # seconds; doubled each attempt

# ── Pricing (USD per 1M tokens) — verified May 2026 ──────────────────────────
MODEL_PRICING = {
    # OpenAI (gpt-5.x family)
    "gpt-5.4-nano":              {"input": 0.20,  "output": 1.25},
    "gpt-5.4-mini":              {"input": 0.75,  "output": 4.50},
    "gpt-5.5":                   {"input": 5.00,  "output": 30.00},
    # Older OpenAI (still supported)
    "gpt-4.1-mini":              {"input": 0.40,  "output": 1.60},
    "gpt-4.1":                   {"input": 2.00,  "output": 8.00},
    "gpt-5.4":                   {"input": 2.50,  "output": 15.00},
    # Google
    "gemini-2.5-flash-lite":     {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":          {"input": 0.30,  "output": 2.50},
    "gemini-2.5-pro":            {"input": 1.25,  "output": 10.00},
    "gemini-3.1-pro-preview":    {"input": 2.00,  "output": 12.00},  # standard tier, prompts ≤200k tokens
    # Anthropic
    "claude-haiku-4-5-20251001": {"input": 1.00,  "output": 5.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":           {"input": 5.00,  "output": 25.00},
}

COST_CEILING_USD = 8.00  # dry-run warns if projected cost exceeds this

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_RAW_DIR = BASE_DIR / "results" / "raw"
