#!/usr/bin/env python3
"""
Entry point for the AI demographic bias experiment.

Usage
-----
python run_experiment.py --dry-run                  # cost estimate, no API calls
python run_experiment.py --mode smoke               # small models only
python run_experiment.py --mode full                # all active models (small + medium)
python run_experiment.py --mode full --resume PATH  # continue from existing CSV
python run_experiment.py --mode full --tiers flagship  # flagship only (Phase 2)
python run_experiment.py --mode full --blind        # PII-removed control run
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _check_keys(models: list[dict]) -> None:
    needed = {m["provider"] for m in models}
    missing = []
    if "openai"    in needed and not os.getenv("OPENAI_API_KEY"):    missing.append("OPENAI_API_KEY")
    if "google"    in needed and not os.getenv("GOOGLE_API_KEY"):     missing.append("GOOGLE_API_KEY")
    if "anthropic" in needed and not os.getenv("ANTHROPIC_API_KEY"): missing.append("ANTHROPIC_API_KEY")
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Copy .env.template → .env and fill in your keys.")
        sys.exit(1)


def _estimate_cost(models: list[dict], n_conditions: int, n_trials: int) -> float:
    from config.settings import MODEL_PRICING
    from bias_framework.cv_template import render_cv
    from bias_framework.rubric import SYSTEM_PROMPT, build_user_message
    from bias_framework.job_posting import JOB_POSTING
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    sample_cv  = render_cv("chinese_male")
    sample_msg = build_user_message(sample_cv, JOB_POSTING)
    input_toks  = len(enc.encode(SYSTEM_PROMPT)) + len(enc.encode(sample_msg))
    output_toks = 250  # conservative estimate

    total = 0.0
    print(f"\nDry-run cost estimate  ({input_toks} input + {output_toks} output tokens/call)")
    print(f"{'Model':<30} {'Calls':>6} {'Cost':>8}")
    print("-" * 48)
    for m in models:
        pricing = MODEL_PRICING.get(m["id"])
        if pricing is None or pricing["input"] is None:
            print(f"  {m['id']:<28} pricing TBD — verify before running")
            continue
        n_calls = n_conditions * n_trials
        cost = (
            input_toks  * n_calls * pricing["input"]  / 1_000_000 +
            output_toks * n_calls * pricing["output"] / 1_000_000
        )
        total += cost
        print(f"  {m['id']:<28} {n_calls:>6}   ${cost:>6.2f}")

    print("-" * 48)
    print(f"  {'TOTAL':<28}          ${total:>6.2f}")
    return total


def _parse_args():
    p = argparse.ArgumentParser(description="AI demographic bias experiment")
    p.add_argument("--mode",   choices=["smoke", "full"], default="full",
                   help="smoke = small models only; full = all active models")
    p.add_argument("--tiers",  nargs="+", choices=["small", "medium", "flagship"],
                   default=None, help="Override which tiers to run (overrides --mode)")
    p.add_argument("--resume", metavar="PATH", help="Path to existing CSV to resume from")
    p.add_argument("--dry-run", action="store_true", help="Print cost estimate and exit")
    p.add_argument("--blind", action="store_true",
                   help="PII-removed control run: replace candidate name with [CANDIDATE]")
    return p.parse_args()


def main():
    args = _parse_args()

    from config import settings

    # Determine model set
    if args.tiers:
        models = [m for m in settings.MODELS if m["tier"] in args.tiers]
    elif args.mode == "smoke":
        models = settings.MODELS_BY_TIER["small"]
    else:
        models = settings.MODELS  # all active (uncommented)

    if not models:
        print("No models selected. Check config/settings.py — flagship or Anthropic rows may still be commented out.")
        sys.exit(1)

    n_conditions = len(settings.DEMOGRAPHIC_CONDITIONS)
    n_trials     = settings.TRIALS_PER_CONDITION

    if args.dry_run:
        total = _estimate_cost(models, n_conditions, n_trials)
        if total > settings.COST_CEILING_USD:
            print(f"\nWARNING: Estimated cost ${total:.2f} exceeds ceiling ${settings.COST_CEILING_USD:.2f}")
        print("\nDry-run only — no API calls made.")
        return

    _check_keys(models)

    # Determine output path
    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            print(f"ERROR: Resume file not found: {output_path}")
            sys.exit(1)
    else:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        prefix = "results_blind" if args.blind else "results"
        output_path = settings.RESULTS_RAW_DIR / f"{prefix}_{ts}.csv"

    from bias_framework import storage, experiment

    completed_keys = storage.load_completed_keys(output_path) if args.resume else set()

    tiers_str = ", ".join(sorted({m["tier"] for m in models}))
    blind_tag = "  [BLIND — name=[CANDIDATE]]" if args.blind else ""
    print(f"Models: {len(models)} ({tiers_str}){blind_tag}")
    print(f"Conditions: {n_conditions}  ·  Trials: {n_trials}  ·  Total calls: {len(models) * n_conditions * n_trials}")

    asyncio.run(
        experiment.run(models, output_path, completed_keys, n_trials, blind=args.blind)
    )

    print(f"\nResults written to: {output_path}")
    print("Next step: python generate_viz.py", output_path, "--out results/charts/")


if __name__ == "__main__":
    main()
