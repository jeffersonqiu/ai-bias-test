from __future__ import annotations

import asyncio
from pathlib import Path

from config import settings
from bias_framework.clients.base import ScoreResult
from bias_framework.clients.openai_client import OpenAIClient
from bias_framework.clients.gemini_client import GeminiClient
from bias_framework.clients.anthropic_client import AnthropicClient
from bias_framework.cv_template import render_cv
from bias_framework.rubric import SYSTEM_PROMPT, build_user_message
from bias_framework.job_posting import JOB_POSTING
from bias_framework import storage

_PROVIDER_CLIENT = {
    "openai":    OpenAIClient,
    "google":    GeminiClient,
    "anthropic": AnthropicClient,
}

_RATE_LIMIT_KEYWORDS = {"429", "resource_exhausted", "quota", "resource exhausted"}
_RETRYABLE_KEYWORDS  = _RATE_LIMIT_KEYWORDS | {"rate limit", "ratelimit", "timeout", "connection", "serviceexception", "overloaded"}


async def _with_retry(
    client,
    system_prompt: str,
    user_message: str,
    condition_id: str,
    trial_number: int,
) -> ScoreResult:
    for attempt in range(settings.RETRY_ATTEMPTS):
        result = await client.score_cv(system_prompt, user_message, condition_id, trial_number)
        if result.score is not None:
            return result
        err = (result.error or "").lower()
        is_retryable = any(k in err for k in _RETRYABLE_KEYWORDS)
        if is_retryable and attempt < settings.RETRY_ATTEMPTS - 1:
            base = 20.0 if any(k in err for k in _RATE_LIMIT_KEYWORDS) else settings.RETRY_BASE_DELAY
            await asyncio.sleep(base * (2 ** attempt))
        else:
            return result
    return result


async def run(
    models: list[dict],
    output_path: Path,
    completed_keys: set,
    trials_per_condition: int = settings.TRIALS_PER_CONDITION,
    blind: bool = False,
) -> None:
    conditions = list(settings.DEMOGRAPHIC_CONDITIONS.keys())

    # Pre-render all user messages; in blind mode every CV uses "[CANDIDATE]"
    user_messages = {
        cid: build_user_message(render_cv(cid, blind=blind), JOB_POSTING)
        for cid in conditions
    }
    system_prompt = SYSTEM_PROMPT

    # Build clients
    clients = {m["id"]: _PROVIDER_CLIENT[m["provider"]](m) for m in models}

    # Per-provider semaphores
    semaphores = {
        provider: asyncio.Semaphore(limit)
        for provider, limit in settings.CONCURRENCY.items()
    }

    # Build pending task list (skip already-completed keys)
    pending = [
        (model, condition_id, trial_num)
        for model in models
        for condition_id in conditions
        for trial_num in range(1, trials_per_condition + 1)
        if (model["id"], condition_id, trial_num) not in completed_keys
    ]

    total = len(pending)
    skipped = (len(models) * len(conditions) * trials_per_condition) - total
    if skipped:
        print(f"Resuming: skipping {skipped} already-completed trials.")
    if total == 0:
        print("All trials already completed.")
        return

    print(f"Running {total} trials → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_count = 0
    errors = 0

    async def _run_one(model_cfg: dict, condition_id: str, trial_num: int) -> ScoreResult:
        nonlocal completed_count, errors
        sem = semaphores[model_cfg["provider"]]
        client = clients[model_cfg["id"]]

        async with sem:
            result = await _with_retry(
                client, system_prompt, user_messages[condition_id], condition_id, trial_num
            )
            # Per-provider pacing (currently 0 across the board on paid tiers)
            delay = settings.REQUEST_DELAY.get(model_cfg["provider"], 0.0)
            if delay > 0:
                await asyncio.sleep(delay)

        storage.append_result(result, output_path)
        completed_count += 1
        if result.score is None:
            errors += 1
        status = f"score={result.score}" if result.score else f"ERR {result.error or ''}"[:40]
        print(
            f"\r[{completed_count}/{total}] "
            f"{result.model_id}/{condition_id}/t{trial_num} → {status}   ",
            end="",
            flush=True,
        )
        return result

    tasks = [_run_one(m, c, t) for m, c, t in pending]
    await asyncio.gather(*tasks)
    print()
    success = total - errors
    print(f"Done: {success}/{total} successful  |  {errors} errors/refusals")
    if errors:
        print(f"  → Check {output_path} — rows with score='' indicate failures.")
