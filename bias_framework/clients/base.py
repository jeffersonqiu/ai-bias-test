from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreResult:
    model_id: str
    provider: str
    condition_id: str
    trial_number: int
    score: Optional[int]        # None on parse failure or refusal
    reasoning: str
    raw_response: str
    latency_ms: int
    timestamp: str              # ISO 8601
    # Per-dimension sub-scores (None on old-format rows or parse failure)
    score_technical:     Optional[int] = None  # Python/SQL/ML/cloud
    score_business:      Optional[int] = None  # Quantified business impact
    score_experience:    Optional[int] = None  # Years + SG banking domain fit
    score_communication: Optional[int] = None  # CV clarity and specificity
    score_education:     Optional[int] = None  # Degree relevance + institution
    error: Optional[str] = None # "{ExceptionType}: {message}" when failed


def _extract_dim(data: dict, key: str) -> Optional[int]:
    v = data.get(key)
    return int(v) if v is not None else None


class LLMClient(ABC):
    def __init__(self, model_config: dict):
        self.model_id = model_config["id"]
        self.provider = model_config["provider"]
        self.tier = model_config["tier"]

    @abstractmethod
    async def score_cv(
        self,
        system_prompt: str,
        user_message: str,
        condition_id: str,
        trial_number: int,
    ) -> ScoreResult: ...


def _safe_parse_json(raw: str) -> tuple[Optional[int], str]:
    """Extract (score, reasoning) from a raw model response.

    Handles: markdown fences, score-as-float, trailing prose.
    Returns (None, raw[:500]) when parsing fails.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Try the full text first, then fall back to first balanced {...} block
    candidates = [text]
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    else:
        return None, raw[:500]

    score_raw = data.get("score")
    reasoning = str(data.get("reasoning", ""))

    if score_raw is None:
        # Model returned JSON but omitted score — treat as refusal
        return None, reasoning

    try:
        score = int(float(score_raw))
    except (TypeError, ValueError):
        return None, reasoning

    if not (1 <= score <= 10):
        return None, reasoning

    return score, reasoning
