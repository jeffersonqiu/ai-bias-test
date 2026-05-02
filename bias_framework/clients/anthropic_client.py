from __future__ import annotations

import time
from datetime import datetime, timezone

import anthropic

from .base import LLMClient, ScoreResult, _extract_dim

# Anthropic has no JSON mode — enforce structure via tool-use with forced calling.
_SCORE_TOOL = {
    "name": "submit_cv_score",
    "description": "Submit the CV evaluation result with per-dimension and overall scores.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score_technical": {
                "type": "integer",
                "description": "Technical skills score 1–10 (Python, SQL, ML, cloud)",
                "minimum": 1, "maximum": 10,
            },
            "score_business": {
                "type": "integer",
                "description": "Business impact score 1–10 (quantified outcomes, revenue/cost/risk)",
                "minimum": 1, "maximum": 10,
            },
            "score_experience": {
                "type": "integer",
                "description": "Relevant experience score 1–10 (years + SG banking domain fit)",
                "minimum": 1, "maximum": 10,
            },
            "score_communication": {
                "type": "integer",
                "description": "Communication quality score 1–10 (CV clarity and specificity)",
                "minimum": 1, "maximum": 10,
            },
            "score_education": {
                "type": "integer",
                "description": "Education score 1–10 (degree relevance and institution)",
                "minimum": 1, "maximum": 10,
            },
            "score": {
                "type": "integer",
                "description": "Holistic overall score 1–10 (not a mechanical average)",
                "minimum": 1, "maximum": 10,
            },
            "reasoning": {
                "type": "string",
                "description": "1–2 sentence explanation of the overall score",
            },
        },
        "required": [
            "score_technical", "score_business", "score_experience",
            "score_communication", "score_education", "score", "reasoning",
        ],
    },
}


class AnthropicClient(LLMClient):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self._client = anthropic.AsyncAnthropic()

    async def score_cv(
        self,
        system_prompt: str,
        user_message: str,
        condition_id: str,
        trial_number: int,
    ) -> ScoreResult:
        start = time.monotonic()
        ts = datetime.now(timezone.utc).isoformat()
        try:
            response = await self._client.messages.create(
                model=self.model_id,
                max_tokens=512,
                # cache_control on the system prompt caches the rubric across all
                # trials for the same model — saves ~90% of input-token cost
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
                tools=[_SCORE_TOOL],
                tool_choice={"type": "tool", "name": "submit_cv_score"},
                temperature=0,
            )
            latency = int((time.monotonic() - start) * 1000)

            tool_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if tool_block is None:
                raise ValueError("No tool_use block in response")

            data = tool_block.input
            score = int(data["score"])
            reasoning = str(data.get("reasoning", ""))

            if not (1 <= score <= 10):
                raise ValueError(f"Score {score!r} out of 1–10 range")

            return ScoreResult(
                model_id=self.model_id,
                provider=self.provider,
                condition_id=condition_id,
                trial_number=trial_number,
                score=score,
                reasoning=reasoning,
                raw_response=str(data),
                latency_ms=latency,
                timestamp=ts,
                score_technical=_extract_dim(data, "score_technical"),
                score_business=_extract_dim(data, "score_business"),
                score_experience=_extract_dim(data, "score_experience"),
                score_communication=_extract_dim(data, "score_communication"),
                score_education=_extract_dim(data, "score_education"),
            )
        except Exception as exc:
            return ScoreResult(
                model_id=self.model_id,
                provider=self.provider,
                condition_id=condition_id,
                trial_number=trial_number,
                score=None,
                reasoning="",
                raw_response="",
                latency_ms=int((time.monotonic() - start) * 1000),
                timestamp=ts,
                error=f"{type(exc).__name__}: {exc}",
            )
