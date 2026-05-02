from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from openai import AsyncOpenAI

from .base import LLMClient, ScoreResult, _extract_dim

# Structured Outputs guarantee schema compliance — no parse fallback needed
_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "cv_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score_technical":     {"type": "integer"},
                "score_business":      {"type": "integer"},
                "score_experience":    {"type": "integer"},
                "score_communication": {"type": "integer"},
                "score_education":     {"type": "integer"},
                "score":               {"type": "integer"},
                "reasoning":           {"type": "string"},
            },
            "required": [
                "score_technical", "score_business", "score_experience",
                "score_communication", "score_education", "score", "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}


class OpenAIClient(LLMClient):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self._client = AsyncOpenAI()
        # Some models (e.g. reasoning models like gpt-5.5) reject temperature=0;
        # config can set "temperature": None to omit the parameter entirely.
        self._temperature = model_config.get("temperature", 0)

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
            kwargs: dict = dict(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                response_format=_RESPONSE_FORMAT,
                max_completion_tokens=512,
            )
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            response = await self._client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            latency = int((time.monotonic() - start) * 1000)

            data = json.loads(raw)
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
                raw_response=raw,
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
