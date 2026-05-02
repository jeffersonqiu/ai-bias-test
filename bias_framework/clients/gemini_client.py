from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from google import genai
from google.genai import types

from .base import LLMClient, ScoreResult, _extract_dim

# google-genai SDK is synchronous — bridge to asyncio via a thread pool.
_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="gemini")

_SCORE_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "score_technical":     types.Schema(type="INTEGER"),
        "score_business":      types.Schema(type="INTEGER"),
        "score_experience":    types.Schema(type="INTEGER"),
        "score_communication": types.Schema(type="INTEGER"),
        "score_education":     types.Schema(type="INTEGER"),
        "score":               types.Schema(type="INTEGER"),
        "reasoning":           types.Schema(type="STRING"),
    },
    required=[
        "score_technical", "score_business", "score_experience",
        "score_communication", "score_education", "score", "reasoning",
    ],
)


class GeminiClient(LLMClient):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self._client = genai.Client()

    def _sync_generate(self, system_prompt: str, user_message: str) -> str:
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=_SCORE_SCHEMA,
                temperature=0.0,
            ),
        )
        return response.text

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
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                _executor, self._sync_generate, system_prompt, user_message
            )
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
