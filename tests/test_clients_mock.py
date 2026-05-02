"""Verify each provider client correctly parses its own response shape."""
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bias_framework.clients.base import _safe_parse_json


# ── _safe_parse_json edge cases ───────────────────────────────────────────────

def test_parse_clean_json():
    s, r = _safe_parse_json('{"score": 8, "reasoning": "Strong CV."}')
    assert s == 8
    assert r == "Strong CV."


def test_parse_markdown_fence():
    raw = '```json\n{"score": 7, "reasoning": "Good."}\n```'
    s, r = _safe_parse_json(raw)
    assert s == 7


def test_parse_float_score():
    s, r = _safe_parse_json('{"score": 8.0, "reasoning": "Good."}')
    assert s == 8


def test_parse_trailing_text():
    raw = '{"score": 9, "reasoning": "Excellent."} Some extra text here.'
    s, r = _safe_parse_json(raw)
    assert s == 9


def test_parse_out_of_range():
    s, r = _safe_parse_json('{"score": 11, "reasoning": "Too high."}')
    assert s is None


def test_parse_missing_score():
    s, r = _safe_parse_json('{"reasoning": "I cannot score based on names."}')
    assert s is None


def test_parse_garbage():
    s, r = _safe_parse_json("I refuse to evaluate this candidate.")
    assert s is None


# ── OpenAI client mock ────────────────────────────────────────────────────────

def test_openai_client_parses_response():
    import json
    from bias_framework.clients.openai_client import OpenAIClient

    model_cfg = {"id": "gpt-5.4-mini", "provider": "openai", "tier": "small"}

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"score": 8, "reasoning": "Strong data background."})
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    async def run():
        with patch("bias_framework.clients.openai_client.AsyncOpenAI") as mock_cls:
            mock_api = MagicMock()
            mock_api.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_api
            client = OpenAIClient(model_cfg)
            result = await client.score_cv("sys", "user", "chinese_male", 1)
        assert result.score == 8, f"Expected 8, got {result.score}"
        assert result.error is None
        print("  ✓ OpenAI client parse test passed")

    asyncio.run(run())


# ── Anthropic client mock ─────────────────────────────────────────────────────

def test_anthropic_client_parses_tool_use():
    from bias_framework.clients.anthropic_client import AnthropicClient

    model_cfg = {"id": "claude-haiku-4-5-20251001", "provider": "anthropic", "tier": "small"}

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.input = {"score": 7, "reasoning": "Solid candidate."}
    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]

    async def run():
        with patch("bias_framework.clients.anthropic_client.anthropic.AsyncAnthropic") as mock_cls:
            mock_api = MagicMock()
            mock_api.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_api
            client = AnthropicClient(model_cfg)
            result = await client.score_cv("sys", "user", "malay_female", 1)
        assert result.score == 7, f"Expected 7, got {result.score}"
        assert result.error is None
        print("  ✓ Anthropic client parse test passed")

    asyncio.run(run())


def test_anthropic_client_missing_tool_block():
    """Model returns no tool_use block — should result in score=None with error."""
    from bias_framework.clients.anthropic_client import AnthropicClient

    model_cfg = {"id": "claude-haiku-4-5-20251001", "provider": "anthropic", "tier": "small"}

    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    async def run():
        with patch("bias_framework.clients.anthropic_client.anthropic.AsyncAnthropic") as mock_cls:
            mock_api = MagicMock()
            mock_api.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_api
            client = AnthropicClient(model_cfg)
            result = await client.score_cv("sys", "user", "indian_male", 1)
        assert result.score is None
        assert result.error is not None
        print("  ✓ Anthropic refusal/no-tool-block test passed")

    asyncio.run(run())


# ── Gemini client mock ────────────────────────────────────────────────────────

def test_gemini_client_parses_response():
    import json
    from bias_framework.clients.gemini_client import GeminiClient

    model_cfg = {"id": "gemini-2.5-flash-lite", "provider": "google", "tier": "small"}

    async def run():
        with patch("bias_framework.clients.gemini_client.genai.Client"):
            client = GeminiClient(model_cfg)
            mock_text = json.dumps({"score": 9, "reasoning": "Exceptional candidate."})
            with patch.object(client, "_sync_generate", return_value=mock_text):
                result = await client.score_cv("sys", "user", "chinese_female", 1)
        assert result.score == 9, f"Expected 9, got {result.score}"
        assert result.error is None
        print("  ✓ Gemini client parse test passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Client mock tests ===")
    test_parse_clean_json()
    test_parse_markdown_fence()
    test_parse_float_score()
    test_parse_trailing_text()
    test_parse_out_of_range()
    test_parse_missing_score()
    test_parse_garbage()
    test_openai_client_parses_response()
    test_anthropic_client_parses_tool_use()
    test_anthropic_client_missing_tool_block()
    test_gemini_client_parses_response()
    print("\nAll client mock tests passed.")
