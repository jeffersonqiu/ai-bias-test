"""Verify all 6 CV variants render cleanly and prompts stay within token budget."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bias_framework.cv_template import render_cv, CV_TEMPLATE
from bias_framework.rubric import SYSTEM_PROMPT, build_user_message
from bias_framework.job_posting import JOB_POSTING
from config.settings import DEMOGRAPHIC_CONDITIONS


def test_all_conditions_render():
    for cid, meta in DEMOGRAPHIC_CONDITIONS.items():
        cv = render_cv(cid)
        assert meta["name"] in cv, f"{cid}: expected name '{meta['name']}' not found in CV"
        assert "{name}" not in cv, f"{cid}: unfilled {{name}} slot found"
        assert "applicant@proton.me" in cv, f"{cid}: contact info missing"
        assert "NTU" in cv, f"{cid}: university missing"
        print(f"  ✓ {cid}: '{meta['name']}'")


def test_all_cvs_are_same_except_name():
    rendered = {cid: render_cv(cid) for cid in DEMOGRAPHIC_CONDITIONS}
    # Strip the name line and compare; remainder should be identical
    def strip_name(text: str, name: str) -> str:
        return text.replace(name, "NAME_PLACEHOLDER")

    stripped = {
        cid: strip_name(cv, DEMOGRAPHIC_CONDITIONS[cid]["name"])
        for cid, cv in rendered.items()
    }
    first = next(iter(stripped.values()))
    for cid, cv in stripped.items():
        assert cv == first, f"{cid}: CV body differs from baseline after stripping name"
    print("  ✓ All CVs identical except candidate name")


def test_prompt_token_budget():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except ImportError:
        print("  ⚠ tiktoken not installed; skipping token count test")
        return

    sample_cv  = render_cv("chinese_male")
    user_msg   = build_user_message(sample_cv, JOB_POSTING)
    sys_tokens  = len(enc.encode(SYSTEM_PROMPT))
    user_tokens = len(enc.encode(user_msg))
    total = sys_tokens + user_tokens

    print(f"  System prompt: {sys_tokens} tokens")
    print(f"  User message:  {user_tokens} tokens")
    print(f"  Total input:   {total} tokens")
    assert total < 3000, f"Prompt too long: {total} tokens (limit 3000)"
    print("  ✓ Token budget OK")


if __name__ == "__main__":
    print("\n=== CV render tests ===")
    test_all_conditions_render()
    test_all_cvs_are_same_except_name()
    test_prompt_token_budget()
    print("\nAll tests passed.")
