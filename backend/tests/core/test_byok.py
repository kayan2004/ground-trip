"""Coverage for app/core/byok.py - the settings-copy mechanism that lets a
BYOK request use its own provider/model/key without ever mutating the
process-global Settings singleton returned by get_settings().
"""

import asyncio

import pytest

from app.core.byok import BYOKOverride, BYOKValidationError, build_byok_settings
from app.core.config import get_settings


def test_build_byok_settings_applies_override_without_mutating_base():
    base = get_settings()
    original_provider = base.llm_provider
    original_gemini_key = base.gemini.api_key

    override = BYOKOverride(
        provider="gemini", model="gemini-3.1-flash-lite", api_key="user-supplied-key"
    )
    copy = build_byok_settings(base, override)

    assert copy.llm_provider == "gemini"
    assert copy.gemini.api_key == "user-supplied-key"
    assert copy.gemini.model == "gemini-3.1-flash-lite"

    # The shared singleton must be untouched.
    assert base.llm_provider == original_provider
    assert base.gemini.api_key == original_gemini_key
    assert base is not copy
    assert base.gemini is not copy.gemini


def test_build_byok_settings_rejects_non_allowlisted_combo():
    base = get_settings()
    override = BYOKOverride(provider="gemini", model="gemini-3.1-pro-preview", api_key="key")

    with pytest.raises(BYOKValidationError):
        build_byok_settings(base, override)


def test_build_byok_settings_supports_every_allowlisted_provider():
    base = get_settings()

    anthropic_copy = build_byok_settings(
        base, BYOKOverride(provider="anthropic", model="claude-haiku-4-5", api_key="a-key")
    )
    assert anthropic_copy.anthropic.api_key == "a-key"
    assert anthropic_copy.llm_provider == "anthropic"

    openai_copy = build_byok_settings(
        base, BYOKOverride(provider="openai", model="gpt-5.4-nano", api_key="o-key")
    )
    assert openai_copy.openai.api_key == "o-key"
    assert openai_copy.llm_provider == "openai"


@pytest.mark.asyncio(loop_scope="session")
async def test_build_byok_settings_concurrent_calls_do_not_cross_contaminate():
    """The exact bug class the BYOK spec called out: mutating the shared
    Settings singleton in place would leak one caller's key into every
    other concurrent request. Two concurrent build_byok_settings() calls
    with different keys must each get their own independent object.
    """
    base = get_settings()

    async def build(key: str, model: str) -> str:
        await asyncio.sleep(0)
        copy = build_byok_settings(
            base, BYOKOverride(provider="gemini", model=model, api_key=key)
        )
        await asyncio.sleep(0)
        # Re-read after another concurrent call may have interleaved -
        # this is what would fail if the shared singleton were mutated.
        return copy.gemini.api_key

    results = await asyncio.gather(
        build("key-A", "gemini-3.1-flash-lite"),
        build("key-B", "gemma-4-26b-a4b-it"),
        build("key-C", "gemini-3.1-flash-lite"),
    )

    assert results == ["key-A", "key-B", "key-C"]
    # The shared singleton was never touched by any of the concurrent calls.
    assert base.gemini.api_key != "key-A"
    assert base.gemini.api_key != "key-B"
    assert base.gemini.api_key != "key-C"
