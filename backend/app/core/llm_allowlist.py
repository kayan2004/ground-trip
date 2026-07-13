from typing import NamedTuple


class AllowlistedModel(NamedTuple):
    provider: str
    model: str


# Single source of truth for which (provider, model) pairs a BYOK request
# may target - anything else is a 400, not a passthrough. Deliberately
# excludes expensive/large tiers (gemini-3.1-pro-preview, full gpt-5.4/5.5,
# Claude Sonnet/Opus) even for BYOK requests: there's no reason to let this
# backend proxy arbitrary expensive models just because the caller supplied
# their own key.
BYOK_ALLOWLIST: tuple[AllowlistedModel, ...] = (
    AllowlistedModel("gemini", "gemini-3.1-flash-lite"),
    AllowlistedModel("gemini", "gemma-4-26b-a4b-it"),
    AllowlistedModel("anthropic", "claude-haiku-4-5"),
    AllowlistedModel("openai", "gpt-5.4-nano"),
    AllowlistedModel("openai", "gpt-5.4-mini"),
)


def is_allowlisted(provider: str, model: str) -> bool:
    return AllowlistedModel(provider, model) in BYOK_ALLOWLIST
