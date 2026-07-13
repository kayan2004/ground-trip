from dataclasses import dataclass

from app.core.config import Settings
from app.core.llm_allowlist import is_allowlisted


class BYOKValidationError(ValueError):
    """A malformed or non-allowlisted BYOK request - the route layer turns
    this into a 400."""


@dataclass(slots=True, frozen=True)
class BYOKOverride:
    provider: str
    model: str
    api_key: str


def build_byok_settings(base_settings: Settings, override: BYOKOverride) -> Settings:
    """Returns a request-scoped deep copy of `base_settings` with the given
    provider/model/key applied. Never mutates `base_settings` - that's the
    process-global `get_settings()` singleton (see app/core/lifespan.py,
    which assigns it once to `app.state.settings` and shares that same
    reference across every concurrent request via ToolContext.settings). A
    plain in-place mutation here would leak one caller's key into every
    other concurrent request. `model_copy(deep=True)` on a pydantic
    BaseSettings/BaseModel produces entirely new nested objects (a new
    GeminiSettings/AnthropicSettings/OpenAISettings instance, not a shared
    reference to the singleton's), so this is safe to call concurrently
    from multiple requests with different overrides - each call's copy is
    independent.
    """
    if not is_allowlisted(override.provider, override.model):
        raise BYOKValidationError(
            f"Provider/model combination ({override.provider!r}, {override.model!r}) "
            "is not allowlisted for bring-your-own-key requests."
        )

    trip_settings = base_settings.model_copy(deep=True)
    trip_settings.llm_provider = override.provider
    if override.provider == "gemini":
        trip_settings.gemini.api_key = override.api_key
        trip_settings.gemini.model = override.model
    elif override.provider == "anthropic":
        trip_settings.anthropic.api_key = override.api_key
        trip_settings.anthropic.model = override.model
    elif override.provider == "openai":
        trip_settings.openai.api_key = override.api_key
        trip_settings.openai.model = override.model
    return trip_settings
