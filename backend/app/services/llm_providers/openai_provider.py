import time

import httpx

from app.core.config import Settings
from app.services.llm_providers.errors import raise_for_status_with_body
from app.services.llm_providers.protocol import Message
from app.services.llm_providers.usage_logging import log_completion_usage


class OpenAIProvider:
    """Pure REST over the shared httpx.AsyncClient, no OpenAI SDK dependency
    - same pattern as AnthropicProvider. OpenAI's chat completions API keeps
    system and user turns in one flat messages array (unlike Anthropic's
    separate top-level `system` field), so messages are passed through as-is
    rather than using protocol.split_system_and_user.
    """

    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._http_client = http_client

    async def complete(
        self,
        messages: list[Message],
        **opts: object,
    ) -> str:
        settings = self._settings
        if not settings.openai.api_key:
            raise RuntimeError("OpenAI API key is not configured.")

        model = settings.openai.model
        max_tokens = opts.get("max_tokens", settings.openai.max_tokens)
        temperature = opts.get("temperature", settings.openai.temperature)

        started_at = time.monotonic()
        response = await self._http_client.post(
            f"{settings.openai.api_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                # The gpt-5.x chat-completions surface uses
                # max_completion_tokens, not the older max_tokens field -
                # NOT live-verified (no OPENAI_API_KEY in this repo). If a
                # real OpenAI call ever 400s complaining about an unknown
                # field, this is the first thing to check.
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": message["role"], "content": message["content"]}
                    for message in messages
                ],
            },
            timeout=settings.weather.request_timeout_seconds,
        )
        elapsed_seconds = time.monotonic() - started_at
        raise_for_status_with_body(
            response, context=f"OpenAI generation using model '{model}'"
        )
        payload = response.json()

        usage = payload.get("usage") or {}
        log_completion_usage(
            provider="openai",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_seconds=elapsed_seconds,
        )

        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return (message.get("content") or "").strip()
