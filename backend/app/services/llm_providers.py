from typing import Protocol

import httpx

from app.core.config import Settings


class LLMProvider(Protocol):
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class AnthropicProvider:
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not settings.anthropic_api_key:
            raise RuntimeError("Anthropic API key is not configured.")

        response = await http_client.post(
            f"{settings.anthropic_api_base_url}/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": settings.anthropic_api_version,
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=settings.weather_request_timeout_seconds,
        )
        _raise_for_status_with_body(
            response, context=f"Anthropic generation using model '{model}'"
        )
        payload = response.json()
        content_blocks = payload.get("content") or []
        text_parts = [
            block.get("text", "").strip()
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part for part in text_parts if part)


class GeminiProvider:
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured.")

        response = await http_client.post(
            f"{settings.gemini_api_base_url}/{settings.gemini_api_version}"
            f"/models/{model}:generateContent",
            headers={
                "x-goog-api-key": settings.gemini_api_key,
                "content-type": "application/json",
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=settings.weather_request_timeout_seconds,
        )
        _raise_for_status_with_body(
            response, context=f"Gemini generation using model '{model}'"
        )
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_parts = [
            part.get("text", "").strip() for part in parts if isinstance(part, dict)
        ]
        return "\n\n".join(part for part in text_parts if part)


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    raise RuntimeError(
        f"Unknown llm_provider '{settings.llm_provider}' - expected 'anthropic' or 'gemini'."
    )


def _raise_for_status_with_body(response: httpx.Response, *, context: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        raise RuntimeError(
            f"{context} failed with status {response.status_code}. Response body: {body}"
        ) from exc
