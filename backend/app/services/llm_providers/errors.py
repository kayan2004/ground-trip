import httpx


class LLMAuthenticationError(RuntimeError):
    """The configured LLM API key was rejected by the provider (401/403).

    Distinguished from a plain RuntimeError so callers can choose to
    propagate this as a client-facing 4xx (a BYOK request) instead of
    letting it degrade gracefully into a "partial" agent-run status (the
    server's own default key) - see app/agent/graph.py's node-level
    try/except blocks and app/api/routes/agent_runs.py.
    """

    def __init__(self, *, status_code: int, context: str, body: str) -> None:
        self.status_code = status_code
        super().__init__(
            f"{context} failed with status {status_code} (authentication rejected). "
            f"Response body: {body}"
        )


def raise_for_status_with_body(response: httpx.Response, *, context: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        if response.status_code in (401, 403):
            raise LLMAuthenticationError(
                status_code=response.status_code, context=context, body=body
            ) from exc
        raise RuntimeError(
            f"{context} failed with status {response.status_code}. Response body: {body}"
        ) from exc
