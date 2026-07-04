# Provider-Agnostic LLM Layer + Gemini/Gemma 4 Provider

**Goal:** Decouple the three LLM call sites (field extraction, trip synthesis, cluster naming)
from Anthropic's specific REST shape, and add Gemini/Gemma 4 as a second, config-selectable
provider.

## Motivation

Cost/flexibility: the ability to switch which LLM provider backs the app via configuration,
without code changes. Not a fallback/failover mechanism — whichever provider is configured is
used, with no automatic switching on error. (Context: this need surfaced while the Anthropic
account was blocked on a billing issue, but the design is a general capability, not a
workaround for that specific incident.)

## Scope

All three LLM call sites become provider-agnostic:

- `extract_request_fields` (Claude field extraction from a user's trip prompt)
- `synthesize_trip_response` (final trip-planning answer synthesis)
- `propose_cluster_tag` (offline HDBSCAN cluster naming, `scripts/cluster_destinations.py`)

**Out of scope:**
- `list_anthropic_models` / `GET /tools/anthropic-models` — a debug/dev endpoint that lists
  Anthropic's own model catalog. Gemini's model-listing API has an entirely different shape;
  generalizing this debug endpoint has no bearing on the app's actual behavior and isn't worth
  the churn.
- Automatic fallback/retry between providers.
- Per-call-site provider override (extraction on one provider, synthesis on another). One global
  switch controls all three call sites.
- Any new third-party LLM SDK (e.g. LiteLLM). This project deliberately uses raw `httpx` calls
  with no provider SDKs (see `CLAUDE.md` conventions); the adapter layer below is small enough
  that a dependency isn't justified.

## Provider: Gemini API serving Gemma 4

Gemma 4 (Google's open-weight model family, Apache 2.0, released April 2026, built from the same
research as Gemini 3) is served through the same Gemini Developer API used for proprietary Gemini
models — REST endpoint, API key, `generateContent`. This fits the existing codebase pattern of
direct `httpx` REST calls (no SDK) exactly.

- Fast tier: `gemma-4-26b-a4b-it` (26B params, Mixture-of-Experts, 4B active — cheap/fast)
- Strong tier: `gemma-4-31b-it` (31B dense — highest-quality open Gemma 4 variant)

This mirrors the existing Anthropic fast/strong split (Haiku / Sonnet).

**Verified Gemini REST shape** (not assumed from training data — checked against current Google
AI for Developers documentation, July 2026):

Request:
```json
{
  "contents": [{"role": "user", "parts": [{"text": "..."}]}],
  "systemInstruction": {"parts": [{"text": "..."}]},
  "generationConfig": {"maxOutputTokens": 700, "temperature": 0.2}
}
```
`POST {gemini_api_base_url}/{gemini_api_version}/models/{model}:generateContent`
Auth header: `x-goog-api-key: <GEMINI_API_KEY>` (current Google-recommended method, mirrors
Anthropic's `x-api-key` header nicely).

Response: `candidates[0].content.parts[0].text`.

**Operational note (not a code concern, but worth knowing when generating the key):** Google is
phasing out unrestricted Gemini API keys — restricted standard keys work until September 2026,
after which only service-account-bound auth keys are accepted. Generate a *restricted* API key
(scoped to the Generative Language API) in Google AI Studio / Google Cloud Console.

## Architecture

```
app/services/
├── llm.py              (renamed from claude.py) — orchestration: extract_request_fields,
│                         synthesize_trip_response, propose_cluster_tag, choose_model,
│                         plus all the existing prompt-building/post-processing helpers
│                         (unchanged, still provider-agnostic since they only touch text/JSON)
└── llm_providers.py     (new) — LLMProvider protocol, AnthropicProvider, GeminiProvider,
                          get_llm_provider(settings) factory
```

This mirrors the `BaseTool`/`ToolRegistry` pattern already used in `app/agent/tools/` — a shared
interface with a small registry/factory picking the concrete implementation, rather than inline
branching per call site.

### `llm_providers.py`

```python
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
```

- `AnthropicProvider.generate()` — the existing `/v1/messages` request-building and
  `content[].text` response-parsing logic, moved here unchanged from `claude.py`.
- `GeminiProvider.generate()` — builds the request body shown above, POSTs to
  `{gemini_api_base_url}/{gemini_api_version}/models/{model}:generateContent`, parses
  `candidates[0].content.parts[0].text`. Raises the same `RuntimeError`-with-response-body
  pattern as `_raise_for_status_with_body` (reused, not duplicated).
- `get_llm_provider(settings) -> LLMProvider` — returns `AnthropicProvider()` or
  `GeminiProvider()` based on `settings.llm_provider`. Raises a clear `RuntimeError` for an
  unrecognized value rather than silently defaulting.

### `llm.py` changes

`extract_request_fields`, `synthesize_trip_response`, and `propose_cluster_tag` stop building
Anthropic-specific request bodies. Each now:
1. Resolves the provider via `get_llm_provider(settings)`.
2. Resolves which model to use (fast/strong per `choose_model()`, or a fixed model for
   extraction/naming as today).
3. Calls `provider.generate(http_client, settings, system=..., user_content=..., model=...,
   max_tokens=..., temperature=...)`.
4. Runs the *existing, unchanged* text-post-processing: `_extract_json_payload`, Pydantic
   validation, `_normalize_extracted_payload`, etc. None of this changes — it already operates on
   plain text and doesn't know or care which provider produced it.

`choose_anthropic_model()` is renamed `choose_model()`. Same length/failed-tools/richness
heuristic; returns `settings.gemini_fast_model`/`settings.gemini_strong_model` or
`settings.anthropic_fast_model`/`settings.anthropic_strong_model` depending on
`settings.llm_provider`.

## Config additions (`app/core/config.py`, `.env.example`)

Nothing existing is renamed. New settings only:

```python
llm_provider: str = "anthropic"          # "anthropic" | "gemini"
gemini_api_key: str = ""
gemini_api_base_url: str = "https://generativelanguage.googleapis.com"
gemini_api_version: str = "v1beta"
gemini_fast_model: str = "gemma-4-26b-a4b-it"
gemini_strong_model: str = "gemma-4-31b-it"
gemini_max_tokens: int = 700             # mirrors anthropic_max_tokens
gemini_temperature: float = 0.2          # mirrors anthropic_temperature
```

## Call-site import updates

- `app/agent/graph.py` — `from app.services.claude import ...` -> `from app.services.llm import ...`
- `app/services/clustering.py` — same import update for `propose_cluster_tag`
- `app/api/routes/claude.py` — same import update; also switch `choose_anthropic_model` ->
  `choose_model`, and `ExtractionTestResponse.selected_model` (currently hardcodes
  `settings.anthropic_fast_model` directly) needs to report whichever provider's fast model was
  actually used
- `app/api/routes/anthropic.py` — untouched (out of scope, see above)
- `app/schemas/claude.py` — untouched (schema names aren't provider-specific in a way that
  matters; renaming adds churn with no functional benefit)

## Error handling

Unchanged pattern, generalized: a provider HTTP error becomes a `RuntimeError` with the response
body attached (both adapters use the same helper). No fallback between providers. Missing API key
for the *configured* provider fails loudly and immediately — generalizes today's
`"Anthropic API key is not configured"` check to whichever provider is active.

## Testing / verification

No automated test suite exists in this project (documented known gap) — this doesn't introduce
one. Verification plan:
1. Each adapter's request-building and response-parsing checked against a mocked/synthetic
   response (sanity-check the JSON shape handling without a live call).
2. A real end-to-end call against the live Gemini API for at least one call site, once a
   `GEMINI_API_KEY` is available.
3. Confirm the existing Anthropic path is unaffected (`llm_provider=anthropic`, the default) —
   re-run an existing working flow (e.g. `POST /tools/test-extraction`) unchanged.

## Documentation updates

- `backend/README.md` — new section documenting the provider abstraction, how to switch
  providers, and the Gemini API key setup note (restricted key requirement).
- `CLAUDE.md` — update the `services/claude.py` reference to `services/llm.py` +
  `llm_providers.py` in the architecture tree.
