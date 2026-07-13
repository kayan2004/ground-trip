from fastapi import APIRouter

from app.core.llm_allowlist import BYOK_ALLOWLIST

router = APIRouter(tags=["llm-options"])


@router.get("/llm-options")
async def list_llm_options() -> list[dict[str, str]]:
    """The BYOK (provider, model) allowlist - unauthenticated, no secrets,
    just static config the frontend needs to populate its provider/model
    select without duplicating the list.
    """
    return [{"provider": entry.provider, "model": entry.model} for entry in BYOK_ALLOWLIST]
