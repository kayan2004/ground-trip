from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict[str, str | bool]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "app_name": settings.app.name,
        "environment": settings.app.env,
        "debug": settings.app.debug,
        "database_configured": bool(settings.database.url),
        "tool_registry_loaded": (
            request.app.state.resources.get("tool_registry") is not None
        ),
        "weather_provider": "open-meteo",
        "discord_webhook_configured": bool(settings.discord.webhook_url),
    }
