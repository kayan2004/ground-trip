from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes.agent_runs import router as agent_runs_router
from app.api.routes.auth import router as auth_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    application = FastAPI(
        title=settings.app.name,
        debug=settings.app.debug,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.app.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(agent_runs_router)
    application.include_router(auth_router)
    application.include_router(feedback_router)
    application.include_router(health_router)
    return application

app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
