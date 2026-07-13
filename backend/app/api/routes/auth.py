from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import auth_ip_rate_limiter
from app.core.security import ACCESS_TOKEN_COOKIE_NAME, create_access_token
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.auth import Token, UserCreate, UserLogin, UserRead
from app.services.auth import authenticate_user, create_user

router = APIRouter(prefix="/auth", tags=["auth"])


async def _enforce_auth_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not await auth_ip_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests - please slow down.")


def _set_access_token_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    # secure follows the same debug flag that already gates "is this a real
    # deployment" elsewhere in this session's fixes - a Secure cookie is
    # refused by browsers over plain HTTP, so this must be False for local
    # dev (APP_DEBUG=true) and True everywhere else. A production deploy
    # without HTTPS would silently fail to log in rather than send the
    # cookie insecurely - fail closed, not open.
    settings_debug = settings.app.debug
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=not settings_debug,
        samesite="lax",
        max_age=settings.auth.access_token_expire_minutes * 60,
        path="/",
    )


@router.post("/signup", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: UserCreate,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> UserRead:
    await _enforce_auth_rate_limit(request)
    user = await create_user(session, payload)
    return UserRead.model_validate(user)


@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(
    payload: UserLogin,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> Token:
    await _enforce_auth_rate_limit(request)
    user = await authenticate_user(session, payload)
    token = create_access_token(user.email)
    _set_access_token_cookie(response, token)
    return Token(access_token=token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    # Idempotent, no auth required - clearing an already-absent/expired
    # cookie is harmless, and requiring a valid session just to log out
    # would be a pointless UX trap if the cookie already expired.
    response.delete_cookie(key=ACCESS_TOKEN_COOKIE_NAME, path="/")


@router.get("/me", response_model=UserRead, status_code=status.HTTP_200_OK)
async def read_current_user(current_user: User = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current_user)
