"""Coverage for app/core/config.py's AuthSettings.jwt_secret_key validator -
the fix for a real finding from a pre-deployment audit: the field's
dev-convenience default is also the exact placeholder text committed in
.env.example, so anyone deploying without setting JWT_SECRET_KEY would
otherwise boot with a publicly-known, forgeable JWT signing secret.
"""

import pytest
from pydantic import ValidationError

from app.core.config import AuthSettings


def test_rejects_the_exact_env_example_placeholder():
    with pytest.raises(ValidationError, match="placeholder value"):
        AuthSettings(_env_file=None)


def test_rejects_a_short_secret():
    with pytest.raises(ValidationError, match="at least 32 characters"):
        AuthSettings(_env_file=None, jwt_secret_key="too-short")


def test_accepts_a_real_secret():
    real_secret = "a" * 48
    settings = AuthSettings(_env_file=None, jwt_secret_key=real_secret)
    assert settings.jwt_secret_key == real_secret
