"""Coverage for app/core/logging_config.py's RedactSecretsFilter - defense
in depth against a key-shaped string ever reaching a log line, in addition
to (not instead of) the primary safeguard of never logging one in the first
place.
"""

import logging

from app.core.logging_config import RedactSecretsFilter


def _emit(logger: logging.Logger, message: str) -> str:
    records: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _CaptureHandler()
    handler.addFilter(RedactSecretsFilter())
    logger.addHandler(handler)
    try:
        logger.info(message)
    finally:
        logger.removeHandler(handler)
    return records[0]


def test_redaction_filter_scrubs_sk_prefixed_key():
    logger = logging.getLogger("test.redaction.sk")
    logger.setLevel(logging.INFO)
    output = _emit(logger, "request failed with key sk-abcdefghijklmnopqrstuvwxyz")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in output
    assert "[REDACTED]" in output


def test_redaction_filter_scrubs_gemini_shaped_key():
    logger = logging.getLogger("test.redaction.gemini")
    logger.setLevel(logging.INFO)
    output = _emit(logger, "gemini call used AQ.Ab8RN6IU2EeTXob8f6MEYHlYg22h7mGAum5Q")
    assert "AQ.Ab8RN6IU2EeTXob8f6MEYHlYg22h7mGAum5Q" not in output
    assert "[REDACTED]" in output


def test_redaction_filter_scrubs_bearer_header_shape():
    logger = logging.getLogger("test.redaction.bearer")
    logger.setLevel(logging.INFO)
    output = _emit(logger, "outgoing request Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456")
    assert "abcdefghijklmnopqrstuvwxyz123456" not in output
    assert "[REDACTED]" in output


def test_redaction_filter_leaves_normal_messages_untouched():
    logger = logging.getLogger("test.redaction.normal")
    logger.setLevel(logging.INFO)
    output = _emit(logger, "llm_completion provider=gemini model=gemini-3.1-flash-lite tokens=42")
    assert output == "llm_completion provider=gemini model=gemini-3.1-flash-lite tokens=42"
