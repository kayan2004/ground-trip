"""One shared logging setup for both the live app (main.py) and any offline
script that makes LLM calls (currently scripts/cluster_destinations.py's
`name` phase). Without calling this, `logger.info(...)` calls (see
app/services/tool_logs.py, app/services/llm_providers/usage_logging.py) are
silently dropped - Python's root logger has no handler configured by
default, and INFO-level records don't reach the "handler of last resort"
(which only fires at WARNING+).
"""

import logging
import re

# Key-shaped substrings to scrub before any record is emitted. This is
# defense in depth, not the primary safeguard - the primary safeguard is
# that no code path ever passes a BYOK/server API key to a logger call in
# the first place (usage_logging.py only logs provider/model/token counts;
# the agent-runs route never logs the X-LLM-API-Key header or payload).
# This filter exists in case a future change accidentally logs an
# exception message that echoes back an Authorization header or similar.
_KEY_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{10,}"),  # Anthropic/OpenAI-shaped keys
    re.compile(r"AQ\.A[A-Za-z0-9_-]{10,}"),  # this repo's observed Gemini OAuth-shaped key prefix
    re.compile(r"AIza[A-Za-z0-9_-]{10,}"),  # common Gemini API-key prefix
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),  # generic Authorization-header shape
]


class RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = message
        for pattern in _KEY_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Attached to the root logger's *handlers*, not the root logger itself:
    # a Logger-level filter only runs for records that logger's own
    # .handle() processes (i.e. the logger you called .info() on), not for
    # records propagating up from a child logger like "app.llm_usage" -
    # propagation calls each ancestor handler directly, skipping ancestor
    # Logger.filter(). Handler-level filters, by contrast, run for every
    # record that reaches that handler regardless of origin logger, which
    # is what's needed here to catch every logger in the app.
    redaction_filter = RedactSecretsFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redaction_filter)
