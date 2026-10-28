"""
guardrails/redactor.py  – regex-based secret redactor for code chunks.

Strips common credential patterns (API keys, tokens, passwords, AWS IDs)
before code is embedded in the prompt or echoed back in the SSE response.
The redaction is intentionally conservative: it only rewrites lines that
look like assignment statements or export commands that contain a secret.
"""
import re
from re import Match

# ---------------------------------------------------------------------------
# Patterns that look like secret values
# ---------------------------------------------------------------------------
_SECRET_VALUE_RE = re.compile(
    r"""
    (?:
        # Long random-looking base64 / hex / dotted strings (≥16 chars)
        [A-Za-z0-9+/=_\-.]{16,}
        |
        # Dotted prefix tokens  e.g. sk-proj-xxx, AQ.xxx, xoxb-xxx
        (?:[A-Za-z0-9]{1,15}[.\-][A-Za-z0-9\-_]{6,})
        |
        # JWT (three base64url parts separated by dots)
        [A-Za-z0-9\-_]{3,}\.[A-Za-z0-9\-_]{3,}\.[A-Za-z0-9\-_]+
        |
        # UUID-style  xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}
    )
    """,
    re.VERBOSE,
)

# Variable names / key-patterns that suggest the value is a secret
_SECRET_KEY_RE = re.compile(
    r"""(?xi)
    (?:
        api[_\-]?key | apikey | secret[_\-]?key | secret |
        password | passwd | pwd |
        token | access[_\-]?token | refresh[_\-]?token |
        auth[_\-]?token | auth | bearer |
        private[_\-]?key | signing[_\-]?key |
        credential |
        aws[_\-]?secret | aws[_\-]?access[_\-]?key[_\-]?id |
        client[_\-]?secret | oauth[_\-]?secret |
        db[_\-]?password | database[_\-]?password |
        smtp[_\-]?password | ftp[_\-]?password
    )
    """,
)

# Full assignment line: KEY=value  /  key: "value"  / key = "value"
_ASSIGNMENT_RE = re.compile(
    r"""(?x)
    ^
    (?P<indent>[ \t]*)                  # leading whitespace
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)    # identifier
    (?P<sep>\s*[:=]\s*)                 # separator
    (?P<quote>["\']?)                   # optional opening quote
    (?P<value>.+?)                      # value (lazy)
    (?P=quote)                          # matching closing quote (may be empty)
    (?P<tail>[,;]?\s*)                  # trailing comma/semicolon
    $
    """,
    re.MULTILINE,
)

_REDACTED = "[REDACTED]"


def _maybe_redact_line(match: Match[str]) -> str:
    """If the key looks secret-like AND the value looks like a secret, redact."""
    key = match.group("key")
    value = match.group("value")
    if _SECRET_KEY_RE.search(key) and _SECRET_VALUE_RE.search(value):
        return (
            match.group("indent")
            + key
            + match.group("sep")
            + match.group("quote")
            + _REDACTED
            + match.group("quote")
            + match.group("tail")
        )
    return match.group(0)


def redact_secrets(text: str) -> str:
    """
    Return *text* with likely secret assignments replaced by ``[REDACTED]``.

    Operates line-by-line on assignment patterns; non-assignment lines are
    returned unchanged so code logic/comments are preserved.
    """
    return _ASSIGNMENT_RE.sub(_maybe_redact_line, text)


def redact_error(exc: Exception) -> str:
    """
    Return a sanitised error message safe to return to the client.

    Strips raw exception details (which may leak stack frames, file paths
    or credentials) and replaces them with a generic message plus a short
    category tag.
    """
    cls = type(exc).__name__
    # Map common exception types to helpful-but-safe messages
    _SAFE_MAP = {
        "TimeoutError": "The request timed out. Please try again.",
        "asyncio.TimeoutError": "The request timed out. Please try again.",
        "ConnectionRefusedError": "A backend service is temporarily unavailable.",
        "ConnectionError": "A backend service is temporarily unavailable.",
        "PermissionError": "Permission denied by a backend service.",
        "ValueError": "Invalid input or configuration error.",
        "KeyError": "A required field is missing.",
    }
    for key, msg in _SAFE_MAP.items():
        if key in cls:
            return msg
    return f"An internal error occurred ({cls}). Please try again or contact support."
