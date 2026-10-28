"""
guardrails/citation_check.py  – lightweight citation checker for generated answers.

Checks that every URL cited in a markdown-style ``[text](url)`` link inside the
answer actually originated from the retrieved code chunks.  Any URL that does
not start with an allowed host prefix (or that was not seen in the retrieved
context) is flagged.

Usage
-----
    from guardrails.citation_check import check_citations

    result = check_citations(answer_text, retrieved_chunks=["… code …"])
    if result.has_bad_citations:
        ...
"""
import re
from dataclasses import dataclass, field

# Markdown-link citations  [label](url)
_CITATION_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")

# Code-block reference anchors emitted by build_context_window, e.g.
# "Source: retrieval/retrieve.py"
_SOURCE_LINE_RE = re.compile(r"(?:Source|File):\s*(.+)")


@dataclass
class CitationResult:
    total: int = 0
    bad_urls: list[str] = field(default_factory=list)

    @property
    def has_bad_citations(self) -> bool:
        return bool(self.bad_urls)


def check_citations(
    answer: str,
    retrieved_chunks: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
) -> CitationResult:
    """
    Scan *answer* for markdown citation URLs.

    Parameters
    ----------
    answer:
        The generated answer text to scan.
    retrieved_chunks:
        List of code-chunk strings from build_context_window.  Any URL whose
        host appears verbatim in a chunk is considered approved.
    allowed_hosts:
        Additional URL prefixes (e.g. ``["https://github.com/"]``) always
        treated as safe.
    """
    allowed_hosts = list(allowed_hosts or [])
    chunk_text = "\n".join(retrieved_chunks or [])

    urls = _CITATION_RE.findall(answer)  # list of (label, url) tuples
    result = CitationResult(total=len(urls))

    for _label, url in urls:
        # 1. Check explicitly allowed hosts
        if any(url.startswith(h) for h in allowed_hosts):
            continue
        # 2. Check if the URL (or its domain) appears in the retrieved chunks
        if url in chunk_text:
            continue
        # No match → potentially fabricated citation
        result.bad_urls.append(url)

    return result
