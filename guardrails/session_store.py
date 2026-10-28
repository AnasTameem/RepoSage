"""
guardrails/session_store.py – LRU-bounded session store with history trimming.

Maintains an in-memory OrderedDict of chat/query sessions, evicting the oldest
session when MAX_SESSIONS is reached and capping conversation turns per session.
"""
from collections import OrderedDict
from typing import Any

from metrics.request_metrics import FALLBACK_COUNTER


class SessionStore:
    """LRU-capped session store that evicts the oldest session when max_sessions is hit."""

    def __init__(self, max_sessions: int = 500, max_history_turns: int = 5) -> None:
        self._store: OrderedDict[str, dict[str, list[Any]]] = OrderedDict()
        self.max_sessions = max_sessions
        self.max_turns = max_history_turns

    def get(self, session_id: str) -> dict[str, list[Any]]:
        if session_id not in self._store:
            if len(self._store) >= self.max_sessions:
                # Evict oldest session
                self._store.popitem(last=False)
                FALLBACK_COUNTER.labels(reason="session_evicted").inc()
            self._store[session_id] = {"query_history": [], "chat_history": []}
        else:
            # Move to end (most recently used)
            self._store.move_to_end(session_id)
        return self._store[session_id]

    def reset(self, session_id: str) -> None:
        self._store[session_id] = {"query_history": [], "chat_history": []}

    def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._store)

    def trim_history(self, session: dict[str, list[Any]]) -> None:
        """Keep only the last max_turns of each history list."""
        if len(session["query_history"]) > self.max_turns:
            session["query_history"] = session["query_history"][-self.max_turns :]
        if len(session["chat_history"]) > self.max_turns * 2:
            session["chat_history"] = session["chat_history"][-(self.max_turns * 2) :]
