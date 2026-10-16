"""Symbol table: token -> id, frequency, class.

Compiler concept: the *symbol table* a lexer fills while scanning. The first time a
token string is seen it gets a fresh integer id and remembers its token class; every
later sighting only increments its frequency.
"""
from dataclasses import dataclass


@dataclass
class Entry:
    id: int
    frequency: int
    cls: str


class SymbolTable:
    def __init__(self):
        self._entries: dict[str, Entry] = {}

    def add(self, token: str, cls: str) -> int:
        """Record one occurrence of `token`; return its id."""
        entry = self._entries.get(token)
        if entry is None:
            entry = self._entries[token] = Entry(len(self._entries), 0, cls)
        entry.frequency += 1
        return entry.id

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, token: str) -> bool:
        return token in self._entries

    def __getitem__(self, token: str) -> Entry:
        return self._entries[token]

    def rows(self, top: int | None = None) -> list[dict]:
        """Table rows sorted by frequency (then id); optionally only the top ones."""
        items = sorted(self._entries.items(), key=lambda kv: (-kv[1].frequency, kv[1].id))
        rows = [{"token": t, "id": e.id, "frequency": e.frequency, "class": e.cls}
                for t, e in items]
        return rows[:top] if top else rows
