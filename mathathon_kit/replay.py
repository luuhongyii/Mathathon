from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Generic, Iterable, List

from .core import Action, Player, StepRecord


class ReplayLog(Generic[Player, Action]):
    """Small JSONL helper for move-by-move debugging."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, records: Iterable[StepRecord[Player, Action]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False, default=str) + "\n")

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def print_summary(self, limit: int = 50) -> str:
        rows = self.read()
        shown = rows[:limit]
        lines = [
            f"turn={row['turn']} player={row['player']} action={row['action']} "
            f"elapsed={row['elapsed']:.4f}s note={row.get('note', '')}"
            for row in shown
        ]
        if len(rows) > limit:
            lines.append(f"... {len(rows) - limit} more turns")
        return "\n".join(lines)
