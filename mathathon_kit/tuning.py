from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, Sequence, TypeVar

from .core import Action, Bot, Player
from .tournament import RoundRobin, TournamentReport


BotFactory = Callable[..., Bot[Player, Action]]


@dataclass
class TuningResult(Generic[Player, Action]):
    params: Dict[str, Any]
    bot_name: str
    report: TournamentReport[Player, Action]
    score: float


def parameter_grid(grid: Mapping[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


class GridTuner(Generic[Player, Action]):
    def __init__(
        self,
        tournament: RoundRobin[Player, Action],
        bot_factory: BotFactory[Player, Action],
        base_bots: Mapping[str, Bot[Player, Action]],
        target_name: str = "candidate",
    ) -> None:
        self.tournament = tournament
        self.bot_factory = bot_factory
        self.base_bots = dict(base_bots)
        self.target_name = target_name

    def run(
        self,
        grid: Mapping[str, Sequence[Any]],
        seed: int = 0,
        top_k: int = 5,
    ) -> List[TuningResult[Player, Action]]:
        results: List[TuningResult[Player, Action]] = []

        for idx, params in enumerate(parameter_grid(grid)):
            candidate = self.bot_factory(**params)
            bots = dict(self.base_bots)
            bots[self.target_name] = candidate
            report = self.tournament.run(bots, seed=seed + idx * 10_000)
            stats = report.standings[self.target_name]
            # Prefer robust average performance; Elo is a useful tie-breaker.
            score = stats.win_rate * 10_000 + stats.avg_score + report.elo[self.target_name] / 10_000
            results.append(TuningResult(dict(params), self.target_name, report, score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]
