"""Parallel round-robin via ``ProcessPoolExecutor``.

Bots and the state factory must be picklable. For most adapters built on
top of ``@dataclass(frozen=True)`` plus module-level bot classes this
works out of the box.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Generic, List, Mapping, Optional, Sequence, Tuple

from .core import (
    Action,
    Bot,
    GameResult,
    GameState,
    Player,
    SimultaneousSimulator,
    SimultaneousState,
    Simulator,
)
from .tournament import (
    InitialStateFactory,
    MatchStats,
    RoundRobin,
    TournamentReport,
    _collect_errors,
)


def _play_one(args):
    (
        seed,
        order,
        bots_pickled,
        players,
        factory,
        max_turns,
        time_limit,
        simultaneous,
    ) = args
    bots = bots_pickled
    state = factory(seed)
    if simultaneous:
        sim = SimultaneousSimulator(
            players=players,
            max_rounds=max_turns,
            time_limit_per_move=time_limit,
        )
    else:
        sim = Simulator(
            players=players,
            max_turns=max_turns,
            time_limit_per_move=time_limit,
        )
    assigned = {players[i]: bots[order[i]] for i in range(len(players))}
    result = sim.play(state, assigned, seed=seed, record=False)
    return order, result


class ParallelRoundRobin(Generic[Player, Action]):
    """Drop-in replacement for ``RoundRobin`` that fans games across processes."""

    def __init__(
        self,
        players: Sequence[Player],
        initial_state_factory: InitialStateFactory[Player, Action],
        games_per_pair: int = 20,
        max_turns: int = 1_000,
        time_limit_per_move: float = 0.2,
        simultaneous: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        self.players = list(players)
        self.initial_state_factory = initial_state_factory
        self.games_per_pair = games_per_pair
        self.max_turns = max_turns
        self.time_limit_per_move = time_limit_per_move
        self.simultaneous = simultaneous
        self.max_workers = max_workers

    def run(
        self,
        bots: Mapping[str, Bot[Player, Action]],
        seed: int = 0,
    ) -> TournamentReport[Player, Action]:
        import itertools

        names = list(bots.keys())
        n_seats = len(self.players)
        if len(names) < n_seats:
            raise ValueError(
                f"need at least {n_seats} bots for a {n_seats}-seat game"
            )

        jobs = []
        game_id = 0
        for combo in itertools.combinations(names, n_seats):
            base = list(combo)
            for repeat in range(self.games_per_pair):
                shift = repeat % n_seats
                order = base[shift:] + base[:shift]
                jobs.append((
                    seed + game_id,
                    tuple(order),
                    dict(bots),
                    tuple(self.players),
                    self.initial_state_factory,
                    self.max_turns,
                    self.time_limit_per_move,
                    self.simultaneous,
                ))
                game_id += 1

        standings: Dict[str, MatchStats] = defaultdict(MatchStats)
        matrix: Dict[Tuple[str, ...], MatchStats] = defaultdict(MatchStats)
        elo = {name: 1500.0 for name in names}
        rr = RoundRobin(
            players=self.players,
            initial_state_factory=self.initial_state_factory,
            games_per_pair=self.games_per_pair,
            max_turns=self.max_turns,
            time_limit_per_move=self.time_limit_per_move,
            simultaneous=self.simultaneous,
        )

        error_samples: List[str] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            for order, result in pool.map(_play_one, jobs, chunksize=1):
                rr._record_game(list(order), result, standings, matrix)
                rr._update_elo(list(order), result, elo)
                _collect_errors(result, error_samples)

        return TournamentReport(
            dict(standings), dict(matrix), elo, [],
            seats=n_seats, error_samples=error_samples,
        )
