"""Iterated Prisoner's Dilemma (simultaneous-move).

Round-robin between TitForTat, Grim, Pavlov, Random, AlwaysCoop, AlwaysDef.
This shows how to implement ``SimultaneousState`` for a repeated matrix game.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import (
    GrimTriggerBot,
    PavlovBot,
    RandomBot,
    RoundRobin,
    TitForTatBot,
)


COOP, DEFECT = 0, 1
PAYOFF: Dict[Tuple[int, int], Tuple[float, float]] = {
    (COOP, COOP): (3, 3),
    (COOP, DEFECT): (0, 5),
    (DEFECT, COOP): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}


@dataclass(frozen=True)
class IPDState:
    rounds_left: int
    history: Tuple[Tuple[int, int], ...] = ()
    cumulative: Tuple[float, float] = (0.0, 0.0)
    players: Tuple[int, int] = (0, 1)

    def active_players(self) -> Sequence[int]:
        return self.players if self.rounds_left > 0 else ()

    def legal_actions(self, player: int) -> Sequence[int]:
        return (COOP, DEFECT) if self.rounds_left > 0 else ()

    @property
    def current_player(self) -> int:
        # SimultaneousSimulator queries this only when bots assume sequential.
        return self.players[0]

    def apply_joint(self, actions: Mapping[int, int]) -> "IPDState":
        a0, a1 = actions[self.players[0]], actions[self.players[1]]
        p0, p1 = PAYOFF[(a0, a1)]
        return replace(
            self,
            rounds_left=self.rounds_left - 1,
            history=self.history + ((a0, a1),),
            cumulative=(self.cumulative[0] + p0, self.cumulative[1] + p1),
        )

    def is_terminal(self) -> bool:
        return self.rounds_left <= 0

    def score(self, player: int) -> float:
        return self.cumulative[self.players.index(player)]


def make_state(seed: int) -> IPDState:
    return IPDState(rounds_left=50)


class _AlwaysBot:
    def __init__(self, action: int, name: str) -> None:
        self.action = action
        self.name = name

    def choose_action(self, state, player, budget, rng):
        return self.action


class _ObservingWrapper:
    """Feed previous opponent moves into bots that implement ``observe``."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "wrapped")
        self._last_history_len = 0

    def choose_action(self, state: IPDState, player, budget, rng):
        # Replay opponent moves we haven't told the bot about yet.
        if hasattr(self.inner, "observe"):
            opp_idx = 1 - state.players.index(player)
            for h in state.history[self._last_history_len:]:
                self.inner.observe(h[opp_idx])
            self._last_history_len = len(state.history)
        return self.inner.choose_action(state, player, budget, rng)


if __name__ == "__main__":
    bots = {
        "tit_for_tat": _ObservingWrapper(TitForTatBot(cooperate=COOP, defect=DEFECT)),
        "grim": _ObservingWrapper(GrimTriggerBot(cooperate=COOP, defect=DEFECT)),
        "pavlov": _ObservingWrapper(PavlovBot(cooperate=COOP, defect=DEFECT)),
        "random": RandomBot(),
        "always_coop": _AlwaysBot(COOP, "always_coop"),
        "always_def": _AlwaysBot(DEFECT, "always_def"),
    }
    tournament = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=10,
        time_limit_per_move=0.05,
        simultaneous=True,
        max_turns=100,
    )
    print(tournament.run(bots, seed=1).summary())
