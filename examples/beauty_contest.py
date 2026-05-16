"""Keynesian Beauty Contest (Guess 2/3 of the average).

N players each pick an integer in [0, MAX]. The winner is the player whose
guess is closest to ``factor * mean(all_guesses)``. Ties split equally.

Game-theoretically the unique Nash is everyone picking 0, but in the wild
players don't iterate infinitely — bots that model bounded rationality win.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import RandomBot, RoundRobin


MAX_GUESS = 100
FACTOR = 2 / 3
N_PLAYERS = 4


@dataclass(frozen=True)
class BeautyState:
    guesses: Tuple[Optional[int], ...] = (None,) * N_PLAYERS
    players: Tuple[int, ...] = tuple(range(N_PLAYERS))

    def active_players(self) -> Sequence[int]:
        return tuple(p for p, g in zip(self.players, self.guesses) if g is None)

    @property
    def current_player(self) -> int:
        for p, g in zip(self.players, self.guesses):
            if g is None:
                return p
        return self.players[0]

    def legal_actions(self, player: int) -> Sequence[int]:
        return tuple(range(MAX_GUESS + 1))

    def apply_joint(self, actions: Mapping[int, int]) -> "BeautyState":
        new = list(self.guesses)
        for i, p in enumerate(self.players):
            if p in actions:
                new[i] = actions[p]
        return replace(self, guesses=tuple(new))

    def apply(self, action: int) -> "BeautyState":
        idx = self.guesses.index(None)
        new = list(self.guesses)
        new[idx] = action
        return replace(self, guesses=tuple(new))

    def is_terminal(self) -> bool:
        return all(g is not None for g in self.guesses)

    def score(self, player: int) -> float:
        if not self.is_terminal():
            return 0.0
        guesses_clean = [g for g in self.guesses if g is not None]
        target = FACTOR * (sum(guesses_clean) / len(guesses_clean))
        diffs = [abs(g - target) for g in guesses_clean]
        best = min(diffs)
        winners = [i for i, d in enumerate(diffs) if d == best]
        idx = self.players.index(player)
        return 1.0 / len(winners) if idx in winners else 0.0


def make_state(seed: int) -> BeautyState:
    return BeautyState()


@dataclass
class LevelKBot:
    """Iterated best-response from a uniform prior, k levels deep."""

    k: int = 2
    name: str = "level-k"

    def choose_action(self, state: BeautyState, player, budget, rng):
        belief = MAX_GUESS / 2.0
        for _ in range(self.k):
            belief = FACTOR * belief
        return max(0, min(MAX_GUESS, round(belief)))


@dataclass
class ConstantBot:
    value: int
    name: str = "constant"

    def choose_action(self, state: BeautyState, player, budget, rng):
        return max(0, min(MAX_GUESS, self.value))


if __name__ == "__main__":
    bots = {
        "l0": ConstantBot(value=50, name="l0"),
        "l1": LevelKBot(k=1, name="l1"),
        "l2": LevelKBot(k=2, name="l2"),
        "l5": LevelKBot(k=5, name="l5"),
        "always_zero": ConstantBot(value=0, name="zero"),
        "random": RandomBot(),
    }
    tournament = RoundRobin(
        players=tuple(range(N_PLAYERS)),
        initial_state_factory=make_state,
        games_per_pair=8,
        time_limit_per_move=0.05,
        simultaneous=True,
        max_turns=2,
    )
    print(tournament.run(bots, seed=3).summary())
