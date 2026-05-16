"""Colonel Blotto: simultaneous allocation of N soldiers across K battlefields.

You win a battlefield by sending strictly more soldiers than your opponent.
Total wins decide the round; ties on a battlefield score 0.5 each.
This is a one-shot simultaneous game.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import RandomBot, RoundRobin


N_FIELDS = 4
TOTAL_SOLDIERS = 10


@dataclass(frozen=True)
class BlottoState:
    placed: Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]] = (None, None)
    players: Tuple[int, int] = (0, 1)
    n_fields: int = N_FIELDS
    total: int = TOTAL_SOLDIERS

    def active_players(self) -> Sequence[int]:
        return tuple(p for p, x in zip(self.players, self.placed) if x is None)

    @property
    def current_player(self) -> int:
        for p, x in zip(self.players, self.placed):
            if x is None:
                return p
        return self.players[0]

    def legal_actions(self, player: int) -> Sequence[Tuple[int, ...]]:
        return _enumerate_allocations(self.total, self.n_fields)

    def apply_joint(
        self, actions: Mapping[int, Tuple[int, ...]]
    ) -> "BlottoState":
        a0 = actions.get(self.players[0]) or self.placed[0]
        a1 = actions.get(self.players[1]) or self.placed[1]
        return replace(self, placed=(a0, a1))

    def apply(self, action: Tuple[int, ...]) -> "BlottoState":
        # Sequential fallback so plain bots still work.
        idx = self.placed.index(None)
        new_placed = list(self.placed)
        new_placed[idx] = action
        return replace(self, placed=tuple(new_placed))

    def is_terminal(self) -> bool:
        return all(x is not None for x in self.placed)

    def score(self, player: int) -> float:
        if not self.is_terminal():
            return 0.0
        a0, a1 = self.placed
        wins0 = wins1 = 0.0
        assert a0 is not None and a1 is not None
        for x, y in zip(a0, a1):
            if x > y:
                wins0 += 1
            elif y > x:
                wins1 += 1
            else:
                wins0 += 0.5
                wins1 += 0.5
        idx = self.players.index(player)
        return wins0 if idx == 0 else wins1


_alloc_cache: Dict[Tuple[int, int], Tuple[Tuple[int, ...], ...]] = {}


def _enumerate_allocations(total: int, n: int) -> Tuple[Tuple[int, ...], ...]:
    cached = _alloc_cache.get((total, n))
    if cached is not None:
        return cached

    out: List[Tuple[int, ...]] = []

    def rec(remaining: int, slots: int, prefix: List[int]) -> None:
        if slots == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for x in range(remaining + 1):
            rec(remaining - x, slots - 1, prefix + [x])

    rec(total, n, [])
    out_t = tuple(out)
    _alloc_cache[(total, n)] = out_t
    return out_t


def random_allocation(total: int, n: int, rng: random.Random) -> Tuple[int, ...]:
    cuts = sorted(rng.randint(0, total) for _ in range(n - 1))
    cuts = [0] + cuts + [total]
    return tuple(cuts[i + 1] - cuts[i] for i in range(n))


def make_state(seed: int) -> BlottoState:
    return BlottoState()


@dataclass
class UniformBlottoBot:
    name: str = "uniform"

    def choose_action(self, state: BlottoState, player, budget, rng):
        base = state.total // state.n_fields
        rem = state.total - base * state.n_fields
        alloc = [base] * state.n_fields
        for i in range(rem):
            alloc[i] += 1
        rng.shuffle(alloc)
        return tuple(alloc)


@dataclass
class SkewedBlottoBot:
    """Concentrate forces on a random subset of fields."""

    keep: int = 3
    name: str = "skewed"

    def choose_action(self, state: BlottoState, player, budget, rng):
        idx = list(range(state.n_fields))
        rng.shuffle(idx)
        chosen = sorted(idx[: self.keep])
        share = state.total // self.keep
        rem = state.total - share * self.keep
        alloc = [0] * state.n_fields
        for k, i in enumerate(chosen):
            alloc[i] = share + (1 if k < rem else 0)
        return tuple(alloc)


@dataclass
class BestSampleBot:
    """Sample many random allocations; pick the one beating the most random opponents."""

    samples: int = 200
    opponents: int = 30
    name: str = "best-sample"

    def choose_action(self, state: BlottoState, player, budget, rng):
        opps = [random_allocation(state.total, state.n_fields, rng) for _ in range(self.opponents)]
        best = None
        best_score = -1.0
        for _ in range(self.samples):
            cand = random_allocation(state.total, state.n_fields, rng)
            score = sum(_match(cand, o) for o in opps)
            if score > best_score:
                best_score = score
                best = cand
        return best or random_allocation(state.total, state.n_fields, rng)


def _match(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    s = 0.0
    for x, y in zip(a, b):
        if x > y:
            s += 1
        elif y > x:
            s -= 1
    return s


if __name__ == "__main__":
    bots = {
        "uniform": UniformBlottoBot(),
        "skewed3": SkewedBlottoBot(keep=3),
        "skewed2": SkewedBlottoBot(keep=2),
        "random": RandomBot(),
        "best_sample": BestSampleBot(),
    }
    tournament = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=40,
        time_limit_per_move=0.1,
        simultaneous=True,
        max_turns=2,
    )
    print(tournament.run(bots, seed=7).summary())
