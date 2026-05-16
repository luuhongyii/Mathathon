"""First-price sealed-bid auction with private values.

Each round: every player draws a private value v ~ Uniform(0, 100), submits
a bid b in [0, 100], the highest bidder pays their bid and gets the item
(profit v - b). Repeated for ROUNDS rounds. Winner = highest cumulative profit.

This is a classic mechanism-design test: optimal first-price strategy in
2-player uniform values is to bid v/2; with N bidders, bid v*(N-1)/N.
Bots that infer this scale beat naive bots.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import RandomBot, RoundRobin


N_BIDDERS = 3
ROUNDS = 20
VALUE_MAX = 100


@dataclass(frozen=True)
class AuctionState:
    rng_seed: int
    round_idx: int = 0
    private_values: Tuple[int, ...] = ()
    bids: Tuple[Optional[int], ...] = ()
    cumulative_profit: Tuple[float, ...] = (0.0,) * N_BIDDERS
    players: Tuple[int, ...] = tuple(range(N_BIDDERS))

    def __post_init__(self) -> None:
        if not self.private_values:
            import random as _r
            r = _r.Random(self.rng_seed * 1000 + self.round_idx)
            object.__setattr__(self, "private_values", tuple(r.randint(0, VALUE_MAX) for _ in self.players))
            object.__setattr__(self, "bids", (None,) * len(self.players))

    def active_players(self) -> Sequence[int]:
        return tuple(p for p, b in zip(self.players, self.bids) if b is None)

    @property
    def current_player(self) -> int:
        for p, b in zip(self.players, self.bids):
            if b is None:
                return p
        return self.players[0]

    def legal_actions(self, player: int) -> Sequence[int]:
        # Bid must be 0..value (no overbidding above own value).
        v = self.private_values[self.players.index(player)]
        return tuple(range(0, v + 1))

    def apply_joint(self, actions: Mapping[int, int]) -> "AuctionState":
        new_bids = list(self.bids)
        for i, p in enumerate(self.players):
            if p in actions:
                new_bids[i] = actions[p]
        if any(b is None for b in new_bids):
            return replace(self, bids=tuple(new_bids))
        return self._resolve(tuple(new_bids))

    def apply(self, action: int) -> "AuctionState":
        idx = self.bids.index(None)
        new_bids = list(self.bids)
        new_bids[idx] = action
        if any(b is None for b in new_bids):
            return replace(self, bids=tuple(new_bids))
        return self._resolve(tuple(new_bids))

    def _resolve(self, bids: Tuple[int, ...]) -> "AuctionState":
        max_bid = max(bids)
        winners = [i for i, b in enumerate(bids) if b == max_bid]
        share = 1.0 / len(winners)
        new_profit = list(self.cumulative_profit)
        for i in winners:
            new_profit[i] += share * (self.private_values[i] - max_bid)

        next_round = self.round_idx + 1
        if next_round >= ROUNDS:
            return replace(
                self,
                round_idx=next_round,
                bids=bids,
                cumulative_profit=tuple(new_profit),
            )
        # Roll into next round: regenerate private values via __post_init__.
        return AuctionState(
            rng_seed=self.rng_seed,
            round_idx=next_round,
            private_values=(),  # triggers regen
            cumulative_profit=tuple(new_profit),
            players=self.players,
        )

    def is_terminal(self) -> bool:
        return self.round_idx >= ROUNDS

    def score(self, player: int) -> float:
        return self.cumulative_profit[self.players.index(player)]


def make_state(seed: int) -> AuctionState:
    return AuctionState(rng_seed=seed)


@dataclass
class ShadeBot:
    """Bid value * factor. The classic uniform-prior optimum is (N-1)/N."""

    factor: float
    name: str = "shade"

    def choose_action(self, state: AuctionState, player, budget, rng):
        v = state.private_values[state.players.index(player)]
        bid = int(round(v * self.factor))
        return max(0, min(v, bid))


@dataclass
class TruthfulBot:
    name: str = "truthful"

    def choose_action(self, state: AuctionState, player, budget, rng):
        v = state.private_values[state.players.index(player)]
        return v


if __name__ == "__main__":
    bots = {
        "shade_50": ShadeBot(factor=0.50, name="shade_50"),
        "shade_67": ShadeBot(factor=2 / 3, name="shade_67"),  # Nash for N=3
        "shade_75": ShadeBot(factor=0.75, name="shade_75"),
        "shade_90": ShadeBot(factor=0.90, name="shade_90"),
        "truthful": TruthfulBot(),
        "random": RandomBot(),
    }
    tournament = RoundRobin(
        players=tuple(range(N_BIDDERS)),
        initial_state_factory=make_state,
        games_per_pair=20,
        time_limit_per_move=0.05,
        simultaneous=True,
        max_turns=ROUNDS + 1,
    )
    print(tournament.run(bots, seed=11).summary())
