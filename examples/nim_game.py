from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import GreedyBot, MCTSBot, MinimaxBot, RandomBot, RoundRobin


Player = int
Action = int


@dataclass(frozen=True)
class NimState:
    pile: int
    current_index: int = 0
    players: Tuple[Player, Player] = (0, 1)
    max_take: int = 3

    @property
    def current_player(self) -> Player:
        return self.players[self.current_index]

    def legal_actions(self, player: Optional[Player] = None) -> Sequence[Action]:
        if self.is_terminal():
            return []
        return tuple(range(1, min(self.max_take, self.pile) + 1))

    def apply(self, action: Action) -> "NimState":
        return NimState(
            pile=self.pile - action,
            current_index=1 - self.current_index,
            players=self.players,
            max_take=self.max_take,
        )

    def is_terminal(self) -> bool:
        return self.pile <= 0

    def score(self, player: Player) -> float:
        if not self.is_terminal():
            # During search, prefer positions where the opponent faces a multiple of 4.
            return 1.0 if self.pile % (self.max_take + 1) == 0 and self.current_player != player else 0.0
        winner = self.players[1 - self.current_index]
        return 1.0 if player == winner else 0.0


def make_state(seed: int) -> NimState:
    return NimState(pile=21 + seed % 5)


if __name__ == "__main__":
    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(),
        "minimax": MinimaxBot(depth=8),
        "mcts": MCTSBot(simulations=200, rollout_depth=30),
    }
    tournament = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=20,
        time_limit_per_move=0.05,
    )
    print(tournament.run(bots, seed=42).summary())
