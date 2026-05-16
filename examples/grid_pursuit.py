"""Grid Pursuit-Evasion (turn-based, perfect information).

Pursuer (player 0) wants to land on the evader's square within MAX_TURNS.
Evader (player 1) wants to survive. Players alternate moves; legal moves
are 4-neighbour steps + stay, bounded by the grid.

Demonstrates a turn-based grid game suitable for Minimax / MCTS / Beam.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import GreedyBot, MCTSBot, MinimaxBot, RandomBot, RoundRobin


GRID = 6
MAX_TURNS = 30


Cell = Tuple[int, int]
Action = Cell


def _neighbours(cell: Cell) -> Sequence[Cell]:
    x, y = cell
    out = [(x, y)]  # stay
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID and 0 <= ny < GRID:
            out.append((nx, ny))
    return tuple(out)


@dataclass(frozen=True)
class PursuitState:
    pursuer: Cell
    evader: Cell
    current_index: int = 0  # 0 = pursuer's turn, 1 = evader's
    turn: int = 0
    players: Tuple[int, int] = (0, 1)

    @property
    def current_player(self) -> int:
        return self.players[self.current_index]

    def legal_actions(self, player: Optional[int] = None) -> Sequence[Action]:
        if self.is_terminal():
            return ()
        cell = self.pursuer if self.current_index == 0 else self.evader
        return _neighbours(cell)

    def apply(self, action: Action) -> "PursuitState":
        if self.current_index == 0:
            new_pursuer, new_evader = action, self.evader
        else:
            new_pursuer, new_evader = self.pursuer, action
        return PursuitState(
            pursuer=new_pursuer,
            evader=new_evader,
            current_index=1 - self.current_index,
            turn=self.turn + 1,
            players=self.players,
        )

    def is_terminal(self) -> bool:
        return self.pursuer == self.evader or self.turn >= MAX_TURNS

    def score(self, player: int) -> float:
        # Heuristic that doubles as terminal score:
        # pursuer wants distance small, evader wants it large.
        dist = abs(self.pursuer[0] - self.evader[0]) + abs(
            self.pursuer[1] - self.evader[1]
        )
        if self.is_terminal():
            if self.pursuer == self.evader:
                return 1.0 if player == self.players[0] else 0.0
            return 0.0 if player == self.players[0] else 1.0
        # During search: pursuer prefers small dist, evader prefers large.
        if player == self.players[0]:
            return -dist / (2 * GRID)
        return dist / (2 * GRID)


def make_state(seed: int) -> PursuitState:
    import random as _r
    r = _r.Random(seed)
    return PursuitState(
        pursuer=(r.randint(0, GRID - 1), r.randint(0, GRID - 1)),
        evader=(r.randint(0, GRID - 1), r.randint(0, GRID - 1)),
    )


if __name__ == "__main__":
    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(),
        "minimax3": MinimaxBot(depth=3),
        "minimax5": MinimaxBot(depth=5),
        "mcts": MCTSBot(simulations=300, rollout_depth=20),
    }
    tournament = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=10,
        time_limit_per_move=0.1,
        max_turns=MAX_TURNS + 5,
    )
    print(tournament.run(bots, seed=99).summary())
