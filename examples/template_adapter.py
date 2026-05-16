"""COPY ME when the rules drop. Fill-in-the-blanks game adapter + smoke test.

    cp examples/template_adapter.py examples/my_game.py

It ships as a runnable trivial game (a 1..3 countdown) so the smoke test is
green from the start -- replace the TODO bodies with your real rules and keep
running it.

------------------------------------------------------------------------------
OPENING CHECKLIST (first 30 minutes)
------------------------------------------------------------------------------
[ ] 1. Read the rules. Note: #players, turn-based vs simultaneous, the exact
       win/score condition, the per-move time limit, the wire protocol.
[ ] 2. Fill in the state adapter below (TODO markers).
[ ] 3. Run THIS file: RandomBot self-play must finish with no errors.
[ ] 4. Pick the engine (README "Competition-Day Playbook" step 3).
[ ] 5. Wire up the stdio submission (see examples/platform_submission_nim.py):
       - one line in / one line out      -> run_per_move_loop
       - handshake / multi-line / sentinel -> run_protocol_loop
       ALWAYS set IOConfig(fallback=random_legal_fallback(parse_state)).
[ ] 6. If the judge accepts one file only: python tools/bundle.py my_game.py
------------------------------------------------------------------------------
GOTCHAS
- Make the state a frozen dataclass with tuple (not list) fields, so it is
  hashable -- MinimaxBotTT's transposition table silently no-ops otherwise.
- `apply` must return a NEW state, never mutate self.
- `score` must be signed from the argument player's view, the SAME way at
  terminal and non-terminal nodes, or minimax/MCTS will optimise backwards.
- For a simultaneous-move game implement SimultaneousState instead -- see the
  commented sketch at the bottom and examples/iterated_pd.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import GreedyBot, MinimaxBot, RandomBot, RoundRobin, validate_adapter

Player = int
Action = int


@dataclass(frozen=True)
class GameAdapter:
    """TODO: rename to your game and replace every field + method body."""

    # TODO: replace with your real state fields (use tuples, not lists).
    remaining: int = 21
    current_index: int = 0
    players: Tuple[Player, Player] = (0, 1)

    @property
    def current_player(self) -> Player:
        """Whose turn it is."""
        return self.players[self.current_index]

    def legal_actions(self, player: Optional[Player] = None) -> Sequence[Action]:
        """Actions available now. Return an empty sequence at a terminal state."""
        # TODO: replace with your real move generation.
        if self.is_terminal():
            return ()
        return tuple(range(1, min(3, self.remaining) + 1))

    def apply(self, action: Action) -> "GameAdapter":
        """Return the NEXT state. Must NOT mutate self."""
        # TODO: replace with your real transition.
        return GameAdapter(
            remaining=self.remaining - action,
            current_index=1 - self.current_index,
            players=self.players,
        )

    def is_terminal(self) -> bool:
        """True once the game is over."""
        # TODO: replace with your real end condition.
        return self.remaining <= 0

    def score(self, player: Player) -> float:
        """Signed value from ``player``'s view. At a terminal node return the
        true result; at a non-terminal node return a heuristic estimate."""
        # TODO: replace with your real score / heuristic.
        if self.is_terminal():
            winner = self.players[1 - self.current_index]
            return 1.0 if player == winner else 0.0
        return 0.0


def make_state(seed: int) -> GameAdapter:
    """Initial state for game ``seed`` (vary it so tournaments aren't identical)."""
    return GameAdapter(remaining=21 + seed % 5)


if __name__ == "__main__":
    # Step 1: prove the adapter itself is correct (mutation, hashability,
    # legality, termination). Fix every ERROR before reading any tournament.
    checks = validate_adapter(make_state(0))
    print(checks.summary())
    print()

    # Step 2: smoke-test with self-play -- errors must stay 0.
    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(),
        "minimax": MinimaxBot(depth=6),
    }
    report = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=10,
        time_limit_per_move=0.05,
    ).run(bots, seed=1)
    print(report.summary())
    total_errors = sum(s.errors for s in report.standings.values())
    print(f"\nadapter smoke test: {'OK' if checks.ok and total_errors == 0 else 'FIX THE ADAPTER'}")


# -----------------------------------------------------------------------------
# Simultaneous-move variant (matrix games, Blotto, auctions, iterated PD...)
# Implement this Protocol instead of the one above; see examples/iterated_pd.py.
#
# @dataclass(frozen=True)
# class SimAdapter:
#     players: Tuple[Player, Player] = (0, 1)
#     def active_players(self) -> Sequence[Player]: ...
#     def legal_actions(self, player: Player) -> Sequence[Action]: ...
#     def apply_joint(self, actions: Mapping[Player, Action]) -> "SimAdapter": ...
#     def is_terminal(self) -> bool: ...
#     def score(self, player: Player) -> float: ...
#
# Then run RoundRobin(..., simultaneous=True).
# -----------------------------------------------------------------------------
