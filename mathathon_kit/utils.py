"""Reusable building blocks for evaluators and search loops.

- ``IterativeDeepeningMinimax``: Minimax with re-deepening until time runs out.
  Returns the deepest fully-completed depth's best action.
- ``normalize_evaluator``: wraps a heuristic to clip it into ``[-1, 1]`` so
  MCTS UCB tuning constants stay sane.
- ``compose_evaluators``: weighted sum of multiple evaluators, useful when
  you want "win rate proxy + material count + mobility" combined.
- ``with_legality_guard``: wraps a bot's ``choose_action`` to always return
  a legal action (extra safety on top of the simulator's fallback).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence

from .bots import GreedyBot, MinimaxBot
from .core import Action, GameState, Player, TimeBudget, random_legal_action


Evaluator = Callable[[GameState, object], float]


def normalize_evaluator(
    evaluator: Evaluator,
    lower: float = -1.0,
    upper: float = 1.0,
    estimated_min: float = -1.0,
    estimated_max: float = 1.0,
) -> Evaluator:
    """Clip evaluator output into ``[lower, upper]`` after rescaling.

    Pass realistic ``estimated_min`` / ``estimated_max`` from your domain to
    get a meaningful linear rescale; the final value is always clipped.
    """

    span = max(1e-9, estimated_max - estimated_min)

    def _eval(state, player):
        v = evaluator(state, player)
        scaled = lower + (v - estimated_min) / span * (upper - lower)
        return max(lower, min(upper, scaled))

    return _eval


def compose_evaluators(
    *weighted: tuple[Evaluator, float],
) -> Evaluator:
    """Linear combination of evaluators: ``sum(w_i * eval_i)``."""

    def _eval(state, player):
        total = 0.0
        for fn, w in weighted:
            total += w * fn(state, player)
        return total

    return _eval


@dataclass
class IterativeDeepeningMinimax(Generic[Player, Action]):
    """Run minimax at increasing depths until budget expires.

    Always returns the best action from the deepest fully-completed search,
    so partial searches at the next depth don't corrupt the move choice.
    """

    max_depth: int = 32
    evaluator: Evaluator = lambda s, p: s.score(p)
    name: str = "id-minimax"

    def choose_action(
        self,
        state: GameState,
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        if not legal:
            return random_legal_action(state, player, rng)

        best_action = legal[0]
        for d in range(1, self.max_depth + 1):
            if budget.expired():
                break
            # Reserve at least 30% of remaining budget for completing this depth
            # so we don't bail out mid-iteration with a worse move than depth-1.
            inner_budget = budget.fraction(0.7)
            bot = MinimaxBot(depth=d, evaluator=self.evaluator)
            try:
                candidate = bot.choose_action(state, player, inner_budget, rng)
                if candidate in legal:
                    best_action = candidate
            except Exception:
                break
        return best_action


def with_legality_guard(bot, name: Optional[str] = None):
    """Wrap a bot so it always emits an action in ``state.legal_actions``."""

    class _Guarded:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.name = name or getattr(inner, "name", "guarded")

        def choose_action(self, state, player, budget, rng):
            legal = list(state.legal_actions(player))
            if not legal:
                raise ValueError("no legal actions")
            try:
                action = self.inner.choose_action(state, player, budget, rng)
            except Exception:
                return rng.choice(legal)
            return action if action in legal else rng.choice(legal)

    return _Guarded(bot)
