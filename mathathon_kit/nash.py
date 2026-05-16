"""Nash equilibrium solver for 2-player zero-sum matrix games.

Computes the row player's optimal mixed strategy via linear programming.
For an m x n payoff matrix ``A`` (entry ``A[i][j]`` = row's payoff when row
plays i and column plays j), returns a probability distribution over rows
plus the value of the game.

Falls back to fictitious play if scipy isn't available so the kit can be
used in restricted environments.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Sequence, Tuple

from .core import Action, GameState, Player, TimeBudget, random_legal_action


def solve_zero_sum(
    payoff: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float], float]:
    """Return ``(row_strategy, col_strategy, value)`` for the matrix game.

    ``row_strategy[i]`` is the probability with which the row player should
    play action ``i``. ``value`` is the row player's expected payoff at
    equilibrium.

    Uses ``scipy.optimize.linprog`` when available, otherwise falls back to
    fictitious play (which converges to the value but not always to a Nash
    strategy in pathological cases).
    """

    A = [list(row) for row in payoff]
    if not A or not A[0]:
        raise ValueError("payoff matrix is empty")

    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception:
        return _fictitious_play_solver(A)

    m = len(A)
    n = len(A[0])

    # Make sure all entries are non-negative by adding a constant.  The
    # value of the game shifts by the same constant.
    flat = [v for row in A for v in row]
    shift = -min(flat) + 1.0  # strictly positive entries
    A_pos = [[v + shift for v in row] for row in A]

    # Row player: maximize v s.t. for each column j, sum_i x_i * A_pos[i][j] >= v.
    # Variables: x_1..x_m, v.  Objective: minimize -v.
    c = [0.0] * m + [-1.0]
    # Inequality constraint: -A_pos^T x + v * 1 <= 0  ⇔  A_pos^T x >= v
    A_ub = []
    for j in range(n):
        row = [-A_pos[i][j] for i in range(m)] + [1.0]
        A_ub.append(row)
    b_ub = [0.0] * n
    # Equality: sum x_i = 1
    A_eq = [[1.0] * m + [0.0]]
    b_eq = [1.0]
    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return _fictitious_play_solver(A)

    x = list(res.x[:m])
    v_pos = -res.fun
    v = v_pos - shift  # undo the shift

    # Col strategy via the dual / by best-response computation.
    col = _best_response_distribution([row for row in A], x, who="col")
    # Normalize defensively.
    x_sum = sum(x) or 1.0
    x = [max(0.0, xi) / x_sum for xi in x]
    return x, col, v


def _best_response_distribution(
    A: Sequence[Sequence[float]],
    x: Sequence[float],
    who: str = "col",
) -> List[float]:
    """Return a degenerate (pure) best-response distribution.

    Used as a column-strategy approximation when only the LP for the row
    player is solved. For symmetric games or when both players' strategies
    are required, call ``solve_zero_sum`` twice (transpose, negate).
    """
    n = len(A[0])
    m = len(A)
    if who == "col":
        # Col minimises row's payoff.
        col_value = [sum(x[i] * A[i][j] for i in range(m)) for j in range(n)]
        best_j = min(range(n), key=lambda j: col_value[j])
        out = [0.0] * n
        out[best_j] = 1.0
        return out
    raise ValueError("who must be 'col'")


def _fictitious_play_solver(
    A: Sequence[Sequence[float]],
    iterations: int = 5000,
) -> Tuple[List[float], List[float], float]:
    """Fictitious play fallback: each player best-responds to opponent's empirical mix.

    Converges to the value of the game (and to a Nash strategy on most
    games we care about). Slower than LP but dependency-free.
    """
    m = len(A)
    n = len(A[0])
    row_counts = [0] * m
    col_counts = [0] * n
    # Initialize with one play of each.
    row_counts[0] = 1
    col_counts[0] = 1

    for _ in range(iterations):
        col_total = sum(col_counts)
        col_mix = [c / col_total for c in col_counts]
        row_payoffs = [
            sum(col_mix[j] * A[i][j] for j in range(n)) for i in range(m)
        ]
        row_best = max(range(m), key=lambda i: row_payoffs[i])
        row_counts[row_best] += 1

        row_total = sum(row_counts)
        row_mix = [c / row_total for c in row_counts]
        col_payoffs = [
            sum(row_mix[i] * A[i][j] for i in range(m)) for j in range(n)
        ]
        col_best = min(range(n), key=lambda j: col_payoffs[j])
        col_counts[col_best] += 1

    row_total = sum(row_counts)
    col_total = sum(col_counts)
    row_strategy = [c / row_total for c in row_counts]
    col_strategy = [c / col_total for c in col_counts]
    value = sum(
        row_strategy[i] * col_strategy[j] * A[i][j]
        for i in range(m)
        for j in range(n)
    )
    return row_strategy, col_strategy, value


@dataclass
class NashMatrixBot(Generic[Player, Action]):
    """Plays the Nash mixed strategy of a fixed 2-player zero-sum matrix game.

    Configure with the action list and the payoff matrix from your perspective
    (rows = your actions, cols = opponent's actions). Call ``warmup()`` once
    to precompute the strategy, or let ``choose_action`` do it lazily.
    """

    actions: Sequence[Action]
    payoff_matrix: Sequence[Sequence[float]]
    name: str = "nash"
    _strategy: Optional[List[float]] = field(default=None, init=False, repr=False)
    _value: Optional[float] = field(default=None, init=False, repr=False)

    def warmup(self) -> Tuple[List[float], float]:
        if self._strategy is None:
            x, _, v = solve_zero_sum(self.payoff_matrix)
            self._strategy = x
            self._value = v
        assert self._strategy is not None and self._value is not None
        return self._strategy, self._value

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        strat, _ = self.warmup()
        legal = list(state.legal_actions(player))
        candidates = [
            (a, p) for a, p in zip(self.actions, strat) if a in legal
        ]
        if not candidates:
            return random_legal_action(state, player, rng)
        total = sum(p for _, p in candidates) or 1.0
        r = rng.random() * total
        cum = 0.0
        for a, p in candidates:
            cum += p
            if r <= cum:
                return a
        return candidates[-1][0]
