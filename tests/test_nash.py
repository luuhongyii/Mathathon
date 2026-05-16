import random

import pytest

from mathathon_kit import NashMatrixBot, TimeBudget, solve_zero_sum


def test_rps_nash_is_uniform():
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    x, _, v = solve_zero_sum(rps)
    assert all(abs(p - 1 / 3) < 1e-3 for p in x), x
    assert abs(v) < 1e-6


def test_value_correctness_simple_2x2():
    """Matching pennies: row plays H/T, col plays H/T.
    Row payoffs = [[1, -1], [-1, 1]]. Nash = (0.5, 0.5), value = 0."""
    A = [[1, -1], [-1, 1]]
    x, _, v = solve_zero_sum(A)
    assert abs(x[0] - 0.5) < 1e-3
    assert abs(v) < 1e-6


def test_dominated_strategy_pruned():
    """Row 1 dominates row 0; Nash should put all weight on row 1."""
    A = [[0, 0], [1, 1]]
    x, _, v = solve_zero_sum(A)
    assert x[1] > 0.99
    assert abs(v - 1.0) < 1e-6


def test_nash_bot_plays_rps_uniformly():
    class _RPSState:
        players = (0, 1)
        @property
        def current_player(self):
            return 0
        def legal_actions(self, player=None):
            return [0, 1, 2]

    bot = NashMatrixBot(
        actions=[0, 1, 2],
        payoff_matrix=[[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
    )
    counts = [0, 0, 0]
    rng = random.Random(0)
    for _ in range(3000):
        a = bot.choose_action(_RPSState(), 0, TimeBudget(0.05), rng)
        counts[a] += 1
    # Each action should be picked roughly 1/3 of the time.
    for c in counts:
        assert 800 < c < 1200, counts
