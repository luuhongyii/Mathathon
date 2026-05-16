import random

from examples.nim_game import NimState
from mathathon_kit import (
    IterativeDeepeningMinimax,
    RandomBot,
    Simulator,
    TimeBudget,
    compose_evaluators,
    normalize_evaluator,
    with_legality_guard,
)


def test_iterative_deepening_beats_random():
    sim = Simulator(players=(0, 1), max_turns=200, time_limit_per_move=0.1)
    bot = IterativeDeepeningMinimax(max_depth=20)
    # Pile=21 puts player 0 (random) in a winning P-position. The bot only
    # wins when player 0 makes a mistake, which random does ~2/3 of the time.
    # Out of 16 seeds we expect ~10 wins; require at least 8 to be robust.
    wins = 0
    for seed in range(16):
        result = sim.play(NimState(pile=21), {0: RandomBot(), 1: bot}, seed=seed)
        wins += 1 if 1 in result.winners else 0
    assert wins >= 8


def test_normalize_clips():
    raw = lambda s, p: 100.0
    bounded = normalize_evaluator(raw, lower=-1, upper=1, estimated_min=-10, estimated_max=10)
    assert bounded(None, 0) == 1.0


def test_compose_weighted_sum():
    evaluator = compose_evaluators(
        ((lambda s, p: 1.0), 2.0),
        ((lambda s, p: 5.0), 0.5),
    )
    assert evaluator(None, 0) == 2.0 * 1.0 + 0.5 * 5.0


def test_legality_guard():
    class BadBot:
        name = "bad"

        def choose_action(self, state, player, budget, rng):
            return 999

    guarded = with_legality_guard(BadBot())
    state = NimState(pile=10)
    action = guarded.choose_action(state, 0, TimeBudget(0.05), random.Random(0))
    assert action in state.legal_actions(0)
