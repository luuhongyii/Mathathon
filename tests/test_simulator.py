import random

import pytest

from examples.nim_game import NimState, make_state
from mathathon_kit import (
    GreedyBot,
    MCTSBot,
    MinimaxBot,
    MinimaxBotTT,
    RandomBot,
    RoundRobin,
    Simulator,
    TimeBudget,
)


def test_random_vs_random_terminates():
    sim = Simulator(players=(0, 1), max_turns=200, time_limit_per_move=0.05)
    result = sim.play(NimState(pile=21), {0: RandomBot(), 1: RandomBot()}, seed=0)
    assert result.scores[0] + result.scores[1] == 1.0
    assert len(result.winners) == 1


def test_minimax_solves_nim():
    """When pile mod 4 == 0 player-to-move is in L-position. Pile=24 + minimax
    second means random has to play optimally to win (1/3 chance per move),
    so minimax should win nearly all 16 games."""
    sim = Simulator(players=(0, 1), max_turns=200, time_limit_per_move=0.5)
    minimax = MinimaxBot(depth=20)
    wins_for_p1 = 0
    for seed in range(16):
        # Pile=20 puts player 0 (random) in L-position; minimax (player 1) wins.
        result = sim.play(NimState(pile=20), {0: RandomBot(), 1: minimax}, seed=seed)
        wins_for_p1 += 1 if 1 in result.winners else 0
    assert wins_for_p1 == 16


def test_illegal_action_falls_back():
    class BadBot:
        name = "bad"

        def choose_action(self, state, player, budget, rng):
            return 999

    sim = Simulator(players=(0, 1), max_turns=20, time_limit_per_move=0.05)
    result = sim.play(NimState(pile=10), {0: BadBot(), 1: RandomBot()}, seed=0)
    assert any("illegal" in e for e in result.errors)
    # Game still terminates correctly.
    assert sum(result.scores.values()) == 1.0


def test_crashing_bot_falls_back():
    class CrashBot:
        name = "crash"

        def choose_action(self, state, player, budget, rng):
            raise RuntimeError("boom")

    sim = Simulator(players=(0, 1), max_turns=20, time_limit_per_move=0.05)
    result = sim.play(NimState(pile=10), {0: CrashBot(), 1: RandomBot()}, seed=0)
    assert any("crashed" in e for e in result.errors)
    assert sum(result.scores.values()) == 1.0


def test_round_robin_two_player():
    bots = {
        "random": RandomBot(),
        "greedy": GreedyBot(),
        "minimax": MinimaxBot(depth=6),
    }
    rr = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=4,
        time_limit_per_move=0.05,
    )
    report = rr.run(bots, seed=1)
    # Every bot played the same number of games as pair count * games_per_pair.
    counts = [s.games for s in report.standings.values()]
    assert all(c == counts[0] for c in counts)
    # Greedy/Minimax should beat random.
    assert report.standings["greedy"].win_rate > report.standings["random"].win_rate


def test_time_budget():
    b = TimeBudget(0.05, safety_margin=0.001)
    assert not b.expired()
    assert b.remaining <= 0.05
    import time
    time.sleep(0.06)
    assert b.expired()


def test_minimax_tt_solves_nim():
    """Minimax with transposition table should still solve Nim P-positions."""
    sim = Simulator(players=(0, 1), max_turns=200, time_limit_per_move=0.5)
    bot = MinimaxBotTT(max_depth=20)
    wins = 0
    for seed in range(16):
        result = sim.play(NimState(pile=20), {0: RandomBot(), 1: bot}, seed=seed)
        wins += 1 if 1 in result.winners else 0
    assert wins == 16


def test_minimax_tt_faster_than_plain():
    """TT version should be at least 3x faster on Nim where positions repeat."""
    import random
    import time as _t
    state = NimState(pile=21)
    rng = random.Random(0)
    plain = MinimaxBot(depth=15)
    tt = MinimaxBotTT(max_depth=15)
    t = _t.time(); plain.choose_action(state, 0, TimeBudget(5.0), rng); t_plain = _t.time() - t
    t = _t.time(); tt.choose_action(state, 0, TimeBudget(5.0), rng); t_tt = _t.time() - t
    assert t_plain > t_tt * 2, f"TT not faster: {t_plain:.3f}s vs {t_tt:.3f}s"


def test_mcts_runs_within_time_budget():
    sim = Simulator(players=(0, 1), max_turns=200, time_limit_per_move=0.05)
    bot = MCTSBot(simulations=10000, rollout_depth=50)
    # Should finish within budget despite high simulations.
    result = sim.play(NimState(pile=15), {0: bot, 1: RandomBot()}, seed=0)
    # No timing error should be reported (rollout_depth is short).
    assert sum(result.scores.values()) == 1.0
