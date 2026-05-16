from examples.beauty_contest import BeautyState, LevelKBot, ConstantBot, make_state as bc_state
from examples.colonel_blotto import (
    BlottoState,
    UniformBlottoBot,
    SkewedBlottoBot,
    BestSampleBot,
    make_state as bl_state,
)
from examples.iterated_pd import IPDState, _AlwaysBot, COOP, DEFECT, make_state as ipd_state
from mathathon_kit import RandomBot, RoundRobin, SimultaneousSimulator


def test_simultaneous_simulator_runs():
    sim = SimultaneousSimulator(players=(0, 1), max_rounds=50, time_limit_per_move=0.05)
    bots = {0: _AlwaysBot(COOP, "c"), 1: _AlwaysBot(DEFECT, "d")}
    result = sim.play(IPDState(rounds_left=20), bots, seed=0)
    # Always-Defect should score strictly more than always-coop.
    assert result.scores[1] > result.scores[0]


def test_n_player_round_robin():
    bots = {
        "z": ConstantBot(value=0, name="z"),
        "l1": LevelKBot(k=1, name="l1"),
        "l2": LevelKBot(k=2, name="l2"),
        "r": RandomBot(),
    }
    rr = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=bc_state,
        games_per_pair=2,
        time_limit_per_move=0.05,
        simultaneous=True,
        max_turns=2,
    )
    report = rr.run(bots, seed=0)
    assert all(s.games > 0 for s in report.standings.values())
    # l2 typically the strongest in beauty contest with these opponents.
    assert report.standings["l2"].win_rate >= report.standings["z"].win_rate


def test_blotto_terminal_score_sums_to_n_fields():
    state = BlottoState(placed=((10, 0, 0, 0), (3, 3, 2, 2)))
    s0 = state.score(0)
    s1 = state.score(1)
    assert s0 + s1 == state.n_fields


def test_blotto_uniform_alloc_legal():
    state = BlottoState()
    bot = UniformBlottoBot()
    import random as _r
    action = bot.choose_action(state, 0, None, _r.Random(0))
    assert sum(action) == state.total
    assert len(action) == state.n_fields
