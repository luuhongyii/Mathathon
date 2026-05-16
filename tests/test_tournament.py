"""Significance helpers: tell a real improvement apart from round-robin noise."""

from examples.nim_game import make_state
from mathathon_kit import MinimaxBot, RandomBot, RoundRobin, binomial_p_value, wilson_interval
from mathathon_kit.tournament import MatchStats


def test_wilson_interval_brackets_the_rate():
    lo, hi = wilson_interval(28, 50)
    assert lo < 28 / 50 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_handles_extremes():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi < 0.5
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo > 0.5
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_binomial_p_value_separates_noise_from_signal():
    assert binomial_p_value(26, 50) > 0.5      # coin-flip-ish -> not significant
    assert binomial_p_value(45, 50) < 0.001    # lopsided -> significant
    assert binomial_p_value(0, 0) == 1.0       # no games -> nothing to claim


def test_matchstats_significance_methods():
    stats = MatchStats(games=100, wins=70.0, score_sum=70.0)
    lo, hi = stats.win_rate_ci()
    assert lo > 0.5                            # whole CI clear of the baseline
    assert stats.p_value(0.5) < 0.001


def test_significance_report_renders():
    rr = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=10,
        time_limit_per_move=0.02,
    )
    report = rr.run({"rand": RandomBot(), "mm": MinimaxBot(depth=6)}, seed=1)
    text = report.significance()
    assert "verdict" in text
    assert "significant" in text or "noise" in text


def test_significance_baseline_defaults_to_chance_win_rate():
    # A 2-seat tournament -> baseline 0.5; the report carries seats itself.
    rr = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=6,
        time_limit_per_move=0.02,
    )
    report = rr.run({"rand": RandomBot(), "mm": MinimaxBot(depth=6)}, seed=1)
    assert report.seats == 2
    assert "p_vs_0.50" in report.significance()
    assert "p_vs_0.25" in report.significance(baseline=0.25)  # override still works


def test_error_samples_survive_without_keep_results():
    class _CrashBot:
        name = "crasher"

        def choose_action(self, state, player, budget, rng):
            raise RuntimeError("boom")

    rr = RoundRobin(
        players=(0, 1),
        initial_state_factory=make_state,
        games_per_pair=4,
        time_limit_per_move=0.02,
    )
    report = rr.run({"rand": RandomBot(), "crasher": _CrashBot()}, seed=1)
    assert report.error_samples  # populated even though keep_results is False
    assert "boom" in report.error_report()
