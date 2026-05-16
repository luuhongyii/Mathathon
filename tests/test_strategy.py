import random
from typing import Optional, Sequence

from mathathon_kit import (
    EpsilonGreedyBanditBot,
    FictitiousPlayBot,
    RegretMatchingBot,
    TimeBudget,
)


class _DummyState:
    """Tiny adapter that exposes ``legal_actions`` for stateless picks."""

    def __init__(self, actions):
        self._actions = actions
        self.players = (0, 1)

    @property
    def current_player(self):
        return 0

    def legal_actions(self, player: Optional[int] = None) -> Sequence[int]:
        return self._actions


def test_regret_matching_converges_on_rps():
    """Against a fixed-rock opponent, regret matching should learn paper."""

    bot = RegretMatchingBot(actions=(0, 1, 2))  # rock, paper, scissors
    rng = random.Random(0)
    state = _DummyState((0, 1, 2))

    # Payoff matrix from row's perspective.
    def payoff(row, col):
        if row == col:
            return 0.0
        if (row - col) % 3 == 1:
            return 1.0
        return -1.0

    for _ in range(2000):
        action = bot.choose_action(state, 0, TimeBudget(0.01), rng)
        bot.observe(action, {a: payoff(a, 0) for a in (0, 1, 2)})

    avg = bot.average_strategy()
    # Paper should dominate.
    assert avg[1] > avg[0]
    assert avg[1] > avg[2]


def test_fictitious_play_best_responds():
    bot = FictitiousPlayBot(
        actions=(0, 1, 2),
        payoff_fn=lambda me, opp: 1.0 if (me - opp) % 3 == 1 else (0.0 if me == opp else -1.0),
    )
    state = _DummyState((0, 1, 2))
    rng = random.Random(0)

    # Opponent always rock; bot should converge to paper.
    for _ in range(20):
        bot.observe(0)
    action = bot.choose_action(state, 0, TimeBudget(0.01), rng)
    assert action == 1


def test_epsilon_bandit_picks_high_reward():
    bot = EpsilonGreedyBanditBot(actions=("a", "b", "c"), epsilon=0.0)
    bot.observe("a", 10.0)
    bot.observe("b", 1.0)
    bot.observe("c", -5.0)
    state = _DummyState(("a", "b", "c"))
    action = bot.choose_action(state, 0, TimeBudget(0.01), random.Random(0))
    assert action == "a"
