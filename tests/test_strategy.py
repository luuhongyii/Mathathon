import random
from dataclasses import dataclass
from typing import Optional, Sequence

from mathathon_kit import (
    EpsilonGreedyBanditBot,
    FictitiousPlayBot,
    RegretMatchingBot,
    SimultaneousMCTSBot,
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


# ---------------------------------------------------------------------------
# SimultaneousMCTSBot -- decoupled-UCT for simultaneous-move games
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PickHigh:
    """Trivial 2-player simultaneous game: each round both players pick a
    number 0/1/2 and add it to their total; after ``rounds`` rounds the score
    is your total minus the opponent's. Picking 2 strictly dominates -- a
    working search must converge to it. The score is competitive (zero-sum),
    so it exercises the bot's mean-centred reward normalisation."""

    players: tuple = (0, 1)
    round: int = 0
    rounds: int = 3
    totals: tuple = (0, 0)

    def active_players(self):
        return () if self.is_terminal() else self.players

    def legal_actions(self, player):
        return (0, 1, 2)

    def apply_joint(self, actions):
        totals = tuple(self.totals[p] + actions[p] for p in self.players)
        return _PickHigh(self.players, self.round + 1, self.rounds, totals)

    def is_terminal(self):
        return self.round >= self.rounds

    def score(self, player):
        return float(self.totals[player] - self.totals[1 - player])


def test_simultaneous_mcts_finds_the_dominant_action():
    """Plain decoupled-UCT, full rollouts: the search must pick the winner."""
    bot = SimultaneousMCTSBot(simulations=400)
    move = bot.choose_action(_PickHigh(), 0, TimeBudget(1.0), random.Random(0))
    assert move == 2


def test_simultaneous_mcts_with_evaluator_and_action_filter():
    """An evaluator replaces rollouts; an action_filter prunes the branching.
    The evaluator is on the same competitive scale as score()."""
    bot = SimultaneousMCTSBot(
        simulations=300,
        evaluator=lambda s, p: float(s.totals[p] - s.totals[1 - p]),
        action_filter=lambda s, p: (0, 2),  # action 1 pruned away
    )
    move = bot.choose_action(_PickHigh(), 0, TimeBudget(1.0), random.Random(1))
    assert move == 2


def test_simultaneous_mcts_exploitative_mode_runs():
    """With an opponent_policy only the searching player gets a bandit."""
    bot = SimultaneousMCTSBot(
        simulations=300,
        opponent_policy=lambda s, p, r: r.choice(list(s.legal_actions(p))),
    )
    move = bot.choose_action(_PickHigh(), 0, TimeBudget(1.0), random.Random(2))
    assert move == 2


def test_simultaneous_mcts_respects_a_tiny_budget():
    """A search starved of time still returns a legal action, never crashes."""
    bot = SimultaneousMCTSBot(simulations=10_000_000)
    move = bot.choose_action(_PickHigh(), 0, TimeBudget(0.002), random.Random(0))
    assert move in (0, 1, 2)
