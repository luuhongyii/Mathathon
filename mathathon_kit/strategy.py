"""Bots for repeated / matrix / hidden-information games.

These complement the search bots in ``bots.py``:

- ``RegretMatchingBot``: online no-regret learning over a fixed action set,
  Hannan-consistent for repeated matrix games.
- ``FictitiousPlayBot``: best-responds to the empirical mix of opponents'
  moves so far.
- ``EpsilonGreedyBanditBot``: classic UCB/eps-greedy pick over actions when
  payoffs are observed but the game is non-strategic.
- ``ISMCTSBot``: information-set MCTS via determinization. Use when there is
  hidden information and you can sample plausible full states.
- Classic IPD strategies: ``TitForTatBot``, ``GrimTriggerBot``, ``PavlovBot``.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Mapping, Optional, Sequence

from .core import Action, GameState, Player, TimeBudget, random_legal_action


# ---------------------------------------------------------------------------
# Repeated / matrix games
# ---------------------------------------------------------------------------


@dataclass
class RegretMatchingBot(Generic[Player, Action]):
    """Counterfactual regret matching for repeated games with fixed actions.

    Call ``observe(action_played, payoffs_per_action)`` after each round to
    update regrets. ``payoffs_per_action`` is a mapping from action -> the
    reward you would have received had you played that action (counterfactual).

    For symmetric matrix games where you only see the opponent's action, you
    can compute counterfactuals via the known payoff matrix.
    """

    actions: Sequence[Action]
    name: str = "regret-matching"
    regret_sum: Dict[Action, float] = field(default_factory=dict)
    strategy_sum: Dict[Action, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.actions:
            self.regret_sum.setdefault(a, 0.0)
            self.strategy_sum.setdefault(a, 0.0)

    def _current_strategy(self) -> Dict[Action, float]:
        positive = {a: max(0.0, self.regret_sum[a]) for a in self.actions}
        total = sum(positive.values())
        if total <= 0:
            n = len(self.actions)
            return {a: 1.0 / n for a in self.actions}
        return {a: v / total for a, v in positive.items()}

    def average_strategy(self) -> Dict[Action, float]:
        total = sum(self.strategy_sum.values())
        if total <= 0:
            n = len(self.actions)
            return {a: 1.0 / n for a in self.actions}
        return {a: v / total for a, v in self.strategy_sum.items()}

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        strat = self._current_strategy()
        # Sample only from currently legal actions, renormalising.
        candidates = [a for a in self.actions if a in legal]
        if not candidates:
            return rng.choice(legal)
        weights = [max(1e-9, strat.get(a, 0.0)) for a in candidates]
        total = sum(weights)
        weights = [w / total for w in weights]
        for a, w in zip(candidates, weights):
            self.strategy_sum[a] = self.strategy_sum.get(a, 0.0) + w
        r = rng.random()
        cum = 0.0
        for a, w in zip(candidates, weights):
            cum += w
            if r <= cum:
                return a
        return candidates[-1]

    def observe(
        self,
        action_played: Action,
        payoffs_per_action: Mapping[Action, float],
    ) -> None:
        actual = payoffs_per_action.get(action_played, 0.0)
        for a, payoff in payoffs_per_action.items():
            self.regret_sum[a] = self.regret_sum.get(a, 0.0) + (payoff - actual)


@dataclass
class FictitiousPlayBot(Generic[Player, Action]):
    """Best-respond to the opponent's empirical action distribution.

    ``payoff_fn(my_action, opponent_action) -> reward`` describes the
    1-shot game. Call ``observe(opponent_action)`` each round.
    """

    actions: Sequence[Action]
    payoff_fn: Callable[[Action, Action], float]
    name: str = "fictitious-play"
    counts: Dict[Action, int] = field(default_factory=lambda: defaultdict(int))

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        candidates = [a for a in self.actions if a in legal] or legal
        total = sum(self.counts.values())
        if total == 0:
            return rng.choice(candidates)
        best_action = candidates[0]
        best_value = -math.inf
        for a in candidates:
            ev = sum(
                (self.counts[opp] / total) * self.payoff_fn(a, opp)
                for opp in self.counts
            )
            if ev > best_value:
                best_value = ev
                best_action = a
        return best_action

    def observe(self, opponent_action: Action) -> None:
        self.counts[opponent_action] += 1


# ---------------------------------------------------------------------------
# Classic IPD strategies (cheap baselines worth keeping in MetaBot)
# ---------------------------------------------------------------------------


@dataclass
class TitForTatBot(Generic[Player, Action]):
    """Cooperate first, then mirror opponent's previous move.

    Provide ``cooperate`` and ``defect`` action values matching your adapter.
    ``observe(opponent_action)`` is called by the harness or your wrapper.
    """

    cooperate: Action
    defect: Action
    name: str = "tit-for-tat"
    last_opponent: Optional[Action] = None

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        chosen = self.last_opponent if self.last_opponent is not None else self.cooperate
        return chosen if chosen in legal else legal[0]

    def observe(self, opponent_action: Action) -> None:
        self.last_opponent = opponent_action


@dataclass
class GrimTriggerBot(Generic[Player, Action]):
    cooperate: Action
    defect: Action
    name: str = "grim"
    triggered: bool = False

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        chosen = self.defect if self.triggered else self.cooperate
        return chosen if chosen in legal else legal[0]

    def observe(self, opponent_action: Action) -> None:
        if opponent_action == self.defect:
            self.triggered = True


@dataclass
class PavlovBot(Generic[Player, Action]):
    """Win-stay, lose-shift (a.k.a. Pavlov)."""

    cooperate: Action
    defect: Action
    name: str = "pavlov"
    last_self: Optional[Action] = None
    last_opponent: Optional[Action] = None

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        if self.last_self is None:
            chosen = self.cooperate
        elif self.last_self == self.last_opponent:
            chosen = self.last_self
        else:
            chosen = self.defect if self.last_self == self.cooperate else self.cooperate
        if chosen not in legal:
            chosen = legal[0]
        self.last_self = chosen
        return chosen

    def observe(self, opponent_action: Action) -> None:
        self.last_opponent = opponent_action


# ---------------------------------------------------------------------------
# Bandit-style picker for stateless decisions
# ---------------------------------------------------------------------------


@dataclass
class EpsilonGreedyBanditBot(Generic[Player, Action]):
    actions: Sequence[Action]
    epsilon: float = 0.1
    name: str = "eps-bandit"
    rewards: Dict[Action, float] = field(default_factory=dict)
    counts: Dict[Action, int] = field(default_factory=dict)

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        candidates = [a for a in self.actions if a in legal] or legal
        if rng.random() < self.epsilon:
            return rng.choice(candidates)

        def mean(a: Action) -> float:
            c = self.counts.get(a, 0)
            return self.rewards.get(a, 0.0) / c if c else 0.0

        return max(candidates, key=mean)

    def observe(self, action: Action, reward: float) -> None:
        self.counts[action] = self.counts.get(action, 0) + 1
        self.rewards[action] = self.rewards.get(action, 0.0) + reward


# ---------------------------------------------------------------------------
# Information-set MCTS (hidden information via determinization)
# ---------------------------------------------------------------------------


@dataclass
class _ISNode(Generic[Player, Action]):
    parent: Optional["_ISNode[Player, Action]"] = None
    mover: Optional[Player] = None
    action: Optional[Action] = None
    children: Dict[Action, "_ISNode[Player, Action]"] = field(default_factory=dict)
    visits: int = 0
    value: float = 0.0
    availability: int = 0  # how many determinizations exposed this node


@dataclass
class ISMCTSBot(Generic[Player, Action]):
    """Information-set MCTS by determinization.

    ``determinize(public_state, player, rng)`` should return a concrete
    ``GameState`` consistent with what ``player`` knows. The bot runs MCTS
    on a fresh determinization each iteration.
    """

    determinize: Callable[[Any, Player, random.Random], GameState[Player, Action]]
    simulations: int = 400
    rollout_depth: int = 80
    exploration: float = 0.7
    name: str = "ismcts"

    def choose_action(
        self,
        state: Any,
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        root = _ISNode[Player, Action](mover=None)
        legal_root: List[Action] = []
        first_state = self.determinize(state, player, rng)
        legal_root = list(first_state.legal_actions(player))
        if not legal_root:
            return random_legal_action(first_state, player, rng)

        for sim in range(self.simulations):
            if budget.expired():
                break
            sampled_state = first_state if sim == 0 else self.determinize(state, player, rng)
            self._iterate(root, sampled_state, player, rng, budget)

        if not root.children:
            return rng.choice(legal_root)
        return max(root.children.items(), key=lambda kv: kv[1].visits)[0]

    def _iterate(
        self,
        root: _ISNode[Player, Action],
        state: GameState[Player, Action],
        root_player: Player,
        rng: random.Random,
        budget: TimeBudget,
    ) -> None:
        node = root
        path: List[_ISNode[Player, Action]] = [root]
        # Selection / expansion.
        while not state.is_terminal() and not budget.expired():
            actions = list(state.legal_actions(state.current_player))
            if not actions:
                break
            for a in actions:
                child = node.children.get(a)
                if child is None:
                    continue
                child.availability += 1
            untried = [a for a in actions if a not in node.children]
            if untried:
                a = rng.choice(untried)
                mover = state.current_player
                child = _ISNode[Player, Action](
                    parent=node, mover=mover, action=a
                )
                node.children[a] = child
                path.append(child)
                state = state.apply(a)
                node = child
                break  # then rollout from here
            # All explored — UCB select.
            best = max(actions, key=lambda a: self._ucb(node.children[a]))
            state = state.apply(best)
            node = node.children[best]
            path.append(node)

        # Rollout.
        depth = 0
        while not state.is_terminal() and depth < self.rollout_depth and not budget.expired():
            actions = list(state.legal_actions(state.current_player))
            if not actions:
                break
            state = state.apply(rng.choice(actions))
            depth += 1

        # Backprop using each node's mover perspective.
        players = getattr(state, "players", None)
        scores = {p: state.score(p) for p in players} if players else {}
        default = state.score(root_player)
        for n in path:
            n.visits += 1
            if n.mover is None:
                n.value += default
            else:
                n.value += scores.get(n.mover, default)

    def _ucb(self, node: _ISNode[Player, Action]) -> float:
        if node.visits == 0:
            return math.inf
        availability = max(1, node.availability)
        return (node.value / node.visits) + self.exploration * math.sqrt(
            math.log(availability) / node.visits
        )
