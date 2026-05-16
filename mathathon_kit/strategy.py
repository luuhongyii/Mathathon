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


# ---------------------------------------------------------------------------
# Decoupled-UCT for simultaneous-move games (optionally hidden information)
# ---------------------------------------------------------------------------


@dataclass
class _DUCTNode:
    """One tree node for decoupled UCT. Each active player gets an independent
    UCB bandit over its own actions -- ``stats[player][action]`` is the list
    ``[visits, value_sum, availability]``. Children are keyed by the joint move."""

    children: Dict[Any, "_DUCTNode"] = field(default_factory=dict)
    stats: Dict[Any, Dict[Any, List[float]]] = field(default_factory=dict)


@dataclass
class SimultaneousMCTSBot(Generic[Player, Action]):
    """Decoupled-UCT (DUCT) search for **simultaneous-move** games.

    The search bots in ``bots.py`` (Minimax / MCTS / Beam) and ``ISMCTSBot``
    above are all turn-based -- they call ``state.current_player`` /
    ``state.apply``. This bot is the simultaneous-game counterpart: it drives
    the ``SimultaneousState`` interface (``active_players`` / ``legal_actions``
    / ``apply_joint``).

    Each node keeps an *independent* UCB1 bandit per player ("decoupled" UCT);
    a joint move is formed by letting every active player select on its own.
    This is the standard tractable substitute for solving the per-node
    simultaneous subgame exactly. That is *equilibrium-flavoured* search: every
    player is modelled as best-responding. For games where your edge comes from
    exploiting imperfect opponents (you need a counterparty who plays badly),
    pass an ``opponent_policy`` instead -- then only the searching player gets
    a bandit and opponents are sampled from that policy, turning the search
    into an exploitative best response against a fixed opponent model.

    Hidden information: pass a ``determinize(state, player, rng)`` sampler that
    returns a full ``SimultaneousState`` consistent with what ``player`` can
    observe. Every simulation re-determinizes, so the shared tree aggregates
    over the belief -- determinized DUCT. The availability count keeps UCB
    calibrated when the legal-action set varies across determinizations.

    Rewards are min-max normalised across players per rollout, so the
    ``exploration`` constant stays calibrated regardless of score magnitude.
    Tuning hooks, in rough order of impact:

    - ``evaluator(state, player) -> float``: a truncated leaf estimate used
      instead of a rollout. Decisive for long games where one move's effect
      would otherwise be lost in rollout noise.
    - ``action_filter(state, player)``: prune the branching to a curated
      candidate set when the raw legal-action set is too large for the budget.
    - ``rollout_policy(state, player, rng)``: informed playouts, used only
      when no ``evaluator`` is given.
    - ``opponent_policy(state, player, rng)``: switches the search from
      equilibrium DUCT to exploitative best response (see above).
    """

    determinize: Optional[Callable[[Any, Player, random.Random], Any]] = None
    evaluator: Optional[Callable[[Any, Player], float]] = None
    opponent_policy: Optional[Callable[[Any, Player, random.Random], Action]] = None
    rollout_policy: Optional[Callable[[Any, Player, random.Random], Action]] = None
    action_filter: Optional[Callable[[Any, Player], Sequence[Action]]] = None
    simulations: int = 400
    rollout_depth: int = 40
    exploration: float = 1.0
    name: str = "sim-mcts"

    def _candidates(self, state: Any, player: Player) -> List[Action]:
        """Actions the tree branches on. ``action_filter`` prunes the search to
        a curated subset -- essential when the raw legal-action set is large
        (it must return only legal actions). Rollouts still use the full set."""
        if self.action_filter is not None:
            return list(self.action_filter(state, player))
        return list(state.legal_actions(player))

    def choose_action(
        self,
        state: Any,
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        legal = list(state.legal_actions(player))
        if not legal:
            raise ValueError(f"player {player!r} has no legal actions")
        if len(legal) == 1:
            return legal[0]

        # Stop at 80% of the move budget. One overrun forfeits a game, and a
        # GC pause can dwarf a single fast simulation, so a relative headroom
        # is not enough -- reserve an absolute slice of the total limit.
        deadline = 0.8 * getattr(budget, "seconds", float("inf"))
        root = _DUCTNode()
        for sim in range(self.simulations):
            if budget.expired() or budget.elapsed >= deadline:
                break
            sampled = self.determinize(state, player, rng) if self.determinize else state
            self._iterate(root, sampled, player, rng)

        pstats = root.stats.get(player)
        if not pstats:
            return rng.choice(legal)
        ranked = [(a, st) for a, st in pstats.items() if a in legal]
        if not ranked:
            return rng.choice(legal)
        # Most-visited root action -- the robust MCTS choice.
        return max(ranked, key=lambda kv: kv[1][0])[0]

    def _iterate(
        self, root: _DUCTNode, state: Any, root_player: Player, rng: random.Random
    ) -> None:
        node = root
        path: List[tuple] = []  # (node, joint-action dict) chosen at that node
        while not state.is_terminal():
            actives = list(state.active_players())
            joint: Dict[Any, Any] = {}
            for p in actives:
                if self.opponent_policy is not None and p != root_player:
                    # Exploitative mode: opponents are a fixed sampled policy,
                    # not co-optimised bandits.
                    legal = list(state.legal_actions(p))
                    if not legal:
                        continue
                    a = self.opponent_policy(state, p, rng)
                    joint[p] = a if a in legal else rng.choice(legal)
                    continue
                la = self._candidates(state, p)
                if not la:
                    continue
                pstats = node.stats.setdefault(p, {})
                for a in la:
                    pstats.setdefault(a, [0.0, 0.0, 0.0])
                    pstats[a][2] += 1  # availability
                joint[p] = self._select(pstats, la, rng)
            if not joint:
                break
            path.append((node, joint))
            key = tuple(sorted(joint.items(), key=lambda kv: repr(kv[0])))
            child = node.children.get(key)
            fresh = child is None
            if fresh:
                child = _DUCTNode()
                node.children[key] = child
            state = state.apply_joint(joint)
            node = child
            if fresh:
                break  # expand one node per simulation, then roll out

        # Estimate the leaf value. An ``evaluator`` gives a low-variance
        # truncated estimate -- decisive when a long random rollout would
        # otherwise drown a single move's effect in noise. Without one, roll
        # out to a terminal state (optionally with an informed rollout_policy).
        players = list(getattr(state, "players", []))
        if not players:
            return
        if state.is_terminal():
            scores = {p: float(state.score(p)) for p in players}
        elif self.evaluator is not None:
            scores = {p: float(self.evaluator(state, p)) for p in players}
        else:
            depth = 0
            while not state.is_terminal() and depth < self.rollout_depth:
                actives = list(state.active_players())
                joint = {}
                for p in actives:
                    la = list(state.legal_actions(p))
                    if not la:
                        continue
                    if self.rollout_policy is not None:
                        a = self.rollout_policy(state, p, rng)
                        joint[p] = a if a in la else rng.choice(la)
                    else:
                        joint[p] = rng.choice(la)
                if not joint:
                    break
                state = state.apply_joint(joint)
                depth += 1
            players = list(getattr(state, "players", []))
            scores = {p: float(state.score(p)) for p in players}
        if not players:
            return

        # Mean-centred normalisation. Subtracting the per-simulation mean is a
        # control variate: a lucky determinization that lifts every player's
        # score cancels out, so the reward reflects how much *this action* beat
        # the field rather than how good the sampled world was. Without it the
        # determinization variance swamps a single move's signal. Dividing by
        # the spread keeps UCB's ``exploration`` term calibrated.
        span = max(scores.values()) - min(scores.values())
        scale = span if span > 1e-12 else 1.0
        mean = sum(scores.values()) / len(scores)
        norm = {p: 0.5 + 0.5 * (scores[p] - mean) / scale for p in players}
        for nd, joint in path:
            for p, a in joint.items():
                pstats = nd.stats.get(p)
                if pstats is None or a not in pstats:
                    continue  # opponent-policy players carry no bandit stats
                pstats[a][0] += 1.0
                pstats[a][1] += norm.get(p, 0.5)

    def _select(
        self,
        pstats: Dict[Any, List[float]],
        legal: List[Action],
        rng: random.Random,
    ) -> Action:
        """UCB1 over one player's legal actions; unvisited actions first.

        Ties -- and the choice among unvisited actions -- are broken at
        random. Deterministic tie-breaking would make two players with
        identical stats in a symmetric game pick identically every time, so
        the search would only ever explore the diagonal of the joint-action
        space. Random draws keep the bandits independent."""
        untried = [a for a in legal if pstats[a][0] == 0.0]
        if untried:
            return rng.choice(untried)
        best_u = -math.inf
        best: List[Action] = []
        for a in legal:
            visits, value, avail = pstats[a]
            u = (value / visits) + self.exploration * math.sqrt(
                math.log(max(2.0, avail)) / visits
            )
            if u > best_u + 1e-12:
                best_u, best = u, [a]
            elif u > best_u - 1e-12:
                best.append(a)
        return rng.choice(best) if best else legal[0]
