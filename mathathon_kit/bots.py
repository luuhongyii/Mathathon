from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from .core import Action, GameState, Player, TimeBudget, random_legal_action, require_players


Evaluator = Callable[[GameState[Player, Action], Player], float]


def default_evaluator(state: GameState[Player, Action], player: Player) -> float:
    return state.score(player)


class RandomBot(Generic[Player, Action]):
    name = "random"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        return random_legal_action(state, player, rng)


@dataclass
class GreedyBot(Generic[Player, Action]):
    evaluator: Evaluator[Player, Action] = default_evaluator
    noise: float = 0.0
    name: str = "greedy"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        best_action: Optional[Action] = None
        best_value = -math.inf

        for action in state.legal_actions(player):
            value = self.evaluator(state.apply(action), player)
            if self.noise:
                value += rng.uniform(-self.noise, self.noise)
            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            return random_legal_action(state, player, rng)
        return best_action


@dataclass
class MinimaxBot(Generic[Player, Action]):
    depth: int = 3
    evaluator: Evaluator[Player, Action] = default_evaluator
    max_actions: Optional[int] = None
    name: str = "minimax"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        players = require_players(state)
        if len(players) != 2:
            return GreedyBot(self.evaluator).choose_action(state, player, budget, rng)

        best_action: Optional[Action] = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        actions = list(state.legal_actions(player))
        rng.shuffle(actions)
        for action in actions[: self.max_actions]:
            if budget.expired():
                break
            value = self._value(state.apply(action), player, self.depth - 1, alpha, beta, budget)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action if best_action is not None else random_legal_action(state, player, rng)

    def _value(
        self,
        state: GameState[Player, Action],
        root_player: Player,
        depth: int,
        alpha: float,
        beta: float,
        budget: TimeBudget,
    ) -> float:
        if depth <= 0 or state.is_terminal() or budget.expired():
            return self.evaluator(state, root_player)

        actions = list(state.legal_actions(state.current_player))
        if not actions:
            return self.evaluator(state, root_player)

        if state.current_player == root_player:
            value = -math.inf
            for action in actions[: self.max_actions]:
                value = max(value, self._value(state.apply(action), root_player, depth - 1, alpha, beta, budget))
                alpha = max(alpha, value)
                if alpha >= beta or budget.expired():
                    break
            return value

        value = math.inf
        for action in actions[: self.max_actions]:
            value = min(value, self._value(state.apply(action), root_player, depth - 1, alpha, beta, budget))
            beta = min(beta, value)
            if alpha >= beta or budget.expired():
                break
        return value


# ---------------------------------------------------------------------------
# Minimax with transposition table + iterative deepening + move ordering.
# ---------------------------------------------------------------------------


def _state_key(state: Any) -> Any:
    """Best-effort hashable key for a state.

    Adapters that aren't naturally hashable should expose a ``key()`` method
    returning a hashable value (e.g. ``tuple(self.board)``). Falls back to
    ``repr`` which is correct but slower.
    """
    fn = getattr(state, "key", None)
    if callable(fn):
        return fn()
    try:
        hash(state)
        return state
    except TypeError:
        return repr(state)


# Transposition entry: (value, depth, flag) where flag in {"exact", "lower", "upper"}.
_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2


@dataclass
class MinimaxBotTT(Generic[Player, Action]):
    """Alpha-beta minimax with iterative deepening, transposition table,
    and PV-move ordering.

    Substantially faster than ``MinimaxBot`` on games with repeating positions
    (grid games, board games, anything with reachable cycles) because:
    - Transposition table reuses values for already-searched (state, depth).
    - Iterative deepening + previous-iteration PV move tried first → most α-β
      cutoffs happen near the root.
    - Returns the best move from the deepest *completed* depth, so partial
      searches at the next ply never corrupt the move choice.

    Adapters benefit from exposing ``state.key()`` returning a hashable value.
    """

    max_depth: int = 32
    evaluator: Evaluator[Player, Action] = default_evaluator
    use_id: bool = True
    name: str = "minimax-tt"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        players = require_players(state)
        if len(players) != 2:
            return GreedyBot(self.evaluator).choose_action(state, player, budget, rng)

        legal = list(state.legal_actions(player))
        if not legal:
            return random_legal_action(state, player, rng)

        tt: Dict[Tuple[Any, int], Tuple[float, int]] = {}
        # PV table: best move per state from the previous iteration.
        pv: Dict[Any, Action] = {}
        best_action: Action = legal[0]

        depth_range = range(1, self.max_depth + 1) if self.use_id else (self.max_depth,)
        for depth in depth_range:
            if budget.expired():
                break
            try:
                completed_action = self._root_search(
                    state, player, depth, budget, tt, pv, rng
                )
            except _BudgetExhausted:
                break
            if completed_action is not None:
                best_action = completed_action
        return best_action

    def _root_search(
        self,
        state: GameState[Player, Action],
        root_player: Player,
        depth: int,
        budget: TimeBudget,
        tt: Dict[Tuple[Any, int, Any], Tuple[float, int]],
        pv: Dict[Any, Action],
        rng: random.Random,
    ) -> Optional[Action]:
        actions = list(state.legal_actions(root_player))
        key = _state_key(state)
        prev = pv.get(key)
        if prev is not None and prev in actions:
            actions.remove(prev)
            actions.insert(0, prev)
        else:
            rng.shuffle(actions)

        best_value = -math.inf
        best_action: Optional[Action] = None
        alpha = -math.inf
        beta = math.inf
        for action in actions:
            if budget.expired():
                raise _BudgetExhausted()
            value = self._search(
                state.apply(action),
                root_player,
                depth - 1,
                alpha,
                beta,
                budget,
                tt,
                pv,
            )
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        if best_action is not None:
            pv[key] = best_action
        return best_action

    def _search(
        self,
        state: GameState[Player, Action],
        root_player: Player,
        depth: int,
        alpha: float,
        beta: float,
        budget: TimeBudget,
        tt: Dict[Tuple[Any, int, Any], Tuple[float, int]],
        pv: Dict[Any, Action],
    ) -> float:
        if budget.expired():
            raise _BudgetExhausted()

        if state.is_terminal() or depth <= 0:
            return self.evaluator(state, root_player)

        # TT key includes root_player so values cached for opposite roots
        # (e.g. when the same position is evaluated from white's turn vs
        # black's turn) don't collide.
        tt_key = (_state_key(state), depth, root_player)
        cached = tt.get(tt_key)
        if cached is not None:
            value, flag = cached
            if flag == _TT_EXACT:
                return value
            if flag == _TT_LOWER and value >= beta:
                return value
            if flag == _TT_UPPER and value <= alpha:
                return value

        actions = list(state.legal_actions(state.current_player))
        if not actions:
            return self.evaluator(state, root_player)

        s_key = _state_key(state)
        prev = pv.get(s_key)
        if prev is not None and prev in actions:
            actions.remove(prev)
            actions.insert(0, prev)

        original_alpha, original_beta = alpha, beta
        is_max = state.current_player == root_player
        best_action: Optional[Action] = None

        if is_max:
            value = -math.inf
            for action in actions:
                v = self._search(
                    state.apply(action), root_player, depth - 1,
                    alpha, beta, budget, tt, pv,
                )
                if v > value:
                    value = v
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for action in actions:
                v = self._search(
                    state.apply(action), root_player, depth - 1,
                    alpha, beta, budget, tt, pv,
                )
                if v < value:
                    value = v
                    best_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break

        if best_action is not None:
            pv[s_key] = best_action
        if value <= original_alpha:
            tt[tt_key] = (value, _TT_UPPER)
        elif value >= original_beta:
            tt[tt_key] = (value, _TT_LOWER)
        else:
            tt[tt_key] = (value, _TT_EXACT)
        return value


class _BudgetExhausted(Exception):
    """Raised inside the search when the time budget runs out."""


@dataclass
class BeamSearchBot(Generic[Player, Action]):
    width: int = 20
    depth: int = 4
    evaluator: Evaluator[Player, Action] = default_evaluator
    name: str = "beam"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        starts = [(state.apply(action), action) for action in state.legal_actions(player)]
        if not starts:
            return random_legal_action(state, player, rng)

        beam = starts
        for _ in range(self.depth - 1):
            if budget.expired():
                break
            expanded: List[tuple[GameState[Player, Action], Action]] = []
            for candidate_state, first_action in beam:
                if candidate_state.is_terminal():
                    expanded.append((candidate_state, first_action))
                    continue
                for action in candidate_state.legal_actions(candidate_state.current_player):
                    expanded.append((candidate_state.apply(action), first_action))
                    if budget.expired():
                        break
            if not expanded:
                break
            expanded.sort(key=lambda item: self.evaluator(item[0], player), reverse=True)
            beam = expanded[: self.width]

        beam.sort(key=lambda item: self.evaluator(item[0], player), reverse=True)
        return beam[0][1]


@dataclass
class _MctsNode(Generic[Player, Action]):
    state: GameState[Player, Action]
    parent: Optional["_MctsNode[Player, Action]"] = None
    # ``mover`` is the player who just played ``action`` to reach this state.
    # We score this node from ``mover``'s perspective; that fixes the bug
    # where opponent nodes were maximised for the root player.
    mover: Optional[Player] = None
    action: Optional[Action] = None
    untried: List[Action] = field(default_factory=list)
    children: List["_MctsNode[Player, Action]"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # cumulative reward from ``mover``'s perspective.


@dataclass
class MCTSBot(Generic[Player, Action]):
    simulations: int = 500
    rollout_depth: int = 80
    exploration: float = 1.4
    evaluator: Evaluator[Player, Action] = default_evaluator
    name: str = "mcts"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        root = _MctsNode(
            state=state,
            mover=None,  # nothing was played to reach the root
            untried=list(state.legal_actions(player)),
        )
        if not root.untried:
            return random_legal_action(state, player, rng)

        for _ in range(self.simulations):
            if budget.expired():
                break
            leaf, leaf_mover_player = self._select(root, rng)
            scores = self._rollout(leaf.state, rng, budget)
            self._backpropagate(leaf, scores)

        if not root.children:
            return random_legal_action(state, player, rng)
        return max(root.children, key=lambda child: child.visits).action  # type: ignore[return-value]

    def _select(
        self,
        node: _MctsNode[Player, Action],
        rng: random.Random,
    ) -> tuple[_MctsNode[Player, Action], Optional[Player]]:
        while not node.state.is_terminal():
            if node.untried:
                # The player about to move at ``node.state`` is the one whose
                # action we will play, so the resulting child's ``mover`` is
                # that player.
                mover = node.state.current_player
                action = node.untried.pop(rng.randrange(len(node.untried)))
                child_state = node.state.apply(action)
                child = _MctsNode(
                    state=child_state,
                    parent=node,
                    mover=mover,
                    action=action,
                    untried=list(child_state.legal_actions(child_state.current_player))
                    if not child_state.is_terminal()
                    else [],
                )
                node.children.append(child)
                return child, mover
            if not node.children:
                return node, node.mover
            node = max(node.children, key=self._ucb)
        return node, node.mover

    def _ucb(self, node: _MctsNode[Player, Action]) -> float:
        if node.visits == 0:
            return math.inf
        parent_visits = max(1, node.parent.visits if node.parent else 1)
        # ``node.value`` is already from ``node.mover``'s perspective, which is
        # the player who chose this branch from the parent — exactly the
        # quantity the parent wants to maximise.
        return (node.value / node.visits) + self.exploration * math.sqrt(
            math.log(parent_visits) / node.visits
        )

    def _rollout(
        self,
        state: GameState[Player, Action],
        rng: random.Random,
        budget: TimeBudget,
    ) -> dict:
        depth = 0
        while not state.is_terminal() and depth < self.rollout_depth and not budget.expired():
            actions = list(state.legal_actions(state.current_player))
            if not actions:
                break
            state = state.apply(rng.choice(actions))
            depth += 1
        # Evaluate from every known player's perspective so we can backprop
        # the right number into nodes whose ``mover`` differs.
        players = getattr(state, "players", None)
        if players is None:
            # Fall back to a single-perspective evaluator (will be applied to
            # every node).  Most adapters expose ``players``; this branch is
            # only a safety net.
            return {"__default__": self.evaluator(state, state.current_player)}
        return {p: self.evaluator(state, p) for p in players}

    def _backpropagate(self, node: _MctsNode[Player, Action], scores: dict) -> None:
        default = scores.get("__default__", 0.0)
        while node is not None:
            node.visits += 1
            if node.mover is None:
                # Root: aggregate isn't used for selection; keep average reward
                # across players to make ``visits`` count consistent.
                node.value += sum(scores.values()) / max(1, len(scores)) if scores else default
            else:
                node.value += scores.get(node.mover, default)
            node = node.parent  # type: ignore[assignment]


@dataclass
class MetaBot(Generic[Player, Action]):
    bots: Sequence[object]
    name: str = "meta"

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: TimeBudget,
        rng: random.Random,
    ) -> Action:
        # Try strongest strategies first. Give the first bot the most time
        # (1/N of remaining), the next 1/(N-1) of what's left, and so on, so
        # later fallbacks still get something even if the early ones are slow.
        n = len(self.bots)
        for idx, bot in enumerate(self.bots):
            try:
                share = 1.0 / max(1, n - idx)
                child_budget = budget.fraction(share)
                action = bot.choose_action(state, player, child_budget, rng)  # type: ignore[attr-defined]
                if action in state.legal_actions(player):
                    return action
            except Exception:
                continue
        return random_legal_action(state, player, rng)
