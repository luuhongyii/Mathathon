from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Protocol, Sequence, TypeVar


Player = TypeVar("Player")
Action = TypeVar("Action")


class GameState(Protocol[Player, Action]):
    """Sequential, turn-based game adapter.

    For simultaneous-move games implement ``SimultaneousState`` instead.
    Most adapters should also expose a class-level ``players`` tuple so
    bots that need it can enumerate all seats.
    """

    @property
    def current_player(self) -> Player:
        """Return the player whose turn it is."""

    def legal_actions(self, player: Optional[Player] = None) -> Sequence[Action]:
        """Return actions available to the requested player or current player."""

    def apply(self, action: Action) -> "GameState[Player, Action]":
        """Return the next immutable state after applying an action."""

    def is_terminal(self) -> bool:
        """Return True when the game is over."""

    def score(self, player: Player) -> float:
        """Return the final or heuristic score from a player's perspective."""


class SimultaneousState(Protocol[Player, Action]):
    """Game adapter where all live players move at the same time.

    Each round every player produces one action and the engine resolves
    them with ``apply_joint``. Use this for matrix games, Blotto, sealed
    bid auctions, beauty contest, iterated PD, etc.
    """

    players: Sequence[Player]

    def active_players(self) -> Sequence[Player]:
        """Players who must submit an action this round."""

    def legal_actions(self, player: Player) -> Sequence[Action]:
        """Actions available to ``player`` this round."""

    def apply_joint(self, actions: Mapping[Player, Action]) -> "SimultaneousState[Player, Action]":
        """Return the next state given everyone's action this round."""

    def is_terminal(self) -> bool: ...

    def score(self, player: Player) -> float: ...


class Bot(Protocol[Player, Action]):
    name: str

    def choose_action(
        self,
        state: GameState[Player, Action],
        player: Player,
        budget: "TimeBudget",
        rng: random.Random,
    ) -> Action:
        """Choose one legal action before the time budget expires."""


@dataclass
class TimeBudget:
    seconds: float
    safety_margin: float = 0.01
    _start: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    def expired(self) -> bool:
        return self.remaining <= self.safety_margin

    def fraction(self, part: float) -> "TimeBudget":
        return TimeBudget(max(0.001, self.remaining * part), self.safety_margin)


@dataclass
class StepRecord(Generic[Player, Action]):
    turn: int
    player: Player
    action: Action
    elapsed: float
    note: str = ""


@dataclass
class GameResult(Generic[Player, Action]):
    scores: Dict[Player, float]
    winners: List[Player]
    turns: int
    history: List[StepRecord[Player, Action]]
    errors: List[str]


class Simulator(Generic[Player, Action]):
    """Turn-based simulator. Supports any number of players (>=2)."""

    def __init__(
        self,
        players: Sequence[Player],
        max_turns: int = 1_000,
        time_limit_per_move: float = 0.2,
    ) -> None:
        self.players = list(players)
        self.max_turns = max_turns
        self.time_limit_per_move = time_limit_per_move

    def play(
        self,
        initial_state: GameState[Player, Action],
        bots: Dict[Player, Bot[Player, Action]],
        seed: Optional[int] = None,
        record: bool = False,
    ) -> GameResult[Player, Action]:
        rng = random.Random(seed)
        state = initial_state
        history: List[StepRecord[Player, Action]] = []
        errors: List[str] = []

        for turn in range(self.max_turns):
            if state.is_terminal():
                break

            player = state.current_player
            legal = list(state.legal_actions(player))
            if not legal:
                errors.append(f"turn {turn}: player {player!r} had no legal actions")
                break

            bot = bots[player]
            budget = TimeBudget(self.time_limit_per_move)
            start = time.perf_counter()
            note = ""

            try:
                action = bot.choose_action(state, player, budget, rng)
            except Exception as exc:  # Local runner should expose crashes without stopping a tournament.
                action = rng.choice(legal)
                note = f"fallback after exception: {exc!r}"
                errors.append(f"turn {turn}: {getattr(bot, 'name', bot)!r} crashed: {exc!r}")

            if action not in legal:
                errors.append(
                    f"turn {turn}: {getattr(bot, 'name', bot)!r} returned illegal action {action!r}"
                )
                action = rng.choice(legal)
                note = "fallback after illegal action"

            elapsed = time.perf_counter() - start
            if elapsed > self.time_limit_per_move:
                errors.append(
                    f"turn {turn}: {getattr(bot, 'name', bot)!r} exceeded time limit "
                    f"({elapsed:.4f}s > {self.time_limit_per_move:.4f}s)"
                )

            if record:
                history.append(StepRecord(turn, player, action, elapsed, note))

            state = state.apply(action)

        scores = {player: state.score(player) for player in self.players}
        best = max(scores.values()) if scores else 0.0
        winners = [player for player, score in scores.items() if score == best]
        return GameResult(scores, winners, len(history), history, errors)


class SimultaneousSimulator(Generic[Player, Action]):
    """Drives a SimultaneousState. Every active player picks an action per round."""

    def __init__(
        self,
        players: Sequence[Player],
        max_rounds: int = 1_000,
        time_limit_per_move: float = 0.2,
    ) -> None:
        self.players = list(players)
        self.max_rounds = max_rounds
        self.time_limit_per_move = time_limit_per_move

    def play(
        self,
        initial_state: SimultaneousState[Player, Action],
        bots: Dict[Player, Bot[Player, Action]],
        seed: Optional[int] = None,
        record: bool = False,
    ) -> GameResult[Player, Action]:
        rng = random.Random(seed)
        state = initial_state
        history: List[StepRecord[Player, Action]] = []
        errors: List[str] = []

        for rnd in range(self.max_rounds):
            if state.is_terminal():
                break

            actives = list(state.active_players())
            if not actives:
                break

            joint: Dict[Player, Action] = {}
            for player in actives:
                legal = list(state.legal_actions(player))
                if not legal:
                    errors.append(f"round {rnd}: player {player!r} had no legal actions")
                    continue

                bot = bots[player]
                budget = TimeBudget(self.time_limit_per_move)
                start = time.perf_counter()
                note = ""

                try:
                    # ``state`` here exposes ``legal_actions(player)`` so it
                    # can satisfy the sequential ``GameState`` protocol used by
                    # bots that don't know they're playing simultaneously.
                    action = bot.choose_action(state, player, budget, rng)  # type: ignore[arg-type]
                except Exception as exc:
                    action = rng.choice(legal)
                    note = f"fallback after exception: {exc!r}"
                    errors.append(
                        f"round {rnd}: {getattr(bot, 'name', bot)!r} crashed: {exc!r}"
                    )

                if action not in legal:
                    errors.append(
                        f"round {rnd}: {getattr(bot, 'name', bot)!r} returned illegal action {action!r}"
                    )
                    action = rng.choice(legal)
                    note = "fallback after illegal action"

                elapsed = time.perf_counter() - start
                if elapsed > self.time_limit_per_move:
                    errors.append(
                        f"round {rnd}: {getattr(bot, 'name', bot)!r} exceeded time limit "
                        f"({elapsed:.4f}s > {self.time_limit_per_move:.4f}s)"
                    )

                joint[player] = action
                if record:
                    history.append(StepRecord(rnd, player, action, elapsed, note))

            if not joint:
                break
            state = state.apply_joint(joint)

        scores = {player: state.score(player) for player in self.players}
        best = max(scores.values()) if scores else 0.0
        winners = [player for player, score in scores.items() if score == best]
        return GameResult(scores, winners, len(history), history, errors)


def require_players(state: GameState[Player, Action]) -> List[Player]:
    players = getattr(state, "players", None)
    if players is None:
        raise ValueError("state must expose a 'players' sequence for this algorithm")
    return list(players)


def random_legal_action(
    state: GameState[Player, Action],
    player: Player,
    rng: random.Random,
) -> Action:
    legal = list(state.legal_actions(player))
    if not legal:
        raise ValueError(f"player {player!r} has no legal actions")
    return rng.choice(legal)


def iter_limited(items: Iterable[Action], limit: Optional[int]) -> Iterable[Action]:
    for idx, item in enumerate(items):
        if limit is not None and idx >= limit:
            return
        yield item
