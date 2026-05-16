"""Pre-flight checks -- run these the moment you finish a state adapter.

A round-robin that prints clean numbers can still be built on a broken
adapter: ``apply`` that mutates in place, an unhashable state that silently
disables the transposition table, ``legal_actions`` that lists an illegal
move, a game that never terminates. Those bugs cost games, not crashes, so
they are easy to miss. ``validate_adapter`` plays random games and flags them
directly; ``benchmark_bot`` measures decision time so you catch timeouts
before the judge does.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List

from .core import TimeBudget


# ---------------------------------------------------------------------------
# Adapter validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    states_checked: int = 0

    @property
    def ok(self) -> bool:
        """True when no errors were found (warnings are allowed)."""
        return not self.errors

    def summary(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"adapter validation: {head}  ({self.states_checked} states checked)"]
        for msg in self.errors:
            lines.append(f"  ERROR    {msg}")
        for msg in self.warnings:
            lines.append(f"  WARNING  {msg}")
        for msg in self.notes:
            lines.append(f"  note     {msg}")
        if self.ok and not self.warnings:
            lines.append("  (no issues)")
        return "\n".join(lines)


def _add(bucket: List[str], msg: str) -> None:
    """Append ``msg`` once -- the same fault recurs across many states."""
    if msg not in bucket:
        bucket.append(msg)


def _looks_like_default_repr(state: Any) -> bool:
    text = repr(state)
    return text.startswith("<") and " object at 0x" in text


def validate_adapter(
    state: Any,
    *,
    games: int = 20,
    max_turns: int = 500,
    seed: int = 0,
) -> ValidationReport:
    """Battery of correctness checks for a freshly written state adapter.

    Pass an initial state; auto-detects turn-based (``GameState``) vs
    simultaneous (``SimultaneousState``) by the presence of ``apply_joint``.
    Plays ``games`` random playouts and checks every state visited.
    """
    report = ValidationReport()
    rng = random.Random(seed)
    if _looks_like_default_repr(state):
        report.notes.append(
            "state has no custom __repr__; mutation/determinism checks are "
            "weaker -- make the state a @dataclass"
        )
    if hasattr(state, "apply_joint"):
        _validate_simultaneous(state, report, rng, games, max_turns)
    else:
        _validate_turn_based(state, report, rng, games, max_turns)
    return report


def _check_common(state: Any, players: Any, report: ValidationReport) -> None:
    """Checks that apply to both adapter kinds: hashability and score()."""
    try:
        hash(state)
    except TypeError:
        _add(
            report.warnings,
            "state is not hashable -> MinimaxBotTT / ISMCTS caching is "
            "disabled (use a frozen dataclass with tuple, not list, fields)",
        )
    if players is None:
        return
    for player in players:
        try:
            value = state.score(player)
        except Exception as exc:  # noqa: BLE001
            _add(report.errors, f"score({player!r}) raised {exc!r}")
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                _add(report.errors, f"score({player!r}) returned non-number {value!r}")


def _validate_turn_based(
    initial: Any, report: ValidationReport, rng: random.Random, games: int, max_turns: int
) -> None:
    players = getattr(initial, "players", None)
    if players is None:
        _add(
            report.warnings,
            "state has no `players` attribute; tournaments and some bots need it",
        )

    for _ in range(games):
        state = initial
        for _turn in range(max_turns):
            report.states_checked += 1
            terminal = bool(state.is_terminal())

            try:
                current = state.current_player
            except Exception as exc:  # noqa: BLE001
                _add(report.errors, f"current_player raised {exc!r}")
                break
            if players is not None and current not in players:
                _add(
                    report.errors,
                    f"current_player {current!r} is not in players {tuple(players)!r}",
                )

            _check_common(state, players, report)

            legal = list(state.legal_actions(current))
            if terminal:
                if legal:
                    _add(
                        report.warnings,
                        "terminal state still lists legal actions "
                        "(search treats a terminal node as a leaf)",
                    )
                break
            if not legal:
                _add(
                    report.errors,
                    "non-terminal state has no legal actions (the game can deadlock)",
                )
                break

            before = repr(state)
            for action in legal:
                try:
                    state.apply(action)
                except Exception as exc:  # noqa: BLE001
                    _add(report.errors, f"apply({action!r}) raised {exc!r}")
                    continue
                if repr(state) != before:
                    _add(
                        report.errors,
                        f"apply({action!r}) mutated the original state "
                        "(apply must return a NEW state, never mutate self)",
                    )

            choice = rng.choice(legal)
            if repr(state.apply(choice)) != repr(state.apply(choice)):
                _add(
                    report.errors,
                    "apply() is non-deterministic (same action -> different state)",
                )
            state = state.apply(choice)
        else:
            _add(
                report.errors,
                f"a random playout did not terminate within {max_turns} turns "
                "-- minimax / MCTS rollouts may hang or be truncated",
            )


def _validate_simultaneous(
    initial: Any, report: ValidationReport, rng: random.Random, games: int, max_turns: int
) -> None:
    players = getattr(initial, "players", None)
    if players is None:
        _add(report.errors, "simultaneous state must expose a `players` sequence")
        return

    for _ in range(games):
        state = initial
        for _rnd in range(max_turns):
            report.states_checked += 1
            terminal = bool(state.is_terminal())
            _check_common(state, players, report)
            if terminal:
                break

            actives = list(state.active_players())
            if not actives:
                _add(report.warnings, "non-terminal state has no active players")
                break

            joint: dict = {}
            for player in actives:
                legal = list(state.legal_actions(player))
                if not legal:
                    _add(report.errors, f"active player {player!r} has no legal actions")
                    joint = {}
                    break
                joint[player] = rng.choice(legal)
            if not joint:
                break

            before = repr(state)
            try:
                nxt = state.apply_joint(joint)
            except Exception as exc:  # noqa: BLE001
                _add(report.errors, f"apply_joint raised {exc!r}")
                break
            if repr(state) != before:
                _add(
                    report.errors,
                    "apply_joint mutated the original state (it must return a NEW state)",
                )
            state = nxt
        else:
            _add(
                report.errors,
                f"a random playout did not terminate within {max_turns} rounds",
            )


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    samples: int
    mean: float
    p95: float
    worst: float
    overruns: int
    time_limit: float

    @property
    def ok(self) -> bool:
        """True when no sampled move exceeded the time limit."""
        return self.overruns == 0

    def summary(self) -> str:
        head = "OK" if self.ok else f"{self.overruns} OVERRUN(S)"
        return (
            f"latency [{head}]  limit={self.time_limit:.3f}s  "
            f"mean={self.mean:.4f}s  p95={self.p95:.4f}s  "
            f"worst={self.worst:.4f}s  (n={self.samples})"
        )


def benchmark_bot(
    bot: Any,
    state_factory: Callable[[int], Any],
    *,
    positions: int = 40,
    time_limit: float = 0.2,
    seed: int = 0,
) -> BenchmarkResult:
    """Measure a bot's per-move decision time -- turn-based or simultaneous.

    Plays random games from ``state_factory(seed)`` and times every
    ``choose_action`` call. A single move over ``time_limit`` usually means a
    forfeited game on the judge, so check ``result.ok`` before submitting.
    For a simultaneous game the bot is timed in the first active seat each
    round; the game is then advanced with random joint actions.
    """
    rng = random.Random(seed)
    times: List[float] = []
    game = 0
    while len(times) < positions:
        state = state_factory(seed + game)
        game += 1
        simultaneous = hasattr(state, "apply_joint")
        for _ in range(10_000):
            if state.is_terminal() or len(times) >= positions:
                break

            if simultaneous:
                actives = list(state.active_players())
                if not actives:
                    break
                seat = actives[0]
            else:
                actives = None
                seat = state.current_player

            legal = list(state.legal_actions(seat))
            if not legal:
                break

            budget = TimeBudget(time_limit)
            start = time.perf_counter()
            try:
                bot.choose_action(state, seat, budget, rng)
            except Exception:  # noqa: BLE001 -- timing a crash still informs us
                pass
            times.append(time.perf_counter() - start)

            # Advance with random play so we sample varied positions.
            if simultaneous:
                joint = {}
                stuck = False
                for player in actives:  # type: ignore[union-attr]
                    options = list(state.legal_actions(player))
                    if not options:
                        stuck = True
                        break
                    joint[player] = rng.choice(options)
                if stuck:
                    break
                state = state.apply_joint(joint)
            else:
                state = state.apply(rng.choice(legal))
        if game > positions + 50:  # guard against a game that never yields moves
            break

    times.sort()
    n = len(times)
    if n == 0:
        return BenchmarkResult(0, 0.0, 0.0, 0.0, 0, time_limit)
    mean = sum(times) / n
    p95 = times[min(n - 1, int(0.95 * n))]
    worst = times[-1]
    overruns = sum(1 for t in times if t > time_limit)
    return BenchmarkResult(n, mean, p95, worst, overruns, time_limit)
