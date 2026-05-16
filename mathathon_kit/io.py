"""Platform I/O harness.

Most online judging platforms talk to your bot via stdin/stdout in one of
three patterns. This module gives you ready-to-use loops for each so you can
plug in a `choose` function and submit immediately:

1. ``run_per_move_loop``: each turn the judge writes the current state, your
   bot writes one action, repeat until EOF.
2. ``run_one_shot``: the whole game state is dumped once, you write a full
   plan/action and exit.
3. ``run_simultaneous_loop``: both players receive the public state plus the
   last round's revealed actions, you output your next action.

The helpers are protocol-agnostic: you pass parser/formatter callbacks. We
ship JSON-line and whitespace-token presets because those cover ~95% of game
platforms.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, TextIO

from .core import TimeBudget


StateParser = Callable[[str], Any]
ActionFormatter = Callable[[Any], str]
ChooseFn = Callable[[Any, TimeBudget, random.Random], Any]


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


def json_state_parser(line: str) -> Any:
    return json.loads(line)


def json_action_formatter(action: Any) -> str:
    return json.dumps(action, ensure_ascii=False, default=str)


def tokens_state_parser(line: str) -> list[str]:
    return line.strip().split()


def tokens_action_formatter(action: Any) -> str:
    if isinstance(action, (list, tuple)):
        return " ".join(str(x) for x in action)
    return str(action)


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


@dataclass
class IOConfig:
    parse_state: StateParser = json_state_parser
    format_action: ActionFormatter = json_action_formatter
    time_limit_per_move: float = 0.2
    seed: Optional[int] = None
    flush_after_each_move: bool = True
    skip_empty_lines: bool = True
    # Crash safety: a single uncaught exception in parse/choose forfeits the
    # whole game. The loops below catch it, log to stderr, and -- if you supply
    # a ``fallback`` -- emit a safe action instead. ALWAYS pass a fallback for a
    # real submission (e.g. ``lambda s, b, r: r.choice(legal_actions_of(s))``).
    fallback: Optional[ChooseFn] = None
    log_errors: bool = True


def _log(msg: str) -> None:
    """Write a diagnostic to stderr. stdout is the judge protocol channel, so
    debug output must never go there."""
    sys.stderr.write(f"[mathathon_kit.io] {msg}\n")
    sys.stderr.flush()


def _guarded_choose(
    choose: ChooseFn,
    state: Any,
    budget: TimeBudget,
    rng: random.Random,
    cfg: Any,
) -> Any:
    """Run ``choose`` but never propagate an exception out of the I/O loop.

    Returns a sentinel ``_NO_ACTION`` when neither ``choose`` nor the fallback
    produced an action, so the caller can skip emitting for this turn rather
    than crash the whole process.
    """
    try:
        return choose(state, budget, rng)
    except Exception as exc:  # noqa: BLE001 -- a submission must not die here
        if cfg.log_errors:
            _log(f"choose() raised {exc!r}")
        if cfg.fallback is None:
            return _NO_ACTION
        try:
            return cfg.fallback(state, budget, rng)
        except Exception as exc2:  # noqa: BLE001
            if cfg.log_errors:
                _log(f"fallback also raised {exc2!r}")
            return _NO_ACTION


_NO_ACTION = object()


def run_per_move_loop(
    choose: ChooseFn,
    config: Optional[IOConfig] = None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Read one state per line, write one action per line, until EOF."""

    cfg = config or IOConfig()
    rng = random.Random(cfg.seed)
    src = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    for raw in src:
        if cfg.skip_empty_lines and not raw.strip():
            continue
        try:
            state = cfg.parse_state(raw)
        except Exception as exc:  # noqa: BLE001 -- skip the bad line, stay alive
            if cfg.log_errors:
                _log(f"parse_state failed on {raw!r}: {exc!r}")
            continue
        budget = TimeBudget(cfg.time_limit_per_move)
        action = _guarded_choose(choose, state, budget, rng, cfg)
        if action is _NO_ACTION:
            continue
        sink.write(cfg.format_action(action) + "\n")
        if cfg.flush_after_each_move:
            sink.flush()


def run_one_shot(
    choose: ChooseFn,
    config: Optional[IOConfig] = None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Read the entire stdin as one blob, return one action."""

    cfg = config or IOConfig()
    rng = random.Random(cfg.seed)
    src = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    payload = src.read()
    try:
        state = cfg.parse_state(payload)
    except Exception as exc:  # noqa: BLE001
        if cfg.log_errors:
            _log(f"parse_state failed: {exc!r}")
        return
    budget = TimeBudget(cfg.time_limit_per_move)
    action = _guarded_choose(choose, state, budget, rng, cfg)
    if action is _NO_ACTION:
        return
    sink.write(cfg.format_action(action) + "\n")
    sink.flush()


def run_simultaneous_loop(
    choose: ChooseFn,
    config: Optional[IOConfig] = None,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Same as ``run_per_move_loop`` semantically.

    Kept as a separate name so the bot file's intent is clear when you know
    you're playing a simultaneous-move game (the wire protocol is identical:
    judge sends public state, you send action, repeat).
    """

    run_per_move_loop(choose, config, stdin, stdout)


# ---------------------------------------------------------------------------
# Bot wrapper helpers
# ---------------------------------------------------------------------------


def make_choose_from_bot(
    bot,
    state_to_game_state: Callable[[Any], Any],
    player_from_state: Optional[Callable[[Any], Any]] = None,
) -> ChooseFn:
    """Wrap a `Bot` instance into a `choose(raw_state, budget, rng)` function.

    ``state_to_game_state`` converts the parsed wire state into a `GameState`
    your offline bots already understand.
    """

    def _choose(raw_state: Any, budget: TimeBudget, rng: random.Random) -> Any:
        gs = state_to_game_state(raw_state)
        player = player_from_state(raw_state) if player_from_state else gs.current_player
        return bot.choose_action(gs, player, budget, rng)

    return _choose


# ---------------------------------------------------------------------------
# Local pipe simulator (for testing your stdio bot without the platform)
# ---------------------------------------------------------------------------


def simulate_pipe(
    bot_main: Callable[[TextIO, TextIO], None],
    inputs: Iterable[str],
) -> Sequence[str]:
    """Run a stdio-style ``bot_main(stdin, stdout)`` against canned inputs."""

    import io as _io

    in_text = "\n".join(inputs) + "\n"
    in_buf = _io.StringIO(in_text)
    out_buf = _io.StringIO()
    bot_main(in_buf, out_buf)
    return out_buf.getvalue().splitlines()


# ---------------------------------------------------------------------------
# Flexible protocol harness
# ---------------------------------------------------------------------------
#
# ``run_per_move_loop`` assumes one line in -> one line out. Real judges often
# differ: a handshake/config block up front, multi-line turn states, sentinel
# lines between turns. ``run_protocol_loop`` keeps the same crash-safety but
# lets you plug in a ``read_state`` that knows your wire format. Build that
# reader from the helpers below instead of hand-rolling a loop under pressure.


#: Reads one turn's worth of input from the stream. Returns the raw text block
#: (which ``parse_state`` then parses), or ``None`` to signal end-of-game/EOF.
TurnReader = Callable[[TextIO], Optional[str]]


def read_line(stream: TextIO) -> Optional[str]:
    """Return the next non-empty line (newline stripped), or ``None`` at EOF."""
    for raw in stream:
        if raw.strip():
            return raw.rstrip("\n")
    return None


def read_n_lines(stream: TextIO, n: int) -> Optional[str]:
    """Read exactly ``n`` lines and return them joined. ``None`` at EOF."""
    lines: list[str] = []
    for raw in stream:
        lines.append(raw.rstrip("\n"))
        if len(lines) >= n:
            return "\n".join(lines)
    return "\n".join(lines) if lines else None


def read_until(
    stream: TextIO,
    sentinel: str,
    *,
    end_markers: Sequence[str] = (),
    include_sentinel: bool = False,
) -> Optional[str]:
    """Read lines until one (stripped) equals ``sentinel``; return the block.

    ``end_markers`` are lines that mean the game is over -> returns ``None``.
    Use this for ``...state lines...`` then a ``GO`` / ``END_TURN`` marker.
    """
    lines: list[str] = []
    for raw in stream:
        line = raw.rstrip("\n")
        stripped = line.strip()
        if stripped in end_markers:
            return None
        if stripped == sentinel:
            if include_sentinel:
                lines.append(line)
            return "\n".join(lines)
        lines.append(line)
    return "\n".join(lines) if lines else None


def count_prefixed_reader(count_token: int = 0, extra_lines: int = 0) -> TurnReader:
    """Build a reader for "header line states a row count, then that many rows".

    ``count_token`` is which whitespace token of the header holds the count
    (e.g. header ``"GRID 5 5"`` with the row count first -> token 1).
    ``extra_lines`` reads additional fixed lines after the body. The returned
    block includes the header line.
    """

    def _read(stream: TextIO) -> Optional[str]:
        header = read_line(stream)
        if header is None:
            return None
        try:
            n = int(header.split()[count_token])
        except (ValueError, IndexError):
            n = 0
        body: list[str] = []
        for raw in stream:
            body.append(raw.rstrip("\n"))
            if len(body) >= n + extra_lines:
                break
        return "\n".join([header] + body)

    return _read


@dataclass
class ProtocolConfig:
    """Config for ``run_protocol_loop`` -- the multi-line / handshake variant.

    ``read_state`` is mandatory: a ``TurnReader`` built from the helpers above.
    ``handshake`` runs once before the first turn (read the config block, write
    your bot name if required); stash anything ``choose`` needs in a closure.
    """

    read_state: TurnReader
    parse_state: StateParser = json_state_parser
    format_action: ActionFormatter = json_action_formatter
    handshake: Optional[Callable[[TextIO, TextIO], None]] = None
    time_limit_per_move: float = 0.2
    seed: Optional[int] = None
    flush_after_each_move: bool = True
    fallback: Optional[ChooseFn] = None
    log_errors: bool = True


def run_protocol_loop(
    choose: ChooseFn,
    config: ProtocolConfig,
    stdin: Optional[TextIO] = None,
    stdout: Optional[TextIO] = None,
) -> None:
    """Like ``run_per_move_loop`` but for multi-line / handshake protocols.

    Reads a turn via ``config.read_state``; a ``None`` ends the loop. Crash
    safety (fallback + stderr logging) is identical to ``run_per_move_loop``.
    """

    rng = random.Random(config.seed)
    src = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    if config.handshake is not None:
        try:
            config.handshake(src, sink)
        except Exception as exc:  # noqa: BLE001
            if config.log_errors:
                _log(f"handshake failed: {exc!r}")

    while True:
        try:
            raw = config.read_state(src)
        except Exception as exc:  # noqa: BLE001
            if config.log_errors:
                _log(f"read_state failed: {exc!r}")
            break
        if raw is None:
            break
        try:
            state = config.parse_state(raw)
        except Exception as exc:  # noqa: BLE001
            if config.log_errors:
                _log(f"parse_state failed on {raw!r}: {exc!r}")
            continue
        budget = TimeBudget(config.time_limit_per_move)
        action = _guarded_choose(choose, state, budget, rng, config)
        if action is _NO_ACTION:
            continue
        sink.write(config.format_action(action) + "\n")
        if config.flush_after_each_move:
            sink.flush()


# ---------------------------------------------------------------------------
# Crash-safe fallback factory
# ---------------------------------------------------------------------------


def random_legal_fallback(
    state_to_game_state: Callable[[Any], Any],
    player_from_state: Optional[Callable[[Any], Any]] = None,
) -> ChooseFn:
    """Build an ``IOConfig.fallback``: convert the wire state to a GameState and
    return a uniformly random *legal* action.

    Pass the SAME converter you handed ``make_choose_from_bot``. This closes
    the crash-safety loop: if the real engine raises, the bot still emits a
    legal move instead of forfeiting the game.
    """

    def _fallback(raw_state: Any, budget: TimeBudget, rng: random.Random) -> Any:
        gs = state_to_game_state(raw_state)
        player = player_from_state(raw_state) if player_from_state else gs.current_player
        actions = list(gs.legal_actions(player))
        if not actions:
            actions = list(gs.legal_actions())
        return rng.choice(actions)

    return _fallback


# ---------------------------------------------------------------------------
# Transcript capture & replay
# ---------------------------------------------------------------------------
#
# When the judge's wire format surprises you, record the real session and
# replay it offline instead of guessing from a stack trace.


class _TeeReader:
    """Stream wrapper that copies everything read into a log file."""

    def __init__(self, stream: TextIO, log_path: str) -> None:
        self._stream = stream
        self._log = open(log_path, "w", encoding="utf-8")

    def readline(self, *args) -> str:
        line = self._stream.readline(*args)
        if line:
            self._log.write(line)
            self._log.flush()
        return line

    def read(self, *args) -> str:
        data = self._stream.read(*args)
        if data:
            self._log.write(data)
            self._log.flush()
        return data

    def __iter__(self) -> "_TeeReader":
        return self

    def __next__(self) -> str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def close(self) -> None:
        self._log.close()


def record_stdin(log_path: str, stream: Optional[TextIO] = None) -> _TeeReader:
    """Wrap a stream so every line the judge sends is appended to ``log_path``.

    Drop-in: pass the result as the loop's ``stdin``. After the game, feed the
    log to ``replay_transcript`` to debug offline.
    """
    return _TeeReader(stream if stream is not None else sys.stdin, log_path)


def replay_transcript(
    bot_main: Callable[[TextIO, TextIO], None],
    stdin_log: str,
) -> Sequence[str]:
    """Re-run a stdio bot against a previously recorded stdin transcript."""
    import io as _io

    with open(stdin_log, "r", encoding="utf-8") as handle:
        text = handle.read()
    in_buf = _io.StringIO(text)
    out_buf = _io.StringIO()
    bot_main(in_buf, out_buf)
    return out_buf.getvalue().splitlines()
