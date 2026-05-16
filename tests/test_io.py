import io as _io
import json
import random

from mathathon_kit import (
    IOConfig,
    ProtocolConfig,
    count_prefixed_reader,
    json_action_formatter,
    json_state_parser,
    random_legal_fallback,
    read_line,
    read_until,
    record_stdin,
    replay_transcript,
    run_per_move_loop,
    run_one_shot,
    run_protocol_loop,
    simulate_pipe,
    tokens_action_formatter,
    tokens_state_parser,
)


def _double_choose(state, budget, rng):
    return state["x"] * 2 if isinstance(state, dict) else int(state[0]) * 2


def test_json_per_move_loop():
    cfg = IOConfig(
        parse_state=json_state_parser,
        format_action=json_action_formatter,
        time_limit_per_move=0.05,
    )

    def main(stdin, stdout):
        run_per_move_loop(_double_choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, [json.dumps({"x": 3}), json.dumps({"x": 7})])
    assert out == ["6", "14"]


def test_tokens_per_move_loop():
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
    )

    def main(stdin, stdout):
        run_per_move_loop(_double_choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["3", "7"])
    assert out == ["6", "14"]


def test_one_shot_loop():
    cfg = IOConfig(
        parse_state=json_state_parser,
        format_action=json_action_formatter,
        time_limit_per_move=0.05,
    )

    in_buf = _io.StringIO(json.dumps({"x": 5}))
    out_buf = _io.StringIO()

    run_one_shot(_double_choose, cfg, stdin=in_buf, stdout=out_buf)
    assert out_buf.getvalue().strip() == "10"


def test_skips_empty_lines():
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
    )

    def main(stdin, stdout):
        run_per_move_loop(_double_choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["", "5", ""])
    assert out == ["10"]


def _boom(state, budget, rng):
    raise RuntimeError("bot crashed")


def test_loop_survives_choose_crash():
    """An uncaught exception in choose() must not kill the whole process."""
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
        log_errors=False,
    )

    def main(stdin, stdout):
        run_per_move_loop(_boom, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["3", "7"])
    assert out == []  # survived both turns, just emitted nothing


def test_loop_uses_fallback_on_crash():
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
        log_errors=False,
        fallback=lambda state, budget, rng: 0,
    )

    def main(stdin, stdout):
        run_per_move_loop(_boom, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["3", "7"])
    assert out == ["0", "0"]  # fallback kept the bot in the game


def test_loop_survives_parse_crash():
    def bad_parser(line):
        raise ValueError("malformed state")

    cfg = IOConfig(
        parse_state=bad_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
        log_errors=False,
    )

    def main(stdin, stdout):
        run_per_move_loop(_double_choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["3"])
    assert out == []


def test_protocol_loop_multiline_with_sentinel():
    """A turn is several lines terminated by GO; END ends the game."""

    def reader(stream):
        return read_until(stream, "GO", end_markers=("END",))

    cfg = ProtocolConfig(
        read_state=reader,
        parse_state=lambda block: sum(int(t) for t in block.split()),
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
    )

    def choose(state, budget, rng):
        return state

    def main(stdin, stdout):
        run_protocol_loop(choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["1 2", "3", "GO", "10", "GO", "END"])
    assert out == ["6", "10"]


def test_protocol_loop_handshake():
    """The handshake reads a config line and writes a reply before turn one."""
    captured = {}

    def handshake(stdin, stdout):
        captured["player"] = int(read_line(stdin).split()[1])
        stdout.write("READY\n")

    cfg = ProtocolConfig(
        read_state=read_line,
        parse_state=int,
        format_action=tokens_action_formatter,
        handshake=handshake,
        time_limit_per_move=0.05,
    )

    def choose(state, budget, rng):
        return state + captured["player"]

    def main(stdin, stdout):
        run_protocol_loop(choose, cfg, stdin=stdin, stdout=stdout)

    out = simulate_pipe(main, ["PLAYER 1", "5", "7"])
    assert out == ["READY", "6", "8"]


def test_count_prefixed_reader():
    reader = count_prefixed_reader(count_token=0)
    stream = _io.StringIO("2\nrow-a\nrow-b\n3\nx\ny\nz\n")
    assert reader(stream) == "2\nrow-a\nrow-b"
    assert reader(stream) == "3\nx\ny\nz"
    assert reader(stream) is None


def test_random_legal_fallback_returns_legal_move():
    from examples.nim_game import NimState

    fallback = random_legal_fallback(lambda raw: NimState(pile=int(raw[0])))
    action = fallback(["3"], None, random.Random(0))
    assert action in (1, 2, 3)


def test_record_and_replay_round_trip(tmp_path):
    log = str(tmp_path / "stdin.log")
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.05,
    )

    teed = record_stdin(log, stream=_io.StringIO("3\n7\n"))
    out_buf = _io.StringIO()
    run_per_move_loop(_double_choose, cfg, stdin=teed, stdout=out_buf)
    teed.close()
    assert out_buf.getvalue().splitlines() == ["6", "14"]

    def main(stdin, stdout):
        run_per_move_loop(_double_choose, cfg, stdin=stdin, stdout=stdout)

    assert replay_transcript(main, log) == ["6", "14"]
