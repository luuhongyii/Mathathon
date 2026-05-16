"""Platform-style submission for Nim using the stdio harness.

Drop into the platform IDE as your bot file. The judge sends one line per
turn containing the current pile size; we reply with a single integer (1..3).

Local test (no judge):
    echo 21 | python examples/platform_submission_nim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.nim_game import NimState
from mathathon_kit import (
    IOConfig,
    IterativeDeepeningMinimax,
    make_choose_from_bot,
    run_per_move_loop,
    tokens_action_formatter,
    tokens_state_parser,
)


def parse_state(tokens):
    pile = int(tokens[0])
    current_index = int(tokens[1]) if len(tokens) > 1 else 0
    return NimState(pile=pile, current_index=current_index)


bot = IterativeDeepeningMinimax(max_depth=24)
choose = make_choose_from_bot(
    bot,
    state_to_game_state=parse_state,
)

if __name__ == "__main__":
    cfg = IOConfig(
        parse_state=tokens_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.2,
        seed=1,
    )
    run_per_move_loop(choose, cfg)
