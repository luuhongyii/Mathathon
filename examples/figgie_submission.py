"""stdio submission for simplified Figgie — the rehearsal's "ship it" step.

The judge sends one JSON line per turn containing ONLY this player's
observation (its own hand + chips + round), and expects one action token line
back: ``buy <suit>``, ``sell <suit>`` or ``pass``.

Local test (no judge):
    echo '{"hand": [3, 2, 4, 1], "chips": 100, "round": 0}' | python examples/figgie_submission.py

Bundle into one self-contained file for a single-file judge:
    python tools/bundle.py examples/figgie_submission.py -o submission.py
    echo '{"hand": [3, 2, 4, 1], "chips": 100, "round": 0}' | python submission.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.figgie import goal_belief
from mathathon_kit import (
    IOConfig,
    json_state_parser,
    run_per_move_loop,
    tokens_action_formatter,
)


def choose(obs, budget, rng):
    """obs = {"hand": [c0, c1, c2, c3], "chips": int, "round": int}.

    Reply with ``bid <suit> <price>``, ``ask <suit> <price>`` or ``pass``.
    Inference policy: estimate the hidden goal suit from our own hand and bid
    for it, as high up the price ladder as our confidence justifies.
    """
    hand = obs["hand"]
    belief = goal_belief(hand)
    goal = max(range(4), key=lambda s: belief[s])
    confidence = belief[goal]
    ceiling = 14 if confidence > 0.45 else 12 if confidence > 0.30 else 10 if confidence > 0.22 else 8
    for price in (14, 12, 10, 8, 6):
        if price <= ceiling and obs["chips"] >= price:
            return ["bid", goal, price]
    return ["pass"]


def safe_fallback(obs, budget, rng):
    """`pass` is always legal — so a crash in choose() never forfeits a turn."""
    return ["pass"]


if __name__ == "__main__":
    cfg = IOConfig(
        parse_state=json_state_parser,
        format_action=tokens_action_formatter,
        time_limit_per_move=0.1,
        fallback=safe_fallback,
        seed=1,
    )
    run_per_move_loop(choose, cfg)
