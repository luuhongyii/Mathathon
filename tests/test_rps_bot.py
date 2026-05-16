"""Smoke test: the adaptive RPS bot should crush patterned opponents and
roughly tie a truly random one."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.rps_submission import RPSBot, beat, outcome


def simulate(opp_strategy, rounds=203, seed=0):
    """Play a match. ``opp_strategy(opp_hist, my_hist)`` decides simultaneously
    from history only (it cannot see the current-round move)."""
    bot = RPSBot(rng=random.Random(seed))
    opp_rng = random.Random(seed + 999)
    opp_hist, my_hist = [], []
    wins = losses = 0
    for k in range(rounds):
        if k > 0:
            bot.record_opponent(opp_hist[-1])
        my = bot.next_move()
        opp = opp_strategy(opp_hist, my_hist, opp_rng)
        r = outcome(my, opp)
        wins += r == 1
        losses += r == -1
        my_hist.append(my)
        opp_hist.append(opp)
    return wins, losses, rounds


def report(name, opp_strategy):
    w, l, n = simulate(opp_strategy)
    print(f"{name:18s} win={w:3d} loss={l:3d} tie={n-w-l:3d}  edge={w-l:+d}")
    return w, l


def test_rps_bot():
    # always Rock
    w, l = report("always-rock", lambda o, m, r: 0)
    assert w - l > 120, "should dominate a constant opponent"

    # fixed cycle R,P,S
    w, l = report("cycle-RPS", lambda o, m, r: len(o) % 3)
    assert w - l > 120, "should dominate a fixed cycle"

    # mirror: play whatever we played last round
    w, l = report("mirror-last", lambda o, m, r: m[-1] if m else 0)
    assert w - l > 60, "should beat a move-mirroring opponent"

    # counter: play the move that beats our last move
    w, l = report("counter-last", lambda o, m, r: beat(m[-1]) if m else 0)
    assert w - l > 60, "should beat a naive counter opponent"

    # biased random (70% Rock)
    def biased(o, m, r):
        return 0 if r.random() < 0.7 else r.randint(1, 2)
    w, l = report("biased-70%-rock", biased)
    assert w - l > 30, "should exploit a frequency-biased opponent"

    # truly random -> roughly a tie
    edges = [simulate(lambda o, m, r: r.randint(0, 2), seed=s) for s in range(5)]
    avg_edge = sum(w - l for w, l, _ in edges) / len(edges)
    print(f"{'uniform-random':18s} avg edge over 5 seeds = {avg_edge:+.1f}")
    assert abs(avg_edge) < 35, "should not lose badly to true random"


if __name__ == "__main__":
    test_rps_bot()
    print("OK")
