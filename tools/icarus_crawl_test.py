"""Replay the rainbow_nash/Onions/Pieleda/ImmigrantsI 'crawl' match.

That game degenerated: all 4 bots spiralled bids down to 1 by round ~27 and
crawled at +1/round to a ~102-round cap with nobody near 999. We substitute
our bot into each seat (others fixed to the log) to see whether our descent
rate banks more or less position than the field.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
from icarus_rl_train import LearnedBot, Params, clamp_bid, TARGET  # noqa: E402
from icarus_replay import step  # noqa: E402

# Explicit bids rounds 0-32; rounds 33-102 are all-1 crawl.
HEAD = [
    [48, 87, 44, 44], [43, 42, 76, 43], [68, 37, 62, 43], [55, 54, 54, 62],
    [49, 48, 50, 55], [45, 43, 45, 50], [40, 39, 39, 38], [35, 34, 33, 33],
    [30, 29, 23, 28], [26, 25, 24, 24], [22, 21, 21, 21], [19, 17, 14, 21],
    [19, 16, 17, 16], [15, 14, 15, 14], [13, 12, 14, 12], [12, 10, 13, 11],
    [11, 9, 10, 10], [9, 7, 4, 10], [9, 7, 10, 7], [9, 7, 8, 7],
    [7, 6, 8, 6], [7, 5, 7, 5], [6, 5, 6, 4], [5, 4, 4, 5],
    [4, 3, 4, 4], [3, 2, 4, 3], [3, 1, 3, 1], [2, 1, 2, 1],
    [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 100],
    [87, 1, 1, 1],
]
BIDS = HEAD + [[1, 1, 1, 1]] * (103 - len(HEAD))
NAMES = ["rainbow_nash", "Onions", "Pieleda", "ImmigrantsI"]


def replay(seat, trace=False):
    bot = LearnedBot(Params())
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    blocks = 0
    for rnd in range(len(BIDS)):
        order = [seat] + [j for j in range(4) if j != seat]
        my = clamp_bid(int(bot.choose([pos[j] for j in order], [cum[j] for j in order])))
        bids = list(BIDS[rnd])
        bids[seat] = my
        others = [bids[j] for j in range(4) if j != seat]
        blk = step(pos, cum, bids)
        if seat in blk:
            blocks += 1
        if trace and rnd <= 27:
            mk = " BLOCKED" if seat in blk else ""
            print(f"  R{rnd:2d} our={my:3d}  others_max={max(others):3d}  "
                  f"ourpos={pos[seat]:4d}{mk}")
        if any(p >= TARGET for p in pos):
            break
    rank = sorted(range(4), key=lambda i: -pos[i]).index(seat) + 1
    return pos, blocks, rank, rnd + 1


def main():
    print("=== original outcome ===")
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    blk = [0, 0, 0, 0]
    for r in BIDS:
        for i in step(pos, cum, r):
            blk[i] += 1
    for i in range(4):
        print(f"  P{i} {NAMES[i]:14s} pos={pos[i]:4d} blocked={blk[i]}x")

    print("\n=== our bot per seat (others fixed) ===")
    for seat in range(4):
        p, b, rank, rnds = replay(seat)
        res = "WON" if rank == 1 else f"rank {rank}"
        print(f"  seat {seat} ({NAMES[seat]:14s}): pos={p[seat]:4d} "
              f"blocked={b}x  rounds={rnds}  -> {res}  field={sorted(p, reverse=True)}")

    print("\n=== seat 1 (Onions) early-game descent trace ===")
    replay(1, trace=True)


if __name__ == "__main__":
    main()
