"""Replay of observed icarus-game matches.

For each seat we substitute our LearnedBot and keep the other three players'
bids fixed to what they actually played in the log. This is a fixed-opponent
counterfactual: it shows whether our bot, fed the same opponents' bid stream,
would have finished ahead of the bot that originally held the seat.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))

from icarus_rl_train import LearnedBot, Params, clamp_bid  # noqa: E402

TARGET = 999

# rows = rounds, cols = P0,P1,P2,P3.
MATCHES = {
    "yelowsub/Ramon/Immigrants2/Algoholics": {
        "names": ["yelowsub80prcent", "Ramon", "Immigrants2", "Algoholics"],
        "us": None,  # we were not in this match
        "bids": [
            [42, 48, 47, 88], [40, 68, 74, 44], [45, 65, 62, 64],
            [65, 59, 55, 60], [61, 61, 55, 56], [57, 58, 51, 57],
            [58, 56, 49, 53], [54, 55, 49, 52], [53, 52, 46, 50],
            [51, 51, 45, 48], [49, 49, 43, 47], [48, 47, 41, 45],
            [46, 46, 40, 43], [44, 44, 39, 42], [43, 42, 37, 40],
            [41, 41, 36, 38], [39, 39, 34, 37], [38, 37, 33, 35],
            [36, 36, 32, 33], [34, 34, 30, 32], [33, 32, 28, 30],
            [31, 31, 28, 28], [29, 30, 26, 27], [28, 28, 25, 25],
            [26, 26, 23, 13],
        ],
    },
    "Algoholics/Ovon1/Neumannism/rainbow_nash": {
        "names": ["Algoholics", "Ovon1", "Neumannism", "rainbow_nash"],
        "us": None,  # we were not in this match
        "bids": [
            [88, 45, 46, 50], [46, 82, 52, 79], [75, 69, 74, 74],
            [70, 67, 72, 68], [66, 65, 67, 65], [62, 62, 65, 61],
            [58, 59, 60, 59], [55, 56, 58, 54], [54, 54, 51, 53],
            [50, 52, 52, 49], [48, 48, 48, 47], [44, 40, 44, 44],
            [40, 40, 42, 39], [38, 38, 39, 38], [35, 37, 36, 35],
            [33, 34, 30, 33], [30, 32, 29, 30], [28, 30, 31, 29],
            [27, 28, 27, 28], [24, 26, 26, 25], [22, 24, 22, 23],
            [20, 22, 22, 21], [18, 17, 18, 19], [15, 18, 15, 16],
            [14, 16, 12, 16], [12, 13, 13, 14], [10, 12, 11, 11],
        ],
    },
    "Trissis/Gaussian/Pieleda/3Phys1Math": {
        "names": ["TrissisTichuTeam", "GaussianGamblers", "Pieleda",
                  "3Phys1Math"],
        "us": 3,  # P3 = our bot (verified: same OUT stream)
        "bids": [
            [50, 100, 98, 63], [13, 98, 95, 87], [78, 96, 89, 90],
            [76, 48, 93, 88], [76, 60, 87, 79], [76, 54, 73, 77],
            [69, 58, 72, 65], [62, 53, 64, 62], [56, 54, 59, 59],
            [53, 51, 56, 50], [50, 60, 50, 50], [43, 50, 47, 47],
            [44, 45, 46, 44], [41, 46, 44, 37], [37, 50, 41, 41],
            [38, 53, 40, 38], [35, 51, 48, 37], [35, 59, 46, 45],
            [40, 55, 58, 76], [40, 60, 44, 76],
        ],
    },
}


def step(pos, cum, bids):
    """Apply one icarus round: highest bidder is blocked; ties -> lowest cum."""
    for i in range(4):
        cum[i] += bids[i]
    hi = max(bids)
    tied = [i for i in range(4) if bids[i] == hi]
    low_cum = min(cum[i] for i in tied)
    blocked = {i for i in tied if cum[i] == low_cum}
    for i in range(4):
        if i not in blocked:
            pos[i] += bids[i]
    return blocked


def replay(bid_table, seat, verbose=False):
    """Run the match with our bot in `seat`, others fixed to the log."""
    bot = LearnedBot(Params())
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    our_blocks = 0
    for rnd in range(len(bid_table)):
        order = [seat] + [j for j in range(4) if j != seat]
        rp = [pos[j] for j in order]
        rc = [cum[j] for j in order]
        my_bid = clamp_bid(int(bot.choose(rp, rc)))

        bids = list(bid_table[rnd])
        bids[seat] = my_bid
        others = [bids[j] for j in range(4) if j != seat]
        blocked = step(pos, cum, bids)
        if seat in blocked:
            our_blocks += 1
        if verbose:
            mark = " <-- BLOCKED" if seat in blocked else ""
            print(f"  R{rnd:2d} ourbid={my_bid:3d}  others={sorted(others, reverse=True)}"
                  f"  max_other={max(others):3d}  ourpos={pos[seat]:4d}{mark}")
        if any(p >= TARGET for p in pos):
            break

    rank = sorted(range(4), key=lambda i: -pos[i]).index(seat) + 1
    return pos, cum, rank, our_blocks, rnd + 1


def main():
    for title, m in MATCHES.items():
        bids, names, us = m["bids"], m["names"], m["us"]
        print(f"=== {title} ===")

        pos = [0, 0, 0, 0]
        cum = [0, 0, 0, 0]
        blk = [0, 0, 0, 0]
        for rnd in range(len(bids)):
            for i in step(pos, cum, bids[rnd]):
                blk[i] += 1
        for i in range(4):
            tag = "  <- us" if i == us else ""
            print(f"  original P{i} {names[i]:18s} pos={pos[i]:4d} "
                  f"cum={cum[i]:4d} blocked={blk[i]}x{tag}")

        if us is not None:
            print(f"  -- per-round trace, our bot in seat {us} --")
            replay(bids, us, verbose=True)

        for seat in range(4):
            p, c, rank, blocks, _ = replay(bids, seat)
            res = "WON" if rank == 1 else f"rank {rank}"
            print(f"  our bot @ seat {seat} ({names[seat]:18s}): pos={p[seat]:4d} "
                  f"blocked={blocks}x -> {res}")
        print()


if __name__ == "__main__":
    main()
