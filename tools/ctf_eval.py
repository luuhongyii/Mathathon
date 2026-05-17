"""Evaluate a candidate bot vs a reference bot over many games, both colors.
Uses fresh game seeds each run so the candidate's PID-seeded RNG is averaged.
Usage: python tools/ctf_eval.py <cand_exe> <ref_exe> [games_per_side]
Reports candidate win-rate as blue and as red, vs reference.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ctf_sim as S


def play(blue, red, n, seed0):
    tally = {"blue": 0, "red": 0, "draw": 0}
    for g in range(n):
        rng = random.Random(seed0 + g)
        random.seed(seed0 + 100000 + g)
        board = S.REAL_BOARD if g % 4 == 0 else S.gen_board(rng)
        bots = []
        for p in range(4):
            S.BOT = blue if p < 2 else red
            bots.append(S.Proc(board))
        result = S.run_game(board, bots)
        for b in bots:
            b.close()
        tally[result[0]] = tally.get(result[0], 0) + 1
    return tally


def main():
    cand = sys.argv[1]
    ref = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    cb = play(cand, ref, n, 4000)        # candidate is blue
    cr = play(ref, cand, n, 9000)        # candidate is red
    cand_wins = cb["blue"] + cr["red"]
    ref_wins = cb["red"] + cr["blue"]
    draws = cb["draw"] + cr["draw"]
    tot = 2 * n
    print("candidate as blue: %s" % cb)
    print("candidate as red:  %s" % cr)
    print("-" * 40)
    print("candidate wins: %d  reference wins: %d  draws: %d  (of %d)"
          % (cand_wins, ref_wins, draws, tot))
    print("candidate win share (excl draws): %.1f%%"
          % (100.0 * cand_wins / max(1, cand_wins + ref_wins)))


if __name__ == "__main__":
    main()
