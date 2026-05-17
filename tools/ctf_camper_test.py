"""Regression test: our bot vs an oasis-camper opponent (the tungtung case).

The platform game P3 lost showed the enemy parking a passive bot in the oasis,
which walled our attacker off its shortest path to the flag and froze it on 's'
for 110+ rounds (0 captures). This drives our compiled bot (blue, players 0/1)
against red opponents that just sit in the oasis and verifies blue still scores.
"""
import os
import sys

SIZE = 29
EXE = os.path.join(os.path.dirname(__file__), "..", "submission",
                   "capture_the_flag.exe")
os.environ["CTF_BOT"] = EXE

import ctf_sim as sim


class CamperBot:
    """Red bot that walks to a fixed oasis cell and stays there forever."""

    def __init__(self, board, target):
        self.obs = {(i % SIZE, i // SIZE)
                    for i, c in enumerate(board) if c == "#"}
        self.tx, self.ty = target

    def move(self, state):
        nums = [int(n) for n in state.split()]
        x, y = nums[0], nums[1]
        if x < 0:
            return "s"
        if (x, y) == (self.tx, self.ty):
            return "s"
        best = ("s", abs(x - self.tx) + abs(y - self.ty))
        for mv, dx, dy in (("u", 0, -1), ("d", 0, 1), ("l", -1, 0),
                           ("r", 1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                d = abs(nx - self.tx) + abs(ny - self.ty)
                if d < best[1]:
                    best = (mv, d)
        return best[0]


def run_scenario(name, camp_targets, games=8):
    tally = {"blue": 0, "red": 0, "draw": 0, "dq": 0}
    for g in range(games):
        board = sim.REAL_BOARD
        bots = []
        for p in range(4):
            if p < 2:
                bots.append(sim.Proc(board))
            else:
                bots.append(CamperBot(board, camp_targets[p - 2]))
        result = sim.run_game(board, bots)
        for b in bots:
            if isinstance(b, sim.Proc):
                b.close()
        tally[result[0]] = tally.get(result[0], 0) + 1
        print("  %-14s game %d: %-6s turn=%s" % (name, g, result[0], result[1]))
    return tally


def main():
    # Scenario 1 (attack path): campers on the blue-facing oasis edge wall off
    # an attacker's shortest path to the red flag.
    # Scenario 2 (carry path): campers just inside red territory at the oasis
    # edge sit on the chokepoint the carrier crosses on the way home.
    scenarios = [
        ("attack-path", [(16, 12), (12, 16)]),
        ("carry-path", [(16, 15), (12, 15)]),
    ]
    ok = True
    for name, targets in scenarios:
        tally = run_scenario(name, targets)
        print("  %-14s tally: %s" % (name, tally))
        if tally["blue"] < 6:
            ok = False
    print("-" * 30)
    print("PASS" if ok else "FAIL (bot still stalls against a camper)")


if __name__ == "__main__":
    main()
