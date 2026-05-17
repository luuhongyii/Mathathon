"""Head-to-head: current submission bot vs the codex_version bot.

Both teams run a real compiled bot. Plays each board twice with sides
swapped so neither side gets a corner/first-move advantage.
"""
import os
import subprocess
import sys

SIZE = 29
HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
import ctf_sim as sim

MAIN = os.path.join(ROOT, "submission", "capture_the_flag.exe")
CODEX = os.path.join(ROOT, "submission", "codex_version", "capture_the_flag.exe")


class ExeProc:
    """One bot subprocess for a specific executable path."""

    def __init__(self, exe, board):
        self.p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.p.stdin.write(board + "\n")
        self.p.stdin.flush()

    def move(self, state):
        self.p.stdin.write(state + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline().strip()
        return line if line in ("u", "d", "l", "r", "s") else None

    def close(self):
        try:
            self.p.stdin.close()
            self.p.kill()
        except Exception:
            pass


def play(board, blue_exe, red_exe):
    bots = [ExeProc(blue_exe, board), ExeProc(blue_exe, board),
            ExeProc(red_exe, board), ExeProc(red_exe, board)]
    result = sim.run_game(board, bots)
    for b in bots:
        b.close()
    return result[0], result[1]


def main():
    games = 12
    win = {"main": 0, "codex": 0, "draw": 0, "dq": 0}
    for g in range(games):
        rng = __import__("random").Random(5000 + g)
        board = sim.REAL_BOARD if g % 3 == 0 else sim.gen_board(rng)
        # main as blue
        r, t = play(board, MAIN, CODEX)
        if r == "blue":
            win["main"] += 1
        elif r == "red":
            win["codex"] += 1
        else:
            win[r] = win.get(r, 0) + 1
        print("game %2dA: main=blue  -> %-6s turn=%s" % (g, r, t))
        # main as red (sides swapped)
        r, t = play(board, CODEX, MAIN)
        if r == "red":
            win["main"] += 1
        elif r == "blue":
            win["codex"] += 1
        else:
            win[r] = win.get(r, 0) + 1
        print("game %2dB: main=red   -> %-6s turn=%s" % (g, r, t))
    print("-" * 34)
    print("head-to-head:", win)


if __name__ == "__main__":
    main()
