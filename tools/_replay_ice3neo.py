"""One-off replay of the ICE3NEO vs 3Phys1Math (our bot) Greed game.

ICE3NEO (player 0) plays its recorded moves; our bot (player 1) plays live,
so we can A/B weight sets on a game we actually lost. Grid + opponent moves
are the real ones from that match.

Usage:
  python tools/_replay_ice3neo.py [bot_path]
  SG_STEP=1.5 python tools/_replay_ice3neo.py        # weight override
"""
import os
import subprocess
import sys
from pathlib import Path

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]
ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / "tools" / "_replay_ice3neo.txt"
OPP_MOVES = ("u r d l d d l d d r r r u u u u l u r u u l l d l d d "
             "l d r u l l l u u").split()          # ICE3NEO, rounds 0..35


def run(bot_cmd):
    nums = [int(v) for v in GRID_FILE.read_text().split()]
    grid = [[nums[y * GRID + x] for x in range(GRID)] for y in range(GRID)]
    gl = " ".join(str(v) for v in nums)

    proc = subprocess.Popen(bot_cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    pos = [list(STARTS[0]), list(STARTS[1])]
    claimed = set(STARTS)
    score = [1, 1]
    alive = [True, True]

    rounds = 0
    try:
        while rounds < 4000 and alive[1]:
            if rounds == 0:
                proc.stdin.write(gl + "\n")
            proc.stdin.write(f"{pos[1][0]} {pos[1][1]} {pos[0][0]} {pos[0][1]}\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            moves = [OPP_MOVES[rounds] if rounds < len(OPP_MOVES) else None,
                     line.strip() if line else ""]
            if rounds >= len(OPP_MOVES):
                alive[0] = False

            paths = [[], []]
            for p in (0, 1):
                if not alive[p] or moves[p] not in DIRS:
                    continue
                dx, dy = DIRS[moves[p]]
                ax, ay = pos[p][0] + dx, pos[p][1] + dy
                dist = grid[ay][ax] if 0 <= ax < GRID and 0 <= ay < GRID else 1
                for k in range(1, dist + 1):
                    paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))

            maxlen = max(len(paths[0]), len(paths[1]))
            done = [not (alive[p] and moves[p] in DIRS) for p in (0, 1)]
            for p in (0, 1):
                if alive[p] and moves[p] not in DIRS:
                    alive[p] = False
            for k in range(maxlen):
                cell = {}
                for p in (0, 1):
                    if done[p]:
                        continue
                    if k >= len(paths[p]):
                        done[p] = True
                        continue
                    cell[p] = paths[p][k]
                deaths = set()
                for p, c in cell.items():
                    cx, cy = c
                    if not (0 <= cx < GRID and 0 <= cy < GRID) or c in claimed:
                        deaths.add(p)
                if len(cell) == 2 and cell[0] == cell[1]:
                    deaths.update((0, 1))
                for p, c in cell.items():
                    if p in deaths:
                        alive[p] = False
                        done[p] = True
                    else:
                        claimed.add(c)
                        pos[p] = list(c)
                        score[p] += 1
                if all(done):
                    break
            rounds += 1
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    return score, alive, rounds


def main():
    bot_path = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "submission" / "snaky_greed.py")
    p = Path(bot_path)
    cmd = [sys.executable, str(p)] if p.suffix.lower() == ".py" else [str(p)]
    score, alive, rounds = run(cmd)
    bot, opp = score[1], score[0]
    res = "WIN" if bot > opp else ("TIE" if bot == opp else "LOSS")
    knobs = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                     if k.startswith("SG_"))
    print(f"bot(P1)={bot:4d}  ICE3NEO(P0)={opp:4d}  rounds={rounds:3d}  "
          f"bot_alive={alive[1]}  -> {res}    [{knobs or 'defaults'}]")


if __name__ == "__main__":
    main()
