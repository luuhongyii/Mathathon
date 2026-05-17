"""Replay the 2nd ICE3NEO vs 3Phys1Math (our bot) Greed game and check
whether the CURRENT bot reproduces the logged moves.

If the live bot's moves match the recorded P1 line, the log was produced by
this exact bot. If they diverge, the uploaded file was a different/old build.

Usage:
  python tools/_replay_check.py [bot_path]
"""
import os
import subprocess
import sys
from pathlib import Path

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]
ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / "tools" / "_replay_ice3neo2.txt"

OPP_MOVES = "r d l d d r r r r u u u l".split()        # ICE3NEO (P0), R0..12
LOG_P1 = "l u u u r r r r d d r u u".split()           # our bot in the log


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
    bot_moves = []

    rounds = 0
    try:
        while rounds < 4000 and alive[1]:
            if rounds == 0:
                proc.stdin.write(gl + "\n")
            proc.stdin.write(f"{pos[1][0]} {pos[1][1]} {pos[0][0]} {pos[0][1]}\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            bm = line.strip() if line else ""
            bot_moves.append(bm)
            moves = [OPP_MOVES[rounds] if rounds < len(OPP_MOVES) else None, bm]
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
    return score, alive, rounds, bot_moves


def main():
    bot_path = sys.argv[1] if len(sys.argv) > 1 else str(
        ROOT / "submission" / "snaky_greed.py")
    p = Path(bot_path)
    cmd = [sys.executable, str(p)] if p.suffix.lower() == ".py" else [str(p)]
    score, alive, rounds, bm = run(cmd)

    print(f"bot: {bot_path}")
    print(f"{'round':>5}  {'logged':>6}  {'live':>4}  match")
    diverge = None
    for i in range(min(len(LOG_P1), len(bm))):
        ok = LOG_P1[i] == bm[i]
        if not ok and diverge is None:
            diverge = i
        print(f"{i:>5}  {LOG_P1[i]:>6}  {bm[i]:>4}  {'OK' if ok else 'DIFF <<'}")
    knobs = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                     if k.startswith("SG_"))
    print(f"\nweights: {knobs or 'shipped defaults'}")
    if diverge is None:
        print(f"VERDICT: live bot reproduces the log exactly -> the uploaded "
              f"file IS this bot. Final: bot={score[1]} opp={score[0]} "
              f"rounds={rounds} alive={alive[1]}")
    else:
        print(f"VERDICT: live bot DIVERGES at round {diverge} "
              f"(logged '{LOG_P1[diverge]}' vs live '{bm[diverge]}') -> the "
              f"log was produced by a DIFFERENT build, not this bot.")


if __name__ == "__main__":
    main()
