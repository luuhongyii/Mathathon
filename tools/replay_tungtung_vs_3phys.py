"""Replay tungtungsahuur (P0) vs 3Phys1Math (P1) R0 and compare live bot moves."""
import subprocess
import sys
from pathlib import Path

GRID = 32
DIRS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
STARTS = [(8, 8), (23, 23)]
ROOT = Path(__file__).resolve().parents[1]
GRID_FILE = ROOT / "tools" / "gamelogs" / "tungtung_vs_3phys_r0_grid.txt"

OPP_MOVES = "r d r d l d d r d l u l u u r u l l d d l d d d".split()
LOG_P1 = "u l u r r d d d r u u u l u u l u u l l l".split()


def simulate_game():
    nums = [int(v) for v in GRID_FILE.read_text().split()]
    grid = [[nums[y * GRID + x] for x in range(GRID)] for y in range(GRID)]
    pos = [list(STARTS[0]), list(STARTS[1])]
    claimed = set(STARTS)
    score = [1, 1]
    alive = [True, True]
    n = min(len(OPP_MOVES), len(LOG_P1))
    print("Round  P0   P1   score0 score1  pos0      pos1")
    for rnd in range(n):
        moves = [OPP_MOVES[rnd], LOG_P1[rnd]]
        paths = [[], []]
        for p in (0, 1):
            if not alive[p]:
                continue
            dx, dy = DIRS[moves[p]]
            ax, ay = pos[p][0] + dx, pos[p][1] + dy
            dist = grid[ay][ax]
            for k in range(1, dist + 1):
                paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))
        maxlen = max(len(paths[0]), len(paths[1]))
        done = [False, False]
        for k in range(maxlen):
            cell = {}
            for p in (0, 1):
                if done[p] or not alive[p]:
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
            if len(cell) == 2 and cell.get(0) == cell.get(1):
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
        print(
            f"{rnd:>5}  {moves[0]:>3}  {moves[1]:>3}  "
            f"{score[0]:>6} {score[1]:>6}  "
            f"({pos[0][0]:>2},{pos[0][1]:>2})  ({pos[1][0]:>2},{pos[1][1]:>2})"
            + ("" if alive[0] and alive[1] else "  DEATH")
        )
        if not (alive[0] and alive[1]):
            break
    print(f"\nFinal score: P0={score[0]} P1={score[1]}  alive={alive}")


def run_bot(bot_cmd):
    nums = [int(v) for v in GRID_FILE.read_text().split()]
    gl = " ".join(str(v) for v in nums)
    proc = subprocess.Popen(
        bot_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
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
            proc.stdin.write(
                f"{pos[1][0]} {pos[1][1]} {pos[0][0]} {pos[0][1]}\n"
            )
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
                dist = grid_val(nums, ax, ay)
                for k in range(1, dist + 1):
                    paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))
            maxlen = max(len(paths[0]), len(paths[1]) if paths[1] else 0)
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
    return score, bot_moves


def grid_val(nums, x, y):
    if 0 <= x < GRID and 0 <= y < GRID:
        return nums[y * GRID + x]
    return 1


def main():
    if "--sim" in sys.argv:
        simulate_game()
        return
    bot = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1] != "--sim"
        else str(ROOT / "agent" / "snaky_greed.exe")
    )
    p = Path(bot)
    cmd = [sys.executable, str(p)] if p.suffix.lower() == ".py" else [str(p)]
    score, bm = run_bot(cmd)
    print(f"bot: {bot}")
    print(f"{'rnd':>4}  log  live  match")
    div = None
    for i in range(min(len(LOG_P1), len(bm))):
        ok = LOG_P1[i] == bm[i]
        if not ok and div is None:
            div = i
        print(f"{i:>4}  {LOG_P1[i]:>3}  {bm[i]:>4}  {'OK' if ok else 'DIFF'}")
    print(f"\nFirst divergence: round {div if div is not None else 'none (all match)'}")
    print(f"Score after replay: P0={score[0]} P1={score[1]}")


if __name__ == "__main__":
    main()
