"""Simulator for the Greed (snaky-greed) game. Runs two bot subprocesses."""
import random
import subprocess
import sys

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]


def run_game(cmd0, cmd1, seed, verbose=False):
    rng = random.Random(seed)
    grid = [[rng.randint(1, 9) for _ in range(GRID)] for _ in range(GRID)]
    grid_line = " ".join(str(grid[y][x]) for y in range(GRID) for x in range(GRID))

    procs = [subprocess.Popen(c, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, bufsize=1) for c in (cmd0, cmd1)]
    pos = [list(STARTS[0]), list(STARTS[1])]
    claimed = set(STARTS)
    score = [1, 1]
    alive = [True, True]

    def send(p, s):
        procs[p].stdin.write(s + "\n")
        procs[p].stdin.flush()

    def recv(p):
        line = procs[p].stdout.readline()
        return line.strip() if line else ""

    rounds = 0
    try:
        while rounds < 4000 and (alive[0] or alive[1]):
            moves = [None, None]
            for p in (0, 1):
                if not alive[p]:
                    continue
                o = 1 - p
                if rounds == 0:
                    send(p, grid_line)
                send(p, f"{pos[p][0]} {pos[p][1]} {pos[o][0]} {pos[o][1]}")
            for p in (0, 1):
                if alive[p]:
                    moves[p] = recv(p)

            # build intended straight-line paths
            paths = [[], []]
            for p in (0, 1):
                if not alive[p] or moves[p] not in DIRS:
                    continue
                dx, dy = DIRS[moves[p]]
                ax, ay = pos[p][0] + dx, pos[p][1] + dy
                if 0 <= ax < GRID and 0 <= ay < GRID:
                    dist = grid[ay][ax]
                else:
                    dist = 1
                for k in range(1, dist + 1):
                    paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))

            maxlen = max(len(paths[0]), len(paths[1]))
            moving = [alive[0] and bool(paths[0]) or (alive[0] and moves[0] in DIRS),
                      alive[1] and bool(paths[1]) or (alive[1] and moves[1] in DIRS)]
            # a player that submitted a move but has empty path still acts (will die offgrid)
            done = [not (alive[p] and moves[p] in DIRS) for p in (0, 1)]
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
            if verbose:
                print(f"R{rounds}: p0={pos[0]} s{score[0]} a{alive[0]} | "
                      f"p1={pos[1]} s{score[1]} a{alive[1]}")
    finally:
        for p in procs:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=2)
            except Exception:
                p.kill()
    return score, alive, rounds


if __name__ == "__main__":
    bot = [sys.executable, "C:/Users/16323/Desktop/hackthon/submission/snaky_greed.py"]
    rnd = [sys.executable, "C:/Users/16323/Desktop/hackthon/tools/greed_random.py"]
    wins = [0, 0, 0]
    for seed in range(20):
        a, b = (bot, rnd) if seed % 2 == 0 else (rnd, bot)
        score, alive, rounds = run_game(a, b, seed)
        botidx = 0 if seed % 2 == 0 else 1
        bs, os_ = score[botidx], score[1 - botidx]
        res = 0 if bs > os_ else (2 if bs < os_ else 1)
        wins[res] += 1
        print(f"seed {seed:2d}: bot={bs:4d} opp={os_:4d} rounds={rounds:4d} "
              f"{'WIN' if res == 0 else ('TIE' if res == 1 else 'LOSS')}")
    print(f"\nbot W/T/L vs random: {wins[0]}/{wins[1]}/{wins[2]}")
