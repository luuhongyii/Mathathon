"""Deterministic benchmark for snaky_greed: seeded grid AND seeded opponent.

Usage:
  python tools/greed_bench.py [games] [bot_path]

bot_path may be a .py script (run with python) or a compiled binary (.exe / no ext).
Defaults to submission/snaky_greed.py relative to the repo root.
"""
import random
import subprocess
import sys
from pathlib import Path

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]
ROOT = Path(__file__).resolve().parents[1]
EX = sys.executable
RND = [EX, str(ROOT / "tools" / "greed_random.py")]


def bot_cmd(path: str) -> list:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.suffix.lower() == ".py":
        return [EX, str(p)]
    return [str(p)]


def run_game(cmd0, cmd1, seed):
    rng = random.Random(seed)
    grid = [[rng.randint(1, 9) for _ in range(GRID)] for _ in range(GRID)]
    gl = " ".join(str(grid[y][x]) for y in range(GRID) for x in range(GRID))
    # Append the seed as argv to BOTH commands. The random opponent reads
    # it (so its play is reproducible no matter which slot it sits in); the
    # real bot ignores extra argv. Previously only cmd1 got the seed, so on
    # odd seeds -- where the random bot is cmd0 -- the opponent ran
    # unseeded and the "deterministic" benchmark was anything but.
    c0 = list(cmd0) + [str(seed)]
    c1 = list(cmd1) + [str(seed)]
    procs = [subprocess.Popen(c, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, bufsize=1) for c in (c0, c1)]
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
    death_round = [None, None]
    try:
        while rounds < 4000 and (alive[0] or alive[1]):
            for p in (0, 1):
                if not alive[p]:
                    continue
                o = 1 - p
                if rounds == 0:
                    send(p, gl)
                send(p, f"{pos[p][0]} {pos[p][1]} {pos[o][0]} {pos[o][1]}")
            moves = [recv(p) if alive[p] else None for p in (0, 1)]
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
                    death_round[p] = rounds
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
                        if death_round[p] is None:
                            death_round[p] = rounds
                    else:
                        claimed.add(c)
                        pos[p] = list(c)
                        score[p] += 1
                if all(done):
                    break
            rounds += 1
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
    return score, alive, rounds, death_round


def main():
    n = 40
    bot_path = str(ROOT / "submission" / "snaky_greed.py")
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
            if len(sys.argv) > 2:
                bot_path = sys.argv[2]
        except ValueError:
            bot_path = sys.argv[1]
            if len(sys.argv) > 2:
                n = int(sys.argv[2])
    bot = bot_cmd(bot_path)
    print(f"bot: {bot}  games: {n}")
    wins = [0, 0, 0]
    early_deaths = 0
    for seed in range(n):
        a, b = (bot, RND) if seed % 2 == 0 else (RND, bot)
        score, alive, rounds, dr = run_game(a, b, seed)
        bi = 0 if seed % 2 == 0 else 1
        bs, os_ = score[bi], score[1 - bi]
        res = 0 if bs > os_ else (2 if bs < os_ else 1)
        wins[res] += 1
        bot_dr = dr[bi]
        if bot_dr is not None and bot_dr < 30:
            early_deaths += 1
        tag = ['WIN', 'TIE', 'LOSS'][res]
        ed = f' BOT-DIED@{bot_dr}' if bot_dr is not None and bot_dr < 30 else ''
        print(f"seed {seed:2d}: bot={bs:4d} opp={os_:4d} r={rounds:3d} {tag}{ed}")
    print(f"\nW/T/L: {wins[0]}/{wins[1]}/{wins[2]}  "
          f"winrate={wins[0]/n*100:.0f}%  early_deaths={early_deaths}/{n}")


if __name__ == "__main__":
    main()
