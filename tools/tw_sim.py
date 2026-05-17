"""Local Territory Wars simulator/evaluator.

Runs one submitted bot as player 0 against three deterministic greedy bots.
This is intended for quick regression checks and relative comparisons, not as
an official judge replacement.
"""

import argparse
import collections
import random
import subprocess
import sys

SIZE = 31
MAX_ROUNDS = 512
DIRS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
STARTS = [(0, 0), (30, 0), (0, 30), (30, 30)]


def passable(owner, player, x, y):
    if not (0 <= x < SIZE and 0 <= y < SIZE):
        return False
    o = owner[y][x]
    return o < 0 or o == player


def reachable_unclaimed(owner, player, start):
    sx, sy = start
    if not passable(owner, player, sx, sy):
        return 0
    q = collections.deque([(sx, sy)])
    seen = {(sx, sy)}
    count = 0
    while q:
        x, y = q.popleft()
        if owner[y][x] < 0:
            count += 1
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if (nx, ny) not in seen and passable(owner, player, nx, ny):
                seen.add((nx, ny))
                q.append((nx, ny))
    return count


def greedy_move(owner, pos, alive, player, rng):
    x, y = pos[player]
    opts = []
    for mv, (dx, dy) in DIRS.items():
        nx, ny = x + dx, y + dy
        if not passable(owner, player, nx, ny):
            continue
        old = owner[ny][nx]
        if old < 0:
            owner[ny][nx] = player
        gain = reachable_unclaimed(owner, player, (nx, ny))
        if old < 0:
            owner[ny][nx] = old
        fresh = 1 if old < 0 else 0
        exits = sum(
            1
            for ddx, ddy in DIRS.values()
            if passable(owner, player, nx + ddx, ny + ddy)
        )
        jitter = rng.random() * 0.001
        opts.append((fresh * 10000 + gain * 20 + exits + jitter, mv))
    if not opts:
        return "u"
    return max(opts)[1]


def sweeper_move(owner, pos, alive, player, rng, last_dir):
    """P1/Algoholics-style boustrophedon sweeper: claim a fresh cell every
    turn, keep straight runs, hug walls so the fill stays compact, and never
    seal off reachable territory. This is the real threat model - the greedy
    bot above wanders far more than the platform's top bot does."""
    x, y = pos[player]
    opts = []
    for mv, (dx, dy) in DIRS.items():
        nx, ny = x + dx, y + dy
        if not passable(owner, player, nx, ny):
            continue
        old = owner[ny][nx]
        if old < 0:
            owner[ny][nx] = player
        gain = reachable_unclaimed(owner, player, (nx, ny))
        if old < 0:
            owner[ny][nx] = old
        fresh = 1 if old < 0 else 0
        straight = 1 if mv == last_dir else 0
        exits = sum(
            1
            for ddx, ddy in DIRS.values()
            if passable(owner, player, nx + ddx, ny + ddy)
        )
        jitter = rng.random() * 0.001
        # fresh cell first; among those keep momentum and a tight wall-hug;
        # gain keeps it from sealing itself into a dead pocket.
        score = (
            fresh * 100000
            + gain * 60
            + straight * 4000
            - exits * 250
            + jitter
        )
        opts.append((score, mv))
    if not opts:
        return "u"
    return max(opts)[1]


def bot_line_for(player, pos, alive):
    """One input line from `player`'s perspective: own head first, then the
    other three in index order (matching the platform's IN format)."""
    order = [player] + [p for p in range(4) if p != player]
    vals = []
    for p in order:
        if alive[p]:
            vals.extend(pos[p])
        else:
            vals.extend((-1, -1))
    return " ".join(str(v) for v in vals)


def bot_line(pos, alive):
    return bot_line_for(0, pos, alive)


def read_move(proc):
    line = proc.stdout.readline()
    if not line:
        return ""
    return line.strip()[:1]


def _spawn(cmd):
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1,
    )


def run_game(cmd, seed, verbose=False, opp="greedy", ref=None):
    rng = random.Random(seed)
    # P0 is always the candidate bot. P1-3 are:
    #  --ref BOT  -> three copies of BOT (clean A/B: candidate vs reference)
    #  opp=self   -> three copies of the candidate itself
    #  otherwise  -> greedy / sweeper opponents (note: those self-seal early,
    #                so their scores are not representative)
    selfplay = opp == "self"
    procs = [_spawn(cmd)]
    for _ in range(3):
        if ref is not None:
            procs.append(_spawn(ref))
        elif selfplay:
            procs.append(_spawn(cmd))
        else:
            procs.append(None)
    proc = procs[0]
    owner = [[-1] * SIZE for _ in range(SIZE)]
    pos = [list(p) for p in STARTS]
    alive = [True] * 4
    score = [1] * 4
    p0_hist = []
    p0_bounces = 0
    p0_wasted = 0
    last_dir = [None] * 4
    for p, (x, y) in enumerate(STARTS):
        owner[y][x] = p

    try:
        for rnd in range(MAX_ROUNDS):
            if not any(alive):
                break

            moves = [""] * 4
            for p in range(4):
                if not alive[p]:
                    continue
                if procs[p] is not None:           # a bot subprocess
                    procs[p].stdin.write(bot_line_for(p, pos, alive) + "\n")
                    procs[p].stdin.flush()
                    moves[p] = read_move(procs[p])
                else:                              # greedy / sweeper opponent
                    use_sweeper = opp == "sweeper" or (opp == "mixed" and p == 1)
                    if use_sweeper:
                        moves[p] = sweeper_move(
                            owner, pos, alive, p, rng, last_dir[p]
                        )
                    else:
                        moves[p] = greedy_move(owner, pos, alive, p, rng)
                    last_dir[p] = moves[p]

            intended = [None] * 4
            deaths = set()
            for p, mv in enumerate(moves):
                if not alive[p]:
                    continue
                if mv not in DIRS:
                    deaths.add(p)
                    continue
                dx, dy = DIRS[mv]
                nx, ny = pos[p][0] + dx, pos[p][1] + dy
                intended[p] = (nx, ny)
                if not passable(owner, p, nx, ny):
                    deaths.add(p)

            cells = {}
            for p, cell in enumerate(intended):
                if alive[p] and p not in deaths and cell is not None:
                    cells.setdefault(cell, []).append(p)
            for ps in cells.values():
                if len(ps) > 1:
                    deaths.update(ps)

            for p in deaths:
                alive[p] = False

            p0_ru_before = (
                reachable_unclaimed(owner, 0, tuple(pos[0]))
                if alive[0] and 0 not in deaths
                else 0
            )
            for p, cell in enumerate(intended):
                if not alive[p] or p in deaths or cell is None:
                    continue
                nx, ny = cell
                fresh = owner[ny][nx] < 0
                pos[p] = [nx, ny]
                if fresh:
                    owner[ny][nx] = p
                    score[p] += 1
                elif p == 0 and p0_ru_before > 0:
                    p0_wasted += 1          # stepped on an owned cell with
                                            # fresh territory still reachable

            if alive[0]:
                p0_hist.append(tuple(pos[0]))
                if len(p0_hist) >= 3 and p0_hist[-1] == p0_hist[-3]:
                    if reachable_unclaimed(owner, 0, tuple(pos[0])) > 0:
                        p0_bounces += 1

            if verbose:
                print(
                    f"R{rnd + 1:03d} score={score} alive={alive} "
                    f"pos={pos} moves={moves}"
                )
        rounds = rnd + 1
    finally:
        for pr in procs:
            if pr is None:
                continue
            try:
                pr.stdin.close()
            except Exception:
                pass
            try:
                pr.wait(timeout=2)
            except Exception:
                pr.kill()

    rank = 1 + sum(s > score[0] for s in score[1:])
    return {
        "score": score,
        "alive": alive,
        "rounds": rounds,
        "rank": rank,
        "bounces": p0_bounces,
        "wasted": p0_wasted,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot", nargs="+", help="bot command, for example ./tw.exe")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--opp",
        choices=("greedy", "sweeper", "mixed", "self"),
        default="greedy",
        help="opponent type: greedy (weak), sweeper (P1 model), "
        "mixed (1 sweeper + 2 greedy), self (the bot vs 3 copies of itself)",
    )
    ap.add_argument(
        "--ref",
        default=None,
        help="reference bot binary for P1-3 (clean A/B: candidate vs ref)",
    )
    args = ap.parse_args()
    ref = [args.ref] if args.ref else None

    ranks = [0, 0, 0, 0]
    total_score = 0
    total_bounces = 0
    total_wasted = 0
    for i in range(args.games):
        res = run_game(args.bot, args.seed + i, args.verbose, args.opp, ref)
        ranks[res["rank"] - 1] += 1
        total_score += res["score"][0]
        total_bounces += res["bounces"]
        total_wasted += res["wasted"]
        print(
            f"seed {args.seed + i:3d}: p0={res['score'][0]:3d} "
            f"all={res['score']} rank={res['rank']} "
            f"rounds={res['rounds']:3d} bounces={res['bounces']} "
            f"wasted={res['wasted']}"
        )

    n = args.games
    print(
        f"\nrank1/2/3/4={ranks[0]}/{ranks[1]}/{ranks[2]}/{ranks[3]} "
        f"avg_score={total_score / n:.2f} "
        f"avg_bounces={total_bounces / n:.2f} "
        f"avg_wasted={total_wasted / n:.2f}"
    )


if __name__ == "__main__":
    sys.exit(main())
