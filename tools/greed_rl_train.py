"""Offline RL-style tuner for the Greed (snaky-greed) bot.

This does NOT replace the submitted bot. It is a black-box policy search over
the bot's value-function weights -- the bot already exposes every weight via
SG_* environment variables, so the trainer just spawns the real bot binary
with different env and never has to re-implement its search.

Algorithm: Cross-Entropy Method (CEM), the same approach as
icarus_rl_train.py. CEM fits here because the parameter vector is small
(7 continuous weights), the objective is noisy, and the simulator (a head-to-
head Greed game) is a clean black box. Each candidate weight vector is scored
by its average match result across an opponent suite; CEM keeps the elite
fraction each generation and re-fits a Gaussian to them.

Opponent suite (per the training plan):
  * random  -- tools/greed_random.py, the safe-random baseline
  * codex   -- submission/codex_version/snaky_greed.py, the rival to beat
  * self    -- submission/snaky_greed.py at SHIPPED defaults (a fixed mirror;
               pressures the candidate to strictly improve on the baseline)

Usage:
  python tools/greed_rl_train.py                 # train, print best weights
  python tools/greed_rl_train.py --eval-only     # score the shipped defaults
  python tools/greed_rl_train.py --generations 6 --population 20
"""

from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]

ROOT = Path(__file__).resolve().parents[1]
EX = sys.executable
BOT = ROOT / "submission" / "snaky_greed.py"
CODEX = ROOT / "submission" / "codex_version" / "snaky_greed.py"
RANDOM = ROOT / "tools" / "greed_random.py"

# ---------------------------------------------------------------------------
# Parameter space. Each name is an SG_* env var the bot already reads. Ints
# (SG_SURVCAP) and on/off flags (SG_CFILTER, SG_PESSIM) are deliberately NOT
# tuned -- SURV_CAP=12 was found best by hand and the two filters are simply
# kept on; CEM over a continuous box is what this method is good at.
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "SG_SPACE":   (0.20, 2.50),   # weight on neck-aware usable space
    "SG_VOR":     (0.30, 3.50),   # weight on Voronoi territory differential
    "SG_RCH":     (0.00, 0.60),   # weight on raw reachable-cell count
    "SG_STEP":    (0.30, 3.00),   # weight on points scored this move
    "SG_SURVPEN": (20.0, 120.0),  # penalty per missing survival-depth step
    "SG_COLLIDE": (0.0, 150.0),   # penalty for a path crossing opponent reach
    "SG_PESSIMW": (0.50, 5.00),   # multiplier on the pessimistic-survival pen
}
PARAM_NAMES = list(PARAM_BOUNDS)

# Shipped defaults (must match the os.environ.get fallbacks in snaky_greed.py).
DEFAULTS = {
    "SG_SPACE": 0.8,
    "SG_VOR": 1.2,
    "SG_RCH": 0.15,
    "SG_STEP": 1.0,
    "SG_SURVPEN": 55.0,
    "SG_COLLIDE": 30.0,
    "SG_PESSIMW": 2.0,
}

OPPONENTS = ("random", "codex", "self")


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def vec_to_params(v: list[float]) -> dict:
    return {name: clamp(x, *PARAM_BOUNDS[name]) for name, x in zip(PARAM_NAMES, v)}


def params_to_vec(p: dict) -> list[float]:
    return [p[name] for name in PARAM_NAMES]


# ---------------------------------------------------------------------------
# Environments. The candidate bot gets the SG_* weights; every opponent runs
# with a clean env (SG_* stripped) so it plays at shipped defaults -- the
# `self` mirror is therefore the *baseline* bot, a fixed target to beat.
# ---------------------------------------------------------------------------
_CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("SG_")}


def candidate_env(params: dict) -> dict:
    env = dict(_CLEAN_ENV)
    for name, val in params.items():
        env[name] = repr(float(val))
    return env


def opponent_cmd_env(kind: str, seed: int):
    """Return (cmd_list, env) for an opponent process."""
    if kind == "random":
        return [EX, str(RANDOM), str(seed)], _CLEAN_ENV
    if kind == "codex":
        return [EX, str(CODEX)], _CLEAN_ENV
    if kind == "self":
        return [EX, str(BOT)], _CLEAN_ENV
    raise ValueError(f"unknown opponent {kind!r}")


# ---------------------------------------------------------------------------
# Head-to-head Greed simulator. Ported from tools/greed_bench.py:run_game so
# the trainer has no subprocess dependency on the bench script.
# ---------------------------------------------------------------------------
def run_game(cmd0, env0, cmd1, env1, seed):
    rng = random.Random(seed)
    grid = [[rng.randint(1, 9) for _ in range(GRID)] for _ in range(GRID)]
    gl = " ".join(str(grid[y][x]) for y in range(GRID) for x in range(GRID))

    procs = [subprocess.Popen(c, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, bufsize=1, env=e)
             for c, e in ((cmd0, env0), (cmd1, env1))]
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
        for p in procs:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
    return score


def play(params: dict, opp: str, seed: int) -> float:
    """One game of the candidate vs `opp`. Returns 1.0 win / 0.5 tie / 0.0 loss.

    Sides are alternated by seed parity so the candidate is scored equally as
    player 0 (start 8,8) and player 1 (start 23,23)."""
    cand = ([EX, str(BOT)], candidate_env(params))
    ocmd, oenv = opponent_cmd_env(opp, seed)
    other = (ocmd, oenv)
    if seed % 2 == 0:
        score = run_game(*cand, *other, seed)
        my, op = score[0], score[1]
    else:
        score = run_game(*other, *cand, seed)
        my, op = score[1], score[0]
    return 1.0 if my > op else (0.5 if my == op else 0.0)


# ---------------------------------------------------------------------------
# Scoring + CEM.
# ---------------------------------------------------------------------------
def score(params: dict, seeds: int, seed0: int, pool: ThreadPoolExecutor,
          opponents=OPPONENTS) -> dict:
    """Average win-score of `params` across the opponent suite.

    Returns {"overall": float, "<opp>": float, ...}. seed0 shifts the seed
    block so different generations see different boards (avoids overfitting a
    fixed set); eval passes a held-out seed0."""
    tasks = {}
    for opp in opponents:
        for s in range(seeds):
            tasks[(opp, s)] = pool.submit(play, params, opp,
                                          seed0 + 101 * s + 9173 * len(opp))
    per_opp = {opp: 0.0 for opp in opponents}
    for (opp, _s), fut in tasks.items():
        per_opp[opp] += fut.result()
    out = {opp: per_opp[opp] / seeds for opp in opponents}
    out["overall"] = sum(per_opp.values()) / (seeds * len(opponents))
    return out


def default_sigma() -> list[float]:
    return [(hi - lo) * 0.20 for lo, hi in (PARAM_BOUNDS[n] for n in PARAM_NAMES)]


def train(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    mean = params_to_vec(DEFAULTS)
    sigma = default_sigma()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        best_p = dict(DEFAULTS)
        best = score(best_p, args.eval_seeds, args.seed, pool)
        best_s = best["overall"]
        print(f"baseline eval overall={best_s:.4f}  "
              + "  ".join(f"{o}={best[o]:.3f}" for o in OPPONENTS), flush=True)

        for gen in range(args.generations):
            gen_seed = args.seed + 100003 * (gen + 1)
            pop = []
            # Elitism: the current global best is always a population member,
            # re-scored on this generation's seeds for a fair comparison.
            elite_vec = params_to_vec(best_p)
            for vec in [elite_vec] + [
                [rng.gauss(m, s) for m, s in zip(mean, sigma)]
                for _ in range(args.population - 1)
            ]:
                p = vec_to_params(vec)
                sc = score(p, args.train_seeds, gen_seed, pool)
                pop.append((sc["overall"], p))
            pop.sort(key=lambda x: x[0], reverse=True)
            n_elite = max(2, args.population // 4)
            elites = pop[:n_elite]

            mean = [sum(params_to_vec(p)[i] for _s, p in elites) / n_elite
                    for i in range(len(PARAM_NAMES))]
            sigma = [
                max((PARAM_BOUNDS[PARAM_NAMES[i]][1]
                     - PARAM_BOUNDS[PARAM_NAMES[i]][0]) * 0.03,
                    math.sqrt(sum((params_to_vec(p)[i] - mean[i]) ** 2
                                  for _s, p in elites) / n_elite))
                for i in range(len(PARAM_NAMES))
            ]

            cand = elites[0][1]
            ev = score(cand, args.eval_seeds, args.seed, pool)
            mark = ""
            if ev["overall"] > best_s:
                best_s, best_p = ev["overall"], cand
                mark = "  <-- new best"
            print(f"gen {gen + 1:02d}  train_best={elites[0][0]:.4f}  "
                  f"eval={ev['overall']:.4f} ("
                  + " ".join(f"{o}={ev[o]:.2f}" for o in OPPONENTS) + ")"
                  + mark, flush=True)

    return best_p


def print_result(p: dict) -> None:
    print("\nBest weights:")
    for name in PARAM_NAMES:
        print(f"  {name:12s} = {p[name]:.6g}   (default {DEFAULTS[name]:g})")
    print("\nTo deploy, update the os.environ.get fallbacks in "
          "submission/snaky_greed.py:")
    env_to_var = {
        "SG_SPACE": "W_SPACE", "SG_VOR": "W_VOR", "SG_RCH": "W_RCH",
        "SG_STEP": "W_STEP", "SG_SURVPEN": "SURV_PEN",
        "SG_COLLIDE": "COLLIDE_PEN", "SG_PESSIMW": "PESSIM_W",
    }
    for name in PARAM_NAMES:
        var = env_to_var[name]
        print(f'  {var} = float(os.environ.get("{name}", "{p[name]:.6g}"))')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=28)
    ap.add_argument("--train-seeds", type=int, default=12,
                    help="seeds per opponent when scoring a candidate")
    ap.add_argument("--eval-seeds", type=int, default=30,
                    help="seeds per opponent for the held-out eval")
    ap.add_argument("--workers", type=int, default=7,
                    help="parallel games (each game = 2 subprocesses)")
    ap.add_argument("--seed", type=int, default=20260517)
    args = ap.parse_args()

    if args.eval_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            ev = score(dict(DEFAULTS), args.eval_seeds, args.seed, pool)
        print("shipped defaults  overall={:.4f}  ".format(ev["overall"])
              + "  ".join(f"{o}={ev[o]:.3f}" for o in OPPONENTS))
        return

    best = train(args)
    print_result(best)


if __name__ == "__main__":
    main()
