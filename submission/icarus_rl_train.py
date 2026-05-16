"""Offline RL-style tuner for Icarus Game.

This does not replace the submitted bot. It trains/evaluates parameters for the
current opponent-model + expected-value policy, then prints constants that can be
ported back to icarus_game.cpp.

Algorithm: Cross-Entropy Method (CEM), a simple black-box policy search method.
It is a good fit here because the action space is small, the simulator is fast,
and the final submitted policy must be deterministic C++ constants.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable


TARGET = 999


@dataclass
class Params:
    no_info_floor: float = 0.01
    recency_decay: float = 0.70
    tight_sd_mul: float = 0.85
    tight_sd_add: float = 0.35
    trust_base: float = 0.30
    trust_sd_mul: float = 0.33
    trust_n_mul: float = 4.50
    trend_mul: float = 1.50
    block_up: float = 9.0
    block_down: float = 3.0
    block_cap: float = 48.0
    finish_safe: float = 0.80
    setup_safe: float = 0.70


PARAM_BOUNDS = {
    "no_info_floor": (0.000, 0.060),
    "recency_decay": (0.45, 0.90),
    "tight_sd_mul": (0.30, 1.60),
    "tight_sd_add": (0.10, 1.80),
    "trust_base": (0.05, 1.20),
    "trust_sd_mul": (0.05, 0.90),
    "trust_n_mul": (1.00, 10.00),
    "trend_mul": (0.00, 3.50),
    "block_up": (2.00, 18.00),
    "block_down": (0.50, 8.00),
    "block_cap": (12.00, 70.00),
    "finish_safe": (0.50, 0.95),
    "setup_safe": (0.45, 0.95),
}

PARAM_NAMES = list(PARAM_BOUNDS)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def clamp_bid(x: int) -> int:
    return 1 if x < 1 else 100 if x > 100 else x


def normalize(d: list[float]) -> list[float]:
    s = sum(d[1:])
    if s <= 0.0:
        return [0.0] + [0.01] * 100
    return [0.0] + [v / s for v in d[1:]]


def kernel_dist(hist: list[int], sigma: float, recency_decay: float) -> list[float]:
    d = [0.0] * 101
    n = len(hist)
    sigma = max(0.05, sigma)
    denom = 2.0 * sigma * sigma
    for k, v in enumerate(hist):
        recw = recency_decay ** (n - 1 - k)
        for b in range(1, 101):
            dd = b - v
            d[b] += recw * math.exp(-(dd * dd) / denom)
    return normalize(d)


class LearnedBot:
    def __init__(self, params: Params):
        self.p = params
        self.hist = [[] for _ in range(4)]
        self.alive = [True] * 4
        self.zero_streak = [0] * 4
        self.have_last = False
        self.last_pos = [0] * 4
        self.last_cum = [0] * 4
        self.block_bias = 0.0

    def update_memory(self, pos: list[int], cum: list[int]) -> None:
        if not self.have_last:
            return
        for i in range(4):
            if not self.alive[i]:
                continue
            dc = cum[i] - self.last_cum[i]
            if dc < 0 or (pos[i] == 0 and cum[i] == 0 and self.last_cum[i] > 0):
                self.alive[i] = False
                self.hist[i].clear()
                continue
            if dc == 0:
                self.zero_streak[i] += 1
                if self.zero_streak[i] >= 2:
                    self.alive[i] = False
                    self.hist[i].clear()
                continue
            self.zero_streak[i] = 0
            if 1 <= dc <= 100:
                self.hist[i].append(dc)
                if len(self.hist[i]) > 32:
                    del self.hist[i][0]

    def opp_dist(self, i: int, opp_pos: int) -> list[float]:
        h = self.hist[i]
        n = len(h)
        if n == 0:
            p = [0.0] + [0.01] * 100
        else:
            start = max(0, n - 6)
            recent = h[start:]
            mean = sum(recent) / len(recent)
            sd = math.sqrt(sum((x - mean) ** 2 for x in recent) / len(recent))

            sig_t = clamp(self.p.tight_sd_mul * sd + self.p.tight_sd_add, 0.35, 9.0)
            sig_w = max(15.0, 2.5 * sig_t)
            tight = kernel_dist(h, sig_t, self.p.recency_decay)
            wide = kernel_dist(h, sig_w, self.p.recency_decay)

            k_trust = clamp(
                self.p.trust_base + self.p.trust_sd_mul * sd + self.p.trust_n_mul / n,
                0.3,
                9.0,
            )
            w = n / (n + k_trust)
            p = [0.0] + [w * tight[b] + (1.0 - w) * wide[b] for b in range(1, 101)]

            if n >= 4:
                length = min(n, 8)
                half = length // 2
                m1 = sum(h[n - length : n - length + half]) / half
                m2 = sum(h[n - half : n]) / half
                shift = clamp(self.p.trend_mul * ((m2 - m1) / half), -9.0, 9.0)

                recent4 = h[n - min(n, 4) :]
                rmean = sum(recent4) / len(recent4)
                rspread = sum(abs(x - rmean) for x in recent4) / len(recent4)
                if rspread < 2.0:
                    shift = 0.0

                if abs(shift) > 0.3:
                    sh = [0.0] * 101
                    for b in range(1, 101):
                        src = b - shift
                        lo = math.floor(src)
                        fr = src - lo
                        a = clamp_bid(int(lo))
                        bb = clamp_bid(int(lo + 1))
                        sh[b] = p[a] * (1.0 - fr) + p[bb] * fr
                    p = normalize(sh)

        floor_w = self.p.no_info_floor
        p = [0.0] + [(1.0 - floor_w) * p[b] + floor_w / 100.0 for b in range(1, 101)]

        dist = TARGET - opp_pos
        if 1 <= dist <= 100:
            fin = [0.0] * 101
            fs = 0.0
            for b in range(dist, 101):
                fin[b] = math.exp(-(b - dist) / 22.0)
                fs += fin[b]
            if fs > 0.0:
                fin = [0.0] + [fin[b] / fs for b in range(1, 101)]
                p = [0.0] + [0.5 * p[b] + 0.5 * fin[b] for b in range(1, 101)]
        return p

    def choose(self, pos: list[int], cum: list[int]) -> int:
        if self.have_last:
            my_bid = cum[0] - self.last_cum[0]
            blocked = pos[0] == self.last_pos[0] and my_bid > 0
            if blocked:
                self.block_bias = min(self.p.block_cap, self.block_bias + self.p.block_up)
            else:
                self.block_bias = max(0.0, self.block_bias - self.p.block_down)

        self.update_memory(pos, cum)
        my_cum = cum[0]
        my_dist = TARGET - pos[0]
        opps = []
        for i in range(1, 4):
            if not self.alive[i]:
                continue
            d = self.opp_dist(i, pos[i])
            cdf = [0.0] * 101
            for b in range(1, 101):
                cdf[b] = cdf[b - 1] + d[b]
            opps.append((cum[i], TARGET - pos[i], cdf))

        if not opps:
            bid = 1
        else:
            def cdf_at(cdf: list[float], idx: int) -> float:
                if idx <= 0:
                    return 0.0
                if idx >= 100:
                    return 1.0
                return cdf[idx]

            def p_safe(b: int) -> float:
                hi = b + round(self.block_bias)
                p_blocked = 1.0
                for ocum, _odist, cdf in opps:
                    p_blocked *= cdf_at(cdf, hi - 1 if ocum < my_cum else hi)
                return 1.0 - p_blocked

            if 1 <= my_dist <= 100:
                b_fin = my_dist
                ps_fin = -1.0
                for b in range(my_dist, 101):
                    ps = p_safe(b)
                    if ps > ps_fin:
                        ps_fin = ps
                        b_fin = b
                if ps_fin >= self.p.finish_safe or any(odist <= 100 for _c, odist, _cdf in opps):
                    bid = b_fin
                else:
                    b_set = 0
                    best_ev = -1.0
                    for b in range(1, my_dist):
                        ev = b * p_safe(b)
                        if ev > best_ev:
                            best_ev = ev
                            b_set = b
                    bid = b_set if b_set > 0 and p_safe(b_set) >= self.p.setup_safe else b_fin
            else:
                best = 1
                best_ev = -1.0
                for b in range(1, 101):
                    ev = b * p_safe(b)
                    if ev > best_ev:
                        best_ev = ev
                        best = b
                bid = best

        self.last_pos = pos[:]
        self.last_cum = cum[:]
        self.have_last = True
        return clamp_bid(int(bid))


Strategy = Callable[[list[int], list[int], int, random.Random, dict], int]


def uniform_bot(_pos, _cum, _i, rng, _st):
    return rng.randint(1, 100)


def constant_bot(v: int) -> Strategy:
    def f(_pos, _cum, _i, _rng, _st):
        return v
    return f


def smart_bot(pos, _cum, i, rng, _st):
    dist = TARGET - pos[i]
    return dist if dist <= 100 else rng.randint(68, 88)


def aggressive_bot(pos, _cum, i, rng, _st):
    dist = TARGET - pos[i]
    return dist if dist <= 100 else rng.randint(82, 99)


def cautious_bot(pos, _cum, i, rng, _st):
    dist = TARGET - pos[i]
    return dist if dist <= 100 else rng.randint(40, 60)


def snipe_bot(pos, _cum, i, _rng, _st):
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
    others = [pos[j] for j in range(4) if j != i]
    return max(30, min(95, 70 + (max(others) - pos[i]) // 20))


def br_bot(_pos, cum, i, rng, st):
    hist = st.setdefault("h", [[], [], [], []])
    last = st.get("last_cum")
    if last is not None:
        for j in range(4):
            d = cum[j] - last[j]
            if 1 <= d <= 100:
                hist[j].append(d)
    st["last_cum"] = list(cum)
    preds = []
    for j in range(4):
        if j == i:
            continue
        h = hist[j][-8:]
        preds.append(sum(h) / len(h) if h else 70)
    preds.sort()
    return max(30, min(98, int(preds[-1]) - rng.randint(1, 4)))


SUITES: dict[str, list[Strategy]] = {
    "all-random": [uniform_bot, uniform_bot, uniform_bot],
    "all-smart": [smart_bot, smart_bot, smart_bot],
    "all-aggressive": [aggressive_bot, aggressive_bot, aggressive_bot],
    "all-cautious": [cautious_bot, cautious_bot, cautious_bot],
    "all-30": [constant_bot(30), constant_bot(30), constant_bot(30)],
    "constants": [constant_bot(70), constant_bot(85), constant_bot(55)],
    "mixed": [smart_bot, aggressive_bot, cautious_bot],
    "mixed2": [uniform_bot, smart_bot, snipe_bot],
    "two-aggressive": [aggressive_bot, aggressive_bot, cautious_bot],
    "all-bestresp": [br_bot, br_bot, br_bot],
    "br-mix": [br_bot, smart_bot, aggressive_bot],
    "br-snipe": [br_bot, snipe_bot, uniform_bot],
}


def run_match(params: Params, opps: list[Strategy], seed: int) -> int:
    rng = random.Random(seed)
    bot = LearnedBot(params)
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    dead = [False] * 4
    state = [{} for _ in range(4)]
    for _rnd in range(60):
        bids = [0, 0, 0, 0]
        bids[0] = bot.choose(pos, cum)
        for i in range(1, 4):
            if not dead[i]:
                bids[i] = clamp_bid(int(opps[i - 1](pos, cum, i, rng, state[i])))

        active = [i for i in range(4) if not dead[i]]
        for i in active:
            cum[i] += bids[i]
        hi = max(bids[i] for i in active)
        tied = [i for i in active if bids[i] == hi]
        low_cum = min(cum[i] for i in tied)
        blocked = {i for i in tied if cum[i] == low_cum}
        for i in active:
            if i not in blocked:
                pos[i] += bids[i]
        if any(pos[i] >= TARGET for i in active):
            break

    order = sorted(range(4), key=lambda i: -pos[i])
    pts = [0, 0, 0, 0]
    vals = [pos[o] for o in order]
    j = 0
    while j < 4:
        k = j
        while k + 1 < 4 and vals[k + 1] == vals[j]:
            k += 1
        for t in range(j, k + 1):
            pts[order[t]] = [3, 2, 1, 0][k]
        j = k + 1
    return pts[0]


def score(params: Params, seeds_per_suite: int, seed0: int) -> float:
    total = 0
    count = 0
    for name, opps in SUITES.items():
        # Put extra weight on the weak spots from previous testing.
        weight = 2 if name in {"all-bestresp", "br-mix", "br-snipe", "all-30"} else 1
        for s in range(seeds_per_suite):
            pts = run_match(params, opps, seed0 + 1009 * s + 37 * len(name))
            total += weight * pts
            count += weight
    return total / count


def params_to_vec(p: Params) -> list[float]:
    return [getattr(p, name) for name in PARAM_NAMES]


def vec_to_params(v: list[float]) -> Params:
    kwargs = {}
    for name, x in zip(PARAM_NAMES, v):
        lo, hi = PARAM_BOUNDS[name]
        kwargs[name] = clamp(x, lo, hi)
    return Params(**kwargs)


def default_sigma() -> list[float]:
    sig = []
    for name in PARAM_NAMES:
        lo, hi = PARAM_BOUNDS[name]
        sig.append((hi - lo) * 0.18)
    return sig


def train(args: argparse.Namespace) -> Params:
    rng = random.Random(args.seed)
    mean = params_to_vec(Params())
    sigma = default_sigma()
    best_p = Params()
    best_s = score(best_p, args.eval_seeds, args.seed)
    print(f"baseline score={best_s:.4f}")

    for gen in range(args.generations):
        pop = []
        for _ in range(args.population):
            v = [rng.gauss(m, s) for m, s in zip(mean, sigma)]
            p = vec_to_params(v)
            sc = score(p, args.train_seeds, args.seed + 100000 * (gen + 1))
            pop.append((sc, p))
        pop.sort(key=lambda x: x[0], reverse=True)
        elites = pop[: max(2, args.population // 5)]
        mean = [
            sum(params_to_vec(p)[i] for _sc, p in elites) / len(elites)
            for i in range(len(PARAM_NAMES))
        ]
        sigma = [
            max((PARAM_BOUNDS[name][1] - PARAM_BOUNDS[name][0]) * 0.025,
                math.sqrt(sum((params_to_vec(p)[i] - mean[i]) ** 2 for _sc, p in elites) / len(elites)))
            for i, name in enumerate(PARAM_NAMES)
        ]

        candidate = elites[0][1]
        candidate_eval = score(candidate, args.eval_seeds, args.seed + 777)
        if candidate_eval > best_s:
            best_s = candidate_eval
            best_p = candidate
        print(
            f"gen={gen + 1:02d} train_best={elites[0][0]:.4f} "
            f"eval_best={candidate_eval:.4f} global={best_s:.4f}"
        )

    return best_p


def eval_suites(params: Params, seeds_per_suite: int, seed0: int) -> None:
    grand = 0
    grand_n = 0
    for name, opps in SUITES.items():
        total = 0
        wins = 0
        for s in range(seeds_per_suite):
            pts = run_match(params, opps, seed0 + 1009 * s + 37 * len(name))
            total += pts
            wins += pts == 3
        grand += total
        grand_n += seeds_per_suite
        print(
            f"{name:16s}  avg_pts={total / seeds_per_suite:.3f}  "
            f"outright_wins={wins}/{seeds_per_suite} ({100 * wins / seeds_per_suite:.0f}%)"
        )
    print(f"{'OVERALL':16s}  avg_pts={grand / grand_n:.3f}  (3=win, 0=last)")


def print_params(p: Params) -> None:
    print("\nBest parameters:")
    for name in PARAM_NAMES:
        print(f"  {name:15s} = {getattr(p, name):.6g}")
    print("\nC++ constant edits:")
    print(f"  recency decay: pow({p.recency_decay:.6g}, n - 1 - k)")
    print(f"  floorW = {p.no_info_floor:.6g};")
    print(f"  sigT = clamp({p.tight_sd_mul:.6g} * sd + {p.tight_sd_add:.6g}, 0.35, 9.0);")
    print(
        "  kTrust = clamp("
        f"{p.trust_base:.6g} + {p.trust_sd_mul:.6g} * sd + {p.trust_n_mul:.6g} / n, 0.3, 9.0);"
    )
    print(f"  shift = clamp({p.trend_mul:.6g} * slope, -9.0, 9.0);")
    print(f"  if (psFin >= {p.finish_safe:.6g}) return bFin;")
    print(f"  if (bSet > 0 && pSafe(bSet) >= {p.setup_safe:.6g}) return bSet;")
    print(
        f"  if (blocked) blockBias = min({p.block_cap:.6g}, blockBias + {p.block_up:.6g}); "
        f"else blockBias = max(0.0, blockBias - {p.block_down:.6g});"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260516)
    args = ap.parse_args()

    if args.eval_only:
        eval_suites(Params(), args.eval_seeds, args.seed)
    else:
        best = train(args)
        print_params(best)


if __name__ == "__main__":
    main()
