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
    # Defaults are the canonical values from struct Params in icarus_game.cpp
    # (CEM-tuned set, eval: 598/768 outright wins).
    no_info_floor: float = 0.0171807
    recency_decay: float = 0.624394
    tight_sd_mul: float = 1.29265
    tight_sd_add: float = 0.280066
    trust_base: float = 0.05
    trust_sd_mul: float = 0.462153
    trust_n_mul: float = 7.00283
    trend_mul: float = 0.508882
    block_up: float = 8.88151
    block_down: float = 1.15079
    block_cap: float = 66.0955
    finish_safe: float = 0.697257
    setup_safe: float = 0.691082
    # Tight-kernel (best-responder pack) branch.
    tight_sd_thresh: float = 1.5
    tight_sigma: float = 0.2
    tight_w_boost: float = 0.0845868
    # Shield / pack-riding thresholds.
    shield_safe1: float = 0.661246
    shield_evfrac1: float = 0.981208
    shield_safe2: float = 0.793981
    shield_safe3: float = 0.917121
    shield_evfrac3: float = 0.886851
    match_safe: float = 0.545142
    match_evfrac: float = 0.859635
    block_shift_cap: float = 19.5093


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
    "tight_sd_thresh": (1.50, 7.00),
    "tight_sigma": (0.20, 3.00),
    "tight_w_boost": (0.00, 0.45),
    "shield_safe1": (0.55, 0.92),
    "shield_evfrac1": (0.70, 1.05),
    "shield_safe2": (0.60, 0.95),
    "shield_safe3": (0.60, 0.95),
    "shield_evfrac3": (0.65, 1.05),
    "match_safe": (0.35, 0.80),
    "match_evfrac": (0.65, 1.00),
    "block_shift_cap": (6.00, 40.00),
}

PARAM_NAMES = list(PARAM_BOUNDS)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def clamp_bid(x: int) -> int:
    return 1 if x < 1 else 100 if x > 100 else x


def war_cap(top_hist: list[int], pack: bool) -> int:
    """Safe bid ceiling during a descending bid war (else a no-op high cap).

    top_hist is the field's top bid per round. When opponents form a pack and
    that top is trending down, predict next round's top by extrapolating the
    recent average drop and cap our bid 2 below it -- so we never tie the new
    pack top (which is what gets a player blocked).
    """
    if not pack or len(top_hist) < 3:
        return 1000
    w = top_hist[-3:]
    # Require a *consistent* descent (monotonic non-increasing). Noisy
    # opponents rarely drop their top several rounds straight, so this keeps
    # the guard from mis-firing outside a genuine descending war.
    if any(w[k] < w[k + 1] for k in range(len(w) - 1)):
        return 1000
    drop = (w[0] - w[-1]) / (len(w) - 1)
    if drop < 1.0:
        return 1000
    drop = min(drop, 12.0)
    cap = int(w[-1] - drop - 2.0)
    return cap if cap >= 1 else 1


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
        self.top_hist = []

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

            # Tight recent cluster -> reliable shield (best-responders).
            tight_branch = sd < self.p.tight_sd_thresh
            if tight_branch:
                sig_t = self.p.tight_sigma
            else:
                sig_t = clamp(self.p.tight_sd_mul * sd + self.p.tight_sd_add, 0.35, 9.0)
            sig_w = max(12.0, 2.2 * sig_t)
            tight = kernel_dist(h, sig_t, self.p.recency_decay)
            wide = kernel_dist(h, sig_w, self.p.recency_decay)

            k_trust = clamp(
                self.p.trust_base + self.p.trust_sd_mul * sd + self.p.trust_n_mul / n,
                0.25,
                9.0,
            )
            w = n / (n + k_trust)
            if tight_branch:
                w = min(0.92, w + self.p.tight_w_boost)
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

        # Blend in a finishing distribution only when the opponent's recent
        # bid shows it will actually bid that high -- a spiralled-down crawler
        # keeps crawling, and modelling it as a finisher makes us over-bid.
        dist = TARGET - opp_pos
        if 1 <= dist <= 100 and (not h or h[-1] >= dist - 4):
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
            last_bid = self.hist[i][-1] if self.hist[i] else 0
            opps.append((cum[i], TARGET - pos[i], cdf, last_bid))

        # Track the field's top bid each round. In a descending bid war the
        # cautious players who stay clear of this top almost never get blocked
        # (observed: P2/P3), so the war guard below shades us under it.
        cur_top = max((lb for _c, _d, _cdf, lb in opps if lb >= 1), default=0)
        if cur_top >= 1:
            self.top_hist.append(cur_top)
            if len(self.top_hist) > 12:
                del self.top_hist[0]

        if not opps:
            bid = 1
        else:
            def cdf_at(cdf: list[float], idx: int) -> float:
                if idx <= 0:
                    return 0.0
                if idx >= 100:
                    return 1.0
                return cdf[idx]

            def detect_pack() -> bool:
                if len(opps) < 2:
                    return False
                recent = [lb for _c, _d, _cdf, lb in opps if lb >= 1]
                if len(recent) < 2:
                    return False
                return (max(recent) - min(recent)) <= 10

            def recent_ceil() -> int:
                bids = sorted((lb for _c, _d, _cdf, lb in opps if lb >= 1), reverse=True)
                if not bids:
                    return 0
                return bids[1] if len(bids) >= 2 else bids[0]

            ceil_bid = recent_ceil()

            def p_safe(b: int) -> float:
                shift = round(self.block_bias)
                if shift > self.p.block_shift_cap:
                    shift = int(self.p.block_shift_cap)
                if ceil_bid >= 8 and b + 3 <= ceil_bid:
                    shift = min(shift, max(0, ceil_bid - b - 2))
                hi = b + shift
                if hi > 100:
                    hi = 100
                p_blocked = 1.0
                for ocum, _odist, cdf, _lb in opps:
                    p_blocked *= cdf_at(cdf, hi - 1 if ocum < my_cum else hi)
                return 1.0 - p_blocked

            if 1 <= my_dist <= 100:
                # A bigger finishing bid only helps when a rival is ALSO
                # finishing this round (overshoot wins the position
                # tie-break). If nobody else can finish, bid the minimal
                # safe finish -- a b*ps over-bid would stick out above the
                # pack and get us blocked while leading.
                opp_finishing = any(
                    odist <= 100 and lb >= odist - 4
                    for _c, odist, _cdf, lb in opps
                )
                b_fin = my_dist
                ps_fin = -1.0
                fin_ev = -1.0
                for b in range(my_dist, 101):
                    ps = p_safe(b)
                    ev = (b * ps) if opp_finishing else ps
                    if ev > fin_ev or (ev == fin_ev and ps > ps_fin):
                        fin_ev = ev
                        ps_fin = ps
                        b_fin = b
                b_set = 0
                best_ev = -1.0
                for b in range(1, my_dist):
                    ev = b * p_safe(b)
                    if ev > best_ev:
                        best_ev = ev
                        b_set = b

                # A rival is "urgently finishing" only if it is at/ahead of us,
                # within one bid of the line, AND its recent bid is actually
                # large enough to finish. A rival that can reach the line but
                # is bidding far below its distance (a spiralled-down pack) will
                # not finish next round -- panicking and throwing a doomed
                # finishing bid above the pack just gets us blocked.
                urgent_finish = any(
                    odist <= 100 and TARGET - odist >= pos[0] and lb >= odist - 4
                    for _c, odist, _cdf, lb in opps
                )
                if ps_fin >= self.p.finish_safe or urgent_finish:
                    bid = b_fin
                elif b_set > 0 and p_safe(b_set) >= self.p.setup_safe:
                    bid = b_set
                else:
                    # EV-best setup is too risky -- don't fall back to the
                    # doomed full-distance b_fin (a certain block that freezes
                    # us). Step down to the safest setup bid available.
                    safe_set = 0
                    for b in range(1, my_dist):
                        if p_safe(b) >= self.p.setup_safe:
                            safe_set = b
                    bid = safe_set if safe_set > 0 else (b_set if b_set > 0 else b_fin)
            else:
                best = 1
                best_ev = -1.0
                for b in range(1, 101):
                    ev = b * p_safe(b)
                    if ev > best_ev:
                        best_ev = ev
                        best = b

                pack = detect_pack() or self.block_bias >= 8.0
                bid = best
                if pack and ceil_bid >= 8:
                    shield = clamp_bid(ceil_bid - 1)
                    shield_ev = shield * p_safe(shield)
                    if (p_safe(shield) >= self.p.shield_safe1
                            and shield_ev >= best_ev * self.p.shield_evfrac1):
                        bid = shield
                    else:
                        for d in range(2, 5):
                            sb = clamp_bid(ceil_bid - d)
                            ev = sb * p_safe(sb)
                            if p_safe(sb) >= self.p.shield_safe2 and ev > shield_ev:
                                shield_ev = ev
                                shield = sb
                        if (p_safe(shield) >= self.p.shield_safe3
                                and shield_ev >= best_ev * self.p.shield_evfrac3):
                            bid = shield
                        else:
                            if best > ceil_bid:
                                best = clamp_bid(ceil_bid - 1)
                            bid = best
                if bid == best:
                    min_opp_cum = min(c for c, _d, _cdf, _lb in opps)
                    if my_cum <= min_opp_cum and pack and ceil_bid >= 5:
                        match = clamp_bid(ceil_bid)
                        if (p_safe(match) >= self.p.match_safe
                                and match * p_safe(match) > best_ev * self.p.match_evfrac):
                            bid = match

                # Descending-war guard: the EV/shield logic targets last
                # round's pack, but in a war the pack keeps dropping ~2-3/round,
                # so a bid "safely under" the stale top still ties the new top
                # and gets blocked. Extrapolate the descent and cap our bid a
                # margin under the predicted top.
                bid = min(bid, war_cap(self.top_hist, pack))

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


def crawl_bot(pos, _cum, i, rng, st):
    """Conservative descending crawler -- the dominant real-lobby archetype.

    Opens low (~44-58) and shaves 1-2 off its bid each round, then finishes
    with the exact remaining distance once in range. Mirrors observed real
    matches (openers 44-50, smooth descent into a ~20s crawl, jump-finish at
    the end). The training suites otherwise only had aggressive openers
    (smart 68-88, aggressive 82-99), which mistuned our round-0 bid.
    """
    dist = TARGET - pos[i]
    if 1 <= dist <= 100:
        return dist
    b = st.get("cb")
    if b is None:
        b = rng.randint(44, 58)
    else:
        b = max(1, b - rng.randint(1, 2))
    st["cb"] = b
    return b


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


def selfplay_bot(pos, cum, i, rng, st):
    """Self-play: a LearnedBot mirror, seeing itself as player 0.

    Stresses the bot against (near-)mirrors of its own strategy -- a strong
    adversary that exposes any exploitable bias the EV/shield logic has. The
    mirror's params are jittered slightly so the deterministic bots do not
    bid bit-identically every round (which would make every player tie and
    yield zero gradient signal).
    """
    bot = st.get("bot")
    if bot is None:
        base = st["params"]
        v = params_to_vec(base)
        v = [x * (1.0 + 0.06 * (rng.random() - 0.5)) for x in v]
        bot = LearnedBot(vec_to_params(v))
        st["bot"] = bot
    order = [i] + [j for j in range(4) if j != i]
    rp = [pos[j] for j in order]
    rc = [cum[j] for j in order]
    return bot.choose(rp, rc)


def shadow_bot(_pos, cum, _i, rng, st):
    """Shadow opponent: bids myBid + 1 to always sit just above our bot.

    Directly attacks the blockBias recovery logic -- if the bot ever bids
    high this opponent out-bids it by 1, forcing a block; the bot must drive
    its bid down and the shadow must keep failing to block it.
    """
    last = st.get("last_cum")
    my_bid = 70
    if last is not None:
        d = cum[0] - last[0]
        if 1 <= d <= 100:
            my_bid = d
    st["last_cum"] = list(cum)
    return clamp_bid(my_bid + 1)


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
    "self-play": [selfplay_bot, selfplay_bot, selfplay_bot],
    "self-mix": [selfplay_bot, br_bot, aggressive_bot],
    "shadow": [shadow_bot, smart_bot, cautious_bot],
    "shadow-pack": [shadow_bot, br_bot, br_bot],
    # Conservative low-opener crawlers -- the real-lobby archetype.
    "all-crawl": [crawl_bot, crawl_bot, crawl_bot],
    "crawl-mix": [crawl_bot, crawl_bot, br_bot],
    "crawl-aggr": [crawl_bot, crawl_bot, aggressive_bot],
}


def run_match(params: Params, opps: list[Strategy], seed: int) -> int:
    rng = random.Random(seed)
    bot = LearnedBot(params)
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    dead = [False] * 4
    # Self-play opponents need the parameter set to spin up their own bot.
    state = [{"params": params} for _ in range(4)]
    # Round cap 110: real matches can degenerate into a long "crawl" (everyone
    # bids 1, advancing +1/round) that runs ~100 rounds. Training must see the
    # full crawl so tuned params bank enough position before it sets in.
    for _rnd in range(110):
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


def selected_suites(names: str | None) -> dict[str, list[Strategy]]:
    if not names:
        return SUITES
    picked = {}
    for name in names.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in SUITES:
            raise SystemExit(f"unknown suite {name!r}; choices: {', '.join(SUITES)}")
        picked[name] = SUITES[name]
    if not picked:
        raise SystemExit("--suites did not select any suites")
    return picked


def score(params: Params, seeds_per_suite: int, seed0: int,
          suites: dict[str, list[Strategy]] | None = None) -> float:
    suites = suites or SUITES
    total = 0
    count = 0
    for name, opps in suites.items():
        # Put extra weight on the weak spots from previous testing and on the
        # adversarial stress suites (self-play / shadow).
        weight = 2 if name in {"all-bestresp", "br-mix", "br-snipe", "all-30",
                               "self-play", "self-mix", "shadow",
                               "shadow-pack", "all-crawl", "crawl-mix",
                               "crawl-aggr"} else 1
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
    suites = selected_suites(args.suites)
    mean = params_to_vec(Params())
    sigma = default_sigma()
    best_p = Params()
    best_s = score(best_p, args.eval_seeds, args.seed, suites)
    print(f"baseline score={best_s:.4f}")

    for gen in range(args.generations):
        pop = []
        # Elitism: inject the global best as a guaranteed population member so
        # CEM cannot drift away from a known-good solution between generations.
        gen_seed = args.seed + 100000 * (gen + 1)
        pop.append((score(best_p, args.train_seeds, gen_seed, suites), best_p))
        for _ in range(args.population - 1):
            v = [rng.gauss(m, s) for m, s in zip(mean, sigma)]
            p = vec_to_params(v)
            sc = score(p, args.train_seeds, gen_seed, suites)
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
        candidate_eval = score(candidate, args.eval_seeds, args.seed + 777, suites)
        if candidate_eval > best_s:
            best_s = candidate_eval
            best_p = candidate
        print(
            f"gen={gen + 1:02d} train_best={elites[0][0]:.4f} "
            f"eval_best={candidate_eval:.4f} global={best_s:.4f}"
        )

    return best_p


def eval_suites(params: Params, seeds_per_suite: int, seed0: int,
                suites: dict[str, list[Strategy]] | None = None) -> None:
    suites = suites or SUITES
    grand = 0
    grand_n = 0
    for name, opps in suites.items():
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
    # The C++ submission, icarus_game.py and this LearnedBot all read the SAME
    # parameter set. To deploy a tuned result, paste the block below over
    # `struct Params` in icarus_game.cpp (and the PARAMS dict in icarus_game.py).
    print("\nPaste into icarus_game.cpp -> struct Params { ... }:")
    for name in PARAM_NAMES:
        print(f"    double {name:<16s} = {getattr(p, name):.6g};")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--suites", help="comma-separated suite names to train/evaluate")
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--train-seeds", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260516)
    args = ap.parse_args()

    if args.eval_only:
        eval_suites(Params(), args.eval_seeds, args.seed, selected_suites(args.suites))
    else:
        best = train(args)
        print_params(best)


if __name__ == "__main__":
    main()
