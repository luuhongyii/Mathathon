"""Icarus Game submission.

Python port of icarus_game.cpp. It uses the same deterministic opponent model:
learn bid distributions from cumulative deltas, compute the exact probability
of being blocked for each bid, and choose the bid with the best expected gain.

This file is a FAITHFUL MIRROR of icarus_game.cpp -- same parameter set, same
tight-kernel / detectPack / shield logic. The PARAMS dict below is the single
source of truth shared (by value) with the C++ submission and the LearnedBot
in icarus_rl_train.py. Keep all three in lockstep.
"""

import math


TARGET = 999

# Tunable strategy parameters -- mirror of struct Params in icarus_game.cpp.
PARAMS = {
    # CEM-tuned set (eval: 598/768 outright wins). Mirror of struct Params.
    "no_info_floor": 0.0171807,
    "recency_decay": 0.624394,
    "tight_sd_mul": 1.29265,
    "tight_sd_add": 0.280066,
    "trust_base": 0.05,
    "trust_sd_mul": 0.462153,
    "trust_n_mul": 7.00283,
    "trend_mul": 0.508882,
    "block_up": 8.88151,
    "block_down": 1.15079,
    "block_cap": 66.0955,
    "finish_safe": 0.697257,
    "setup_safe": 0.691082,
    "tight_sd_thresh": 1.5,
    "tight_sigma": 0.2,
    "tight_w_boost": 0.0845868,
    "shield_safe1": 0.661246,
    "shield_evfrac1": 0.981208,
    "shield_safe2": 0.793981,
    "shield_safe3": 0.917121,
    "shield_evfrac3": 0.886851,
    "match_safe": 0.545142,
    "match_evfrac": 0.859635,
    "block_shift_cap": 19.5093,
}

bid_hist = [[] for _ in range(4)]
alive = [True, True, True, True]
zero_streak = [0, 0, 0, 0]
last_pos = [0, 0, 0, 0]
last_cum = [0, 0, 0, 0]
have_last = False
block_bias = 0.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_bid(x):
    return 1 if x < 1 else 100 if x > 100 else int(x)


def war_cap(top_hist, pack):
    """Safe bid ceiling during a descending bid war (else a no-op high cap).

    top_hist is the field's top bid per round. When opponents form a pack and
    that top has descended several rounds straight, the EV/shield logic still
    targets last round's (stale, higher) pack -- so a bid "safely under" it
    ties the new, lower top and gets blocked. Extrapolate the descent and cap
    our bid a margin under the predicted next top. The monotonic check keeps
    the guard from mis-firing on noisy opponents that are not in a war.
    """
    if not pack or len(top_hist) < 3:
        return 1000
    w = top_hist[-3:]
    if any(w[k] < w[k + 1] for k in range(len(w) - 1)):
        return 1000
    drop = (w[0] - w[-1]) / (len(w) - 1)
    if drop < 1.0:
        return 1000
    drop = min(drop, 12.0)
    cap = int(w[-1] - drop - 2.0)
    return cap if cap >= 1 else 1


top_hist = []


def normalize(a):
    s = sum(a[1:])
    if s <= 0.0:
        for b in range(1, 101):
            a[b] = 1.0 / 100.0
    else:
        for b in range(1, 101):
            a[b] /= s


def update_memory(pos, cum):
    if not have_last:
        return

    for i in range(4):
        if not alive[i]:
            continue

        dc = cum[i] - last_cum[i]
        if dc < 0 or (pos[i] == 0 and cum[i] == 0 and last_cum[i] > 0):
            alive[i] = False
            bid_hist[i].clear()
            continue

        if dc == 0:
            zero_streak[i] += 1
            if zero_streak[i] >= 2:
                alive[i] = False
                bid_hist[i].clear()
            continue

        zero_streak[i] = 0
        if 1 <= dc <= 100:
            bid_hist[i].append(dc)
            if len(bid_hist[i]) > 32:
                del bid_hist[i][0]


def kernel_dist(hist, sigma):
    d = [0.0] * 101
    sigma = max(0.05, sigma)
    denom = 2.0 * sigma * sigma
    n = len(hist)
    for k, v in enumerate(hist):
        recw = PARAMS["recency_decay"] ** (n - 1 - k)
        for b in range(1, 101):
            dd = b - v
            d[b] += recw * math.exp(-(dd * dd) / denom)
    normalize(d)
    return d


def opp_dist(i, opp_pos):
    hist = bid_hist[i]
    n = len(hist)
    p = [0.0] * 101

    if n == 0:
        for b in range(1, 101):
            p[b] = 1.0 / 100.0
    else:
        recent = hist[max(0, n - 6):]
        mean = sum(recent) / len(recent)
        sd = math.sqrt(sum((x - mean) * (x - mean) for x in recent) / len(recent))

        # Tight recent cluster -> treat as a reliable shield (best-responders).
        tight_branch = sd < PARAMS["tight_sd_thresh"]
        if tight_branch:
            sig_t = PARAMS["tight_sigma"]
        else:
            sig_t = clamp(PARAMS["tight_sd_mul"] * sd + PARAMS["tight_sd_add"], 0.35, 9.0)
        sig_w = max(12.0, 2.2 * sig_t)
        tight = kernel_dist(hist, sig_t)
        wide = kernel_dist(hist, sig_w)

        k_trust = clamp(
            PARAMS["trust_base"] + PARAMS["trust_sd_mul"] * sd + PARAMS["trust_n_mul"] / n,
            0.25, 9.0,
        )
        w = n / (n + k_trust)
        if tight_branch:
            w = min(0.92, w + PARAMS["tight_w_boost"])
        for b in range(1, 101):
            p[b] = w * tight[b] + (1.0 - w) * wide[b]

        if n >= 4:
            length = min(n, 8)
            half = length // 2
            m1 = sum(hist[n - length:n - length + half]) / half
            m2 = sum(hist[n - half:n]) / half
            slope = (m2 - m1) / half
            shift = clamp(PARAMS["trend_mul"] * slope, -9.0, 9.0)

            recent4 = hist[n - min(n, 4):]
            rmean = sum(recent4) / len(recent4)
            rspread = sum(abs(x - rmean) for x in recent4) / len(recent4)
            if rspread < 2.0:
                shift = 0.0

            if abs(shift) > 0.3:
                shifted = [0.0] * 101
                for b in range(1, 101):
                    src = b - shift
                    lo = math.floor(src)
                    fr = src - lo
                    a = clamp_bid(lo)
                    bb = clamp_bid(lo + 1)
                    shifted[b] = p[a] * (1.0 - fr) + p[bb] * fr
                normalize(shifted)
                p = shifted

    floor_w = PARAMS["no_info_floor"]
    for b in range(1, 101):
        p[b] = (1.0 - floor_w) * p[b] + floor_w / 100.0

    # Blend in a "finishing" distribution only when the opponent is within
    # range AND its recent bid shows it will actually bid that high. A
    # spiralled-down crawler keeps crawling; modelling it as a finisher
    # overestimates its bids and makes us over-bid into the pack and blocked.
    dist = TARGET - opp_pos
    if 1 <= dist <= 100 and (not hist or hist[-1] >= dist - 4):
        fin = [0.0] * 101
        fs = 0.0
        for b in range(dist, 101):
            w = math.exp(-(b - dist) / 22.0)
            fin[b] = w
            fs += w
        if fs > 0.0:
            for b in range(1, 101):
                fin[b] /= fs
                p[b] = 0.5 * p[b] + 0.5 * fin[b]

    return p


def detect_pack(opps):
    """True when opponents look like a spiralling best-response pack."""
    if len(opps) < 2:
        return False
    recent = [lb for _c, _d, _cdf, lb in opps if lb >= 1]
    if len(recent) < 2:
        return False
    return (max(recent) - min(recent)) <= 10


def recent_ceil(opps):
    bids = sorted((lb for _c, _d, _cdf, lb in opps if lb >= 1), reverse=True)
    if not bids:
        return 0
    return bids[1] if len(bids) >= 2 else bids[0]


def choose_bid(pos, cum):
    my_cum = cum[0]
    my_dist = TARGET - pos[0]
    opps = []

    for i in range(1, 4):
        if not alive[i]:
            continue
        d = opp_dist(i, pos[i])
        cdf = [0.0] * 101
        for b in range(1, 101):
            cdf[b] = cdf[b - 1] + d[b]
        last_bid = bid_hist[i][-1] if bid_hist[i] else 0
        opps.append((cum[i], TARGET - pos[i], cdf, last_bid))

    # Track the field's top bid each round for the descending-war guard below.
    cur_top = max((lb for _c, _d, _cdf, lb in opps if lb >= 1), default=0)
    if cur_top >= 1:
        top_hist.append(cur_top)
        if len(top_hist) > 12:
            del top_hist[0]

    if not opps:
        return 1

    def cdf_at(cdf, idx):
        if idx <= 0:
            return 0.0
        if idx >= 100:
            return 1.0
        return cdf[idx]

    ceil_bid = recent_ceil(opps)

    def p_safe(b):
        shift = round(block_bias)
        if shift > PARAMS["block_shift_cap"]:
            shift = int(PARAMS["block_shift_cap"])
        if ceil_bid >= 8 and b + 3 <= ceil_bid:
            shift = min(shift, max(0, ceil_bid - b - 2))
        hi = b + shift
        if hi > 100:
            hi = 100
        p_blocked = 1.0
        for opp_cum, _opp_dist, cdf, _lb in opps:
            p_blocked *= cdf_at(cdf, hi - 1 if opp_cum < my_cum else hi)
        return 1.0 - p_blocked

    if 1 <= my_dist <= 100:
        # A bigger finishing bid (>= my_dist) only helps when a rival is ALSO
        # finishing this round: overshooting TARGET wins the final position
        # tie-break against same-round finishers. If NOBODY else can finish
        # this round, reaching the line is all that matters -- bid the minimal
        # safe finish (b*ps would otherwise throw a huge over-bid that sticks
        # out above the pack and gets us blocked while leading).
        opp_finishing = any(
            opp_dist <= 100 and lb >= opp_dist - 4
            for _opp_cum, opp_dist, _cdf, lb in opps
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

        if ps_fin >= PARAMS["finish_safe"]:
            return b_fin
        # A rival is "urgently finishing" only if it is at/ahead of us, within
        # one bid of the line, AND its recent bid is actually large enough to
        # finish. A rival that could reach the line but is bidding far below
        # its distance (a spiralled-down pack) will not finish next round --
        # panicking and throwing a doomed finishing bid above the pack just
        # gets us blocked.
        urgent_finish = any(
            opp_dist <= 100 and TARGET - opp_dist >= pos[0] and lb >= opp_dist - 4
            for _opp_cum, opp_dist, _cdf, lb in opps
        )
        if urgent_finish:
            return b_fin

        if b_set > 0 and p_safe(b_set) >= PARAMS["setup_safe"]:
            return b_set
        # The EV-best setup bid is too risky -- but DON'T fall back to b_fin.
        # When we are not finishing, b_fin is a doomed full-distance bid that
        # sticks out alone above a low pack: a certain block that freezes us.
        # Step down to the highest setup bid that still clears setup_safe; a
        # small safe advance always beats a guaranteed block.
        safe_set = 0
        for b in range(1, my_dist):
            if p_safe(b) >= PARAMS["setup_safe"]:
                safe_set = b
        if safe_set > 0:
            return safe_set
        return b_set if b_set > 0 else b_fin

    best = 1
    best_ev = -1.0
    for b in range(1, 101):
        ev = b * p_safe(b)
        if ev > best_ev:
            best_ev = ev
            best = b

    pack = detect_pack(opps) or block_bias >= 8.0
    # Descending-war guard: cap every non-finishing bid so we never tie the
    # pack's (descending) top -- see war_cap.
    cap = war_cap(top_hist, pack)
    if pack and ceil_bid >= 8:
        shield = clamp_bid(ceil_bid - 1)
        shield_ev = shield * p_safe(shield)
        if (p_safe(shield) >= PARAMS["shield_safe1"]
                and shield_ev >= best_ev * PARAMS["shield_evfrac1"]):
            return min(shield, cap)
        for d in range(2, 5):
            sb = clamp_bid(ceil_bid - d)
            ev = sb * p_safe(sb)
            if p_safe(sb) >= PARAMS["shield_safe2"] and ev > shield_ev:
                shield_ev = ev
                shield = sb
        if (p_safe(shield) >= PARAMS["shield_safe3"]
                and shield_ev >= best_ev * PARAMS["shield_evfrac3"]):
            return min(shield, cap)
        if best > ceil_bid:
            best = clamp_bid(ceil_bid - 1)

    min_opp_cum = min(c for c, _d, _cdf, _lb in opps)
    if my_cum <= min_opp_cum and pack and ceil_bid >= 5:
        match = clamp_bid(ceil_bid)
        if (p_safe(match) >= PARAMS["match_safe"]
                and match * p_safe(match) > best_ev * PARAMS["match_evfrac"]):
            return min(match, cap)

    return min(best, cap)


while True:
    try:
        line = input()
    except EOFError:
        break

    if not line.strip():
        continue

    values = [int(x) for x in line.split()]
    pos = [values[0], values[2], values[4], values[6]]
    cum = [values[1], values[3], values[5], values[7]]

    if have_last:
        my_bid = cum[0] - last_cum[0]
        blocked = pos[0] == last_pos[0] and my_bid > 0
        if blocked:
            block_bias = min(PARAMS["block_cap"], block_bias + PARAMS["block_up"])
        else:
            block_bias = max(0.0, block_bias - PARAMS["block_down"])

    update_memory(pos, cum)
    print(clamp_bid(choose_bid(pos, cum)), flush=True)

    last_pos = pos[:]
    last_cum = cum[:]
    have_last = True
