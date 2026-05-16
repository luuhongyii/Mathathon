"""Icarus Game submission.

Python port of icarus_game.cpp. It uses the same deterministic opponent model:
learn bid distributions from cumulative deltas, compute the exact probability
of being blocked for each bid, and choose the bid with the best expected gain.
"""

import math


TARGET = 999

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
    denom = 2.0 * sigma * sigma
    n = len(hist)
    for k, v in enumerate(hist):
        recw = 0.70 ** (n - 1 - k)
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

        sig_t = clamp(0.85 * sd + 0.35, 0.35, 9.0)
        sig_w = max(15.0, 2.5 * sig_t)
        tight = kernel_dist(hist, sig_t)
        wide = kernel_dist(hist, sig_w)

        k_trust = clamp(0.3 + 0.33 * sd + 4.5 / n, 0.3, 9.0)
        w = n / (n + k_trust)
        for b in range(1, 101):
            p[b] = w * tight[b] + (1.0 - w) * wide[b]

        if n >= 4:
            length = min(n, 8)
            half = length // 2
            m1 = sum(hist[n - length:n - length + half]) / half
            m2 = sum(hist[n - half:n]) / half
            slope = (m2 - m1) / half
            shift = clamp(1.5 * slope, -9.0, 9.0)

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

    floor_w = 0.01
    for b in range(1, 101):
        p[b] = (1.0 - floor_w) * p[b] + floor_w / 100.0

    dist = TARGET - opp_pos
    if 1 <= dist <= 100:
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
        opps.append((cum[i], TARGET - pos[i], cdf))

    if not opps:
        return 1

    def cdf_at(cdf, idx):
        if idx <= 0:
            return 0.0
        if idx >= 100:
            return 1.0
        return cdf[idx]

    def p_safe(b):
        hi = b + round(block_bias)
        p_blocked = 1.0
        for opp_cum, _opp_dist, cdf in opps:
            p_blocked *= cdf_at(cdf, hi - 1 if opp_cum < my_cum else hi)
        return 1.0 - p_blocked

    if 1 <= my_dist <= 100:
        b_fin = my_dist
        ps_fin = -1.0
        for b in range(my_dist, 101):
            ps = p_safe(b)
            if ps > ps_fin:
                ps_fin = ps
                b_fin = b

        if ps_fin >= 0.80:
            return b_fin
        if any(opp_dist <= 100 for _opp_cum, opp_dist, _cdf in opps):
            return b_fin

        b_set = 0
        best_ev = -1.0
        for b in range(1, my_dist):
            ev = b * p_safe(b)
            if ev > best_ev:
                best_ev = ev
                b_set = b
        if b_set > 0 and p_safe(b_set) >= 0.70:
            return b_set
        return b_fin

    best = 1
    best_ev = -1.0
    for b in range(1, 101):
        ev = b * p_safe(b)
        if ev > best_ev:
            best_ev = ev
            best = b
    return best


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
            block_bias = min(48.0, block_bias + 9.0)
        else:
            block_bias = max(0.0, block_bias - 3.0)

    update_memory(pos, cum)
    print(clamp_bid(choose_bid(pos, cum)), flush=True)

    last_pos = pos[:]
    last_cum = cum[:]
    have_last = True
