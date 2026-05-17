# Icarus Game submission (Python port of icarus_game.cpp).
#
# Rules: 4 players race to position 999. Each round every player bids 1..100;
# the bid is added to a cumulative total. The highest bidder is "Icarus" and
# is blocked (does not move); on a tie for highest, the tied player(s) with
# the lowest cumulative total are blocked. Everyone not blocked advances by
# their bid. Final ranking is by position (pessimistic on ties): 3/2/1/0 pts.
#
# This is a faithful port of the CEM-tuned C++ bot: it learns each opponent's
# bid distribution, computes the exact block probability under the tie-break
# rule, and rides just below a detected best-response pack.

import sys
from math import exp, floor

TARGET = 999


# Tunable strategy parameters. Single source of truth shared (by value) with
# icarus_game.cpp and the LearnedBot in icarus_rl_train.py. Keep in lockstep.
class Params:
    no_info_floor = 0.0171807    # uniform floor on every opponent dist
    recency_decay = 0.624394     # kernel recency weight base
    tight_sd_mul = 1.29265       # sigT = mul*sd + add ...
    tight_sd_add = 0.280066
    trust_base = 0.05            # kTrust = base + sd_mul*sd + n_mul/n
    trust_sd_mul = 0.462153
    trust_n_mul = 7.00283
    trend_mul = 0.508882         # trend shift = mul * slope
    block_up = 8.88151           # blockBias increment when blocked
    block_down = 1.15079         # blockBias decay when not blocked
    block_cap = 66.0955          # blockBias hard ceiling
    finish_safe = 0.697257       # pSafe needed to take a finish shot
    setup_safe = 0.691082        # pSafe needed to spend a setup round
    # Tight-kernel (best-responder pack) branch.
    tight_sd_thresh = 1.5        # sd below this -> tight reliable shield
    tight_sigma = 0.2            # sigT override inside the tight branch
    tight_w_boost = 0.0845868    # extra weight on the tight kernel
    # Shield / pack-riding thresholds.
    shield_safe1 = 0.661246      # ride ceiling-1 if pSafe >= this ...
    shield_evfrac1 = 0.981208    # ... and EV >= bestEV * this
    shield_safe2 = 0.793981      # deeper shield candidate gate
    shield_safe3 = 0.917121      # final shield accept gate ...
    shield_evfrac3 = 0.886851    # ... and EV >= bestEV * this
    match_safe = 0.545142        # low-cum match gate
    match_evfrac = 0.859635      # low-cum match EV fraction
    block_shift_cap = 19.5093    # max effective blockBias shift in pSafe
    # No-info opening: Gaussian prior over opponents' round-0 bids.
    open_mean = 52.0
    open_sd = 16.0


PARAMS = Params()

bid_hist = [[] for _ in range(4)]   # observed bids (cumulative deltas)
alive = [True, True, True, True]
blocked_last = [False, False, False, False]
zero_streak = [0, 0, 0, 0]
last_pos = [0, 0, 0, 0]
last_cum = [0, 0, 0, 0]
have_last = False

# Feedback term: when blocked, opponents bid lower than the model thinks.
# Shifts risk assessment downward until safely under them, then decays away.
block_bias = 0.0
my_blocked_last = False

# Field's top bid per round, for the descending-war guard (see war_cap).
top_hist = []


def clampi(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def clampf(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def clamp_bid(x):
    x = int(x)
    return 1 if x < 1 else (100 if x > 100 else x)


def update_memory(pos, cum):
    if not have_last:
        return
    for i in range(4):
        blocked_last[i] = False
        if not alive[i]:
            continue
        dc = cum[i] - last_cum[i]
        if dc > 0 and pos[i] == last_pos[i]:
            blocked_last[i] = True

        # A player whose totals were reset to 0 has died.
        if dc < 0 or (pos[i] == 0 and cum[i] == 0 and last_cum[i] > 0):
            alive[i] = False
            bid_hist[i].clear()
            blocked_last[i] = False
            continue
        # A live player always bids >= 1, so its cumulative total grows every
        # round. No growth means the player was not prompted (dead).
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


def normalize(a):
    s = 0.0
    for b in range(1, 101):
        s += a[b]
    if s <= 0.0:
        for b in range(1, 101):
            a[b] = 1.0 / 100.0
    else:
        for b in range(1, 101):
            a[b] /= s


# Recency-weighted Gaussian-kernel estimate of a bid distribution.
def kernel_dist(h, sigma):
    d = [0.0] * 101
    n = len(h)
    if sigma < 0.05:
        sigma = 0.05
    two_sig2 = 2.0 * sigma * sigma
    for k in range(n):
        recw = PARAMS.recency_decay ** (n - 1 - k)
        v = h[k]
        for b in range(1, 101):
            dd = b - v
            d[b] += recw * exp(-dd * dd / two_sig2)
    normalize(d)
    return d


# Probability distribution of opponent i's next bid over 1..100.
def opp_dist(i, opp_pos):
    h = bid_hist[i]
    n = len(h)
    p = [0.0] * 101

    if n == 0:
        # No data: Gaussian opening prior (real lobbies open ~44-60).
        denom = 2.0 * PARAMS.open_sd * PARAMS.open_sd
        for b in range(1, 101):
            dd = b - PARAMS.open_mean
            p[b] = exp(-(dd * dd) / denom)
        normalize(p)
    else:
        # Spread of the opponent's recent bids over a SHORT recent window.
        start = max(0, n - 6)
        c = n - start
        mean = sum(h[start:n]) / c
        var = 0.0
        for k in range(start, n):
            dd = h[k] - mean
            var += dd * dd
        sd = (var / c) ** 0.5

        # Tight kernel is the point estimate; wide kernel is the hedge.
        tight_branch = sd < PARAMS.tight_sd_thresh
        if tight_branch:
            sig_t = PARAMS.tight_sigma
        else:
            sig_t = clampf(PARAMS.tight_sd_mul * sd + PARAMS.tight_sd_add,
                           0.35, 9.0)
        sig_w = max(12.0, 2.2 * sig_t)
        tight = kernel_dist(h, sig_t)
        wide = kernel_dist(h, sig_w)

        k_trust = clampf(PARAMS.trust_base + PARAMS.trust_sd_mul * sd +
                         PARAMS.trust_n_mul / n, 0.25, 9.0)
        w = n / (n + k_trust)
        if tight_branch:
            w = min(0.92, w + PARAMS.tight_w_boost)
        for b in range(1, 101):
            p[b] = w * tight[b] + (1.0 - w) * wide[b]

        # Trend extrapolation: project the distribution one step in the
        # direction the opponent's bids are moving.
        if n >= 4:
            L = min(n, 8)
            half = L // 2
            m1 = sum(h[n - L:n - L + half])
            m2 = sum(h[n - half:n])
            m1 /= half
            m2 /= half
            slope = (m2 - m1) / half
            shift = clampf(PARAMS.trend_mul * slope, -9.0, 9.0)

            # Skip extrapolation when the most recent bids are already flat.
            rl = min(n, 4)
            rmean = sum(h[n - rl:n]) / rl
            rspread = 0.0
            for k in range(n - rl, n):
                rspread += abs(h[k] - rmean)
            rspread /= rl
            if rspread < 2.0:
                shift = 0.0

            if abs(shift) > 0.3:
                sh = [0.0] * 101
                for b in range(1, 101):
                    src = b - shift
                    lo = int(floor(src))
                    fr = src - lo
                    a = clampi(lo, 1, 100)
                    bb = clampi(lo + 1, 1, 100)
                    sh[b] = p[a] * (1.0 - fr) + p[bb] * fr
                normalize(sh)
                p = sh

    # Tiny uniform floor: never assign a bid exactly zero probability.
    floor_w = PARAMS.no_info_floor
    for b in range(1, 101):
        p[b] = (1.0 - floor_w) * p[b] + floor_w / 100.0

    # Finisher adjustment: an opponent within finishing range will likely bid
    # at or above its remaining distance -- but only if its recent bidding
    # shows it will actually bid that high.
    dist = TARGET - opp_pos
    if 1 <= dist <= 100 and (n == 0 or h[-1] >= dist - 4):
        fin = [0.0] * 101
        fs = 0.0
        for b in range(dist, 101):
            w = exp(-(b - dist) / 22.0)
            fin[b] = w
            fs += w
        if fs > 0.0:
            for b in range(1, 101):
                fin[b] /= fs
            mix = 0.5
            for b in range(1, 101):
                p[b] = (1.0 - mix) * p[b] + mix * fin[b]
    return p


class OppInfo:
    __slots__ = ("cum", "dist", "last_bid", "blocked", "cdf")

    def __init__(self):
        self.cum = 0
        self.dist = 0
        self.last_bid = 0
        self.blocked = False
        self.cdf = [0.0] * 101  # cdf[b] = P(bid <= b), cdf[0] = 0


# True when opponents look like a spiralling best-response pack.
def detect_pack(opps):
    if len(opps) < 2:
        return False
    recent = [o.last_bid for o in opps if o.last_bid >= 1]
    if len(recent) < 2:
        return False
    return (max(recent) - min(recent)) <= 10


def recent_ceil(opps):
    bids = sorted((o.last_bid for o in opps
                   if o.last_bid >= 1 and not (o.blocked and o.last_bid >= 25)),
                  reverse=True)
    if not bids:
        bids = sorted((o.last_bid for o in opps if o.last_bid >= 1),
                      reverse=True)
    if not bids:
        return 0
    return bids[1] if len(bids) >= 2 else bids[0]


# Safe bid ceiling during a descending bid war (else a no-op high cap of 1000).
def war_cap(th, pack):
    m = len(th)
    if not pack or m < 3:
        return 1000
    if m >= 4 and th[m - 4] <= 4 and th[m - 3] <= 4 and \
            th[m - 2] <= 4 and th[m - 1] <= 4:
        return 1
    a, b, c = th[m - 3], th[m - 2], th[m - 1]
    if a < b or b < c:  # require a consistent descent
        return 1000
    drop = (a - c) / 2.0
    if drop < 1.0:
        return 1000
    if drop > 12.0:
        drop = 12.0
    cap = int(c - drop - 2.0)
    return cap if cap >= 1 else 1


def choose_bid(pos, cum):
    my_cum = cum[0]
    my_dist = TARGET - pos[0]

    opps = []
    for i in range(1, 4):
        if not alive[i]:
            continue
        o = OppInfo()
        o.cum = cum[i]
        o.dist = TARGET - pos[i]
        h = bid_hist[i]
        o.last_bid = h[-1] if h else 0
        o.blocked = blocked_last[i]
        d = opp_dist(i, pos[i])
        o.cdf[0] = 0.0
        for b in range(1, 101):
            o.cdf[b] = o.cdf[b - 1] + d[b]
        opps.append(o)

    # Track the field's top bid each round for the descending-war guard.
    cur_top = 0
    for o in opps:
        if not (o.blocked and o.last_bid >= 25):
            cur_top = max(cur_top, o.last_bid)
    if cur_top >= 1:
        top_hist.append(cur_top)
        if len(top_hist) > 12:
            del top_hist[0]

    if not opps:
        return 1  # alone: cannot move, just answer validly

    ceil_bid = recent_ceil(opps)

    def cdf_at(o, idx):
        if idx <= 0:
            return 0.0
        if idx >= 100:
            return 1.0
        return o.cdf[idx]

    # Exact probability that my bid b is the blocked one. Blocked iff every
    # lower-cum opponent bids < b and every equal/higher-cum opponent bids <= b.
    def p_safe(b):
        shift = int(round(block_bias))
        if shift > int(PARAMS.block_shift_cap):
            shift = int(PARAMS.block_shift_cap)
        if ceil_bid >= 8 and b + 3 <= ceil_bid:
            shift = min(shift, max(0, ceil_bid - b - 2))
        hi = b + shift
        if hi > 100:
            hi = 100
        pb = 1.0
        for o in opps:
            if o.cum < my_cum:
                pb *= cdf_at(o, hi - 1)
            else:
                pb *= cdf_at(o, hi)
        return 1.0 - pb

    if 1 <= my_dist <= 100:
        # I can reach the line this round. Pick the finishing bid (>= myDist)
        # most likely to get through unblocked.
        opp_finishing = False
        for o in opps:
            if o.dist <= 100 and o.last_bid >= o.dist - 4:
                opp_finishing = True
        b_fin = my_dist
        ps_fin = -1.0
        fin_ev = -1.0
        for b in range(my_dist, 101):
            ps = p_safe(b)
            ev = b * ps if opp_finishing else ps
            if ev > fin_ev or (ev == fin_ev and ps > ps_fin):
                fin_ev = ev
                ps_fin = ps
                b_fin = b

        # Setup round: advance safely under the pack to shrink the distance.
        b_set = 0
        best_ev = -1.0
        for b in range(1, my_dist):
            ev = b * p_safe(b)
            if ev > best_ev:
                best_ev = ev
                b_set = b

        # A safe finish dominates everything else -- take it.
        if ps_fin >= PARAMS.finish_safe:
            return b_fin

        # If a rival is already at least as far as us and can finish, there
        # is no time to set up.
        urgent_finish = False
        for o in opps:
            if o.dist <= 100 and TARGET - o.dist >= pos[0] and \
                    o.last_bid >= o.dist - 4:
                urgent_finish = True
        if urgent_finish:
            return b_fin

        if b_set > 0 and p_safe(b_set) >= PARAMS.setup_safe:
            return b_set
        # EV-best setup too risky -- step down to the safest setup available.
        safe_set = 0
        for b in range(1, my_dist):
            if p_safe(b) >= PARAMS.setup_safe:
                safe_set = b
        if safe_set > 0:
            return safe_set
        return b_set if b_set > 0 else b_fin

    # Cannot finish yet: maximise expected position gain.
    best = 1
    best_ev = -1.0
    for b in range(1, 101):
        ev = b * p_safe(b)
        if ev > best_ev:
            best_ev = ev
            best = b

    pack = detect_pack(opps) or block_bias >= 8.0
    # Descending-war guard: cap every non-finishing bid.
    cap = war_cap(top_hist, pack)
    for o in opps:
        if o.blocked and o.last_bid >= 25:
            cap = min(cap, max(1, ceil_bid))
    if my_blocked_last and ceil_bid >= 8:
        cap = min(cap, max(1, ceil_bid - 6))
    if pack and ceil_bid >= 8:
        shield = clampi(ceil_bid - 1, 1, 100)
        shield_ev = shield * p_safe(shield)
        # Ride just under the pack when it beats the naive EV peak.
        if p_safe(shield) >= PARAMS.shield_safe1 and \
                shield_ev >= best_ev * PARAMS.shield_evfrac1:
            return min(shield, cap)
        for d in range(2, 5):
            sb = clampi(ceil_bid - d, 1, 100)
            ev = sb * p_safe(sb)
            if p_safe(sb) >= PARAMS.shield_safe2 and ev > shield_ev:
                shield_ev = ev
                shield = sb
        if p_safe(shield) >= PARAMS.shield_safe3 and \
                shield_ev >= best_ev * PARAMS.shield_evfrac3:
            return min(shield, cap)
        if best > ceil_bid:
            best = clampi(ceil_bid - 1, 1, 100)

    # Lowest cumulative: can match the pack without losing tie-breaks as often.
    min_opp_cum = opps[0].cum
    for o in opps:
        min_opp_cum = min(min_opp_cum, o.cum)
    if my_cum <= min_opp_cum and pack and ceil_bid >= 5:
        match = clampi(ceil_bid, 1, 100)
        if p_safe(match) >= PARAMS.match_safe and \
                match * p_safe(match) > best_ev * PARAMS.match_evfrac:
            return min(match, cap)

    return min(best, cap)


def main():
    global have_last, block_bias, my_blocked_last

    buf = []
    for line in sys.stdin:
        buf.extend(line.split())
        while len(buf) >= 8:
            vals = [int(x) for x in buf[:8]]
            del buf[:8]

            pos = [vals[0], vals[2], vals[4], vals[6]]
            cum = [vals[1], vals[3], vals[5], vals[7]]

            # Did my previous bid get blocked? (position unchanged, total grew.)
            if have_last:
                my_bid = cum[0] - last_cum[0]
                blocked = (pos[0] == last_pos[0]) and my_bid > 0
                my_blocked_last = blocked
                if blocked:
                    block_bias = min(PARAMS.block_cap,
                                     block_bias + PARAMS.block_up)
                else:
                    block_bias = max(0.0, block_bias - PARAMS.block_down)
            else:
                my_blocked_last = False

            update_memory(pos, cum)
            sys.stdout.write(str(clamp_bid(choose_bid(pos, cum))) + "\n")
            sys.stdout.flush()

            last_pos[:] = pos
            last_cum[:] = cum
            have_last = True


if __name__ == "__main__":
    main()
