import sys

TARGET = 999

hist = [[] for _ in range(4)]
alive = [True, True, True, True]
zero = [0, 0, 0, 0]
last_pos = [0, 0, 0, 0]
last_cum = [0, 0, 0, 0]
have = False
block_bias = 0


def clamp_bid(x):
    return 1 if x < 1 else 100 if x > 100 else int(x)


def median(xs):
    if not xs:
        return 0
    s = sorted(xs)
    return s[len(s) // 2]


def spread(xs):
    if not xs:
        return 100
    return max(xs) - min(xs)


def update(pos, cum):
    global have
    if not have:
        return
    for i in range(4):
        if not alive[i]:
            continue
        dc = cum[i] - last_cum[i]
        if dc < 0 or (pos[i] == 0 and cum[i] == 0 and last_cum[i] > 0):
            alive[i] = False
            hist[i].clear()
            continue
        if dc == 0:
            zero[i] += 1
            if zero[i] >= 2:
                alive[i] = False
                hist[i].clear()
            continue
        zero[i] = 0
        if 1 <= dc <= 100:
            hist[i].append(dc)
            if len(hist[i]) > 12:
                del hist[i][0]


def predict(i, pos_i):
    h = hist[i]
    dist = TARGET - pos_i
    if 1 <= dist <= 100:
        return dist
    if not h:
        return 64
    r = h[-5:]
    m = median(r)
    if len(r) >= 3 and spread(r[-3:]) <= 2:
        return clamp_bid(round(sum(r[-3:]) / 3))
    if len(r) >= 4:
        d = r[-1] - r[-3]
        if abs(d) <= 12:
            return clamp_bid(r[-1] + d // 2)
    return clamp_bid(m)


def choose(pos, cum):
    global block_bias
    my_pos = pos[0]
    my_dist = TARGET - my_pos
    opp = [i for i in range(1, 4) if alive[i]]
    if not opp:
        return 1

    preds = [(predict(i, pos[i]), i) for i in opp]
    preds.sort(reverse=True)
    top = preds[0][0]
    second = preds[1][0] if len(preds) > 1 else 1

    # First round / no information: aim below common high openers.
    if all(not hist[i] for i in opp):
        return 63

    # If opponents are about to finish, try to finish or force a safe setup.
    danger_finish = any(1 <= TARGET - pos[i] <= 100 for i in opp)
    if 1 <= my_dist <= 100:
        if top > my_dist + 1 or danger_finish:
            return clamp_bid(my_dist)
        setup = clamp_bid(min(my_dist - 1, max(1, top - 1)))
        return setup if setup >= 1 else clamp_bid(my_dist)

    # Ride just below a reliable top bidder. This avoids being Icarus while
    # still taking large steps when someone else is likely to absorb the block.
    top_hist = hist[preds[0][1]]
    top_reliable = len(top_hist) >= 2 and spread(top_hist[-min(4, len(top_hist)):]) <= 10
    if top >= 12 and top_reliable:
        b = top - 1 - block_bias
        if b <= second - 4 and top - second <= 8:
            b = second - 1
        return clamp_bid(b)

    # Conservative EV proxy: if predictions are clustered, sit below cluster;
    # otherwise take the best middle-high bid that is unlikely to be top.
    if top - second <= 10 and top >= 10:
        return clamp_bid(top - 2 - block_bias)
    if top >= 70:
        return clamp_bid(top - 2 - block_bias)
    if second >= 45:
        return clamp_bid(second - 1)
    return 52


for line in sys.stdin:
    if not line.strip():
        continue
    parts = line.split()
    if len(parts) < 8:
        print(1, flush=True)
        continue
    try:
        vals = [int(x) for x in parts[:8]]
    except ValueError:
        print(1, flush=True)
        continue

    pos = [vals[0], vals[2], vals[4], vals[6]]
    cum = [vals[1], vals[3], vals[5], vals[7]]

    if have:
        my_bid = cum[0] - last_cum[0]
        blocked = my_bid > 0 and pos[0] == last_pos[0]
        if blocked:
            block_bias = min(18, block_bias + 3)
        else:
            block_bias = max(0, block_bias - 1)

    update(pos, cum)
    print(clamp_bid(choose(pos, cum)), flush=True)
    last_pos[:] = pos
    last_cum[:] = cum
    have = True
