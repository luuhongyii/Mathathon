"""Fast Icarus Game submission.

Paste this whole file into the Mathathon editor.
This version uses constant-time heuristics to avoid platform timeouts.
"""

import random
from collections import deque


TARGET = 999

history = [deque(maxlen=24) for _ in range(4)]
alive = [True, True, True, True]
last_players = None
round_no = 0


def clamp_bid(x):
    return max(1, min(100, int(round(x))))


def update_memory(players):
    if last_players is None:
        return

    for i in range(4):
        prev_p, prev_c = last_players[i]
        cur_p, cur_c = players[i]
        dc = cur_c - prev_c

        if alive[i] and (dc < 0 or (cur_p == 0 and cur_c == 0 and prev_c > 0)):
            alive[i] = False
            history[i].clear()
            continue

        if alive[i] and 1 <= dc <= 100:
            history[i].append(dc)


def mean_recent(i, fallback):
    h = history[i]
    if not h:
        return fallback
    recent = list(h)[-8:]
    return sum(recent) / len(recent)


def max_recent(i, fallback):
    h = history[i]
    if not h:
        return fallback
    return max(list(h)[-8:])


def choose_bid(players):
    my_pos = players[0][0]
    dist = TARGET - my_pos

    estimates = []
    recent_highs = []

    for i in range(1, 4):
        if not alive[i]:
            continue

        opp_dist = TARGET - players[i][0]
        fallback = 64
        if opp_dist <= 100:
            fallback = max(45, min(100, opp_dist))

        estimates.append(int(round(mean_recent(i, fallback))))
        recent_highs.append(max_recent(i, fallback + 8))

    if not estimates:
        return clamp_bid(dist if dist <= 100 else 70)

    estimates.sort()
    recent_highs.sort()

    expected_high = estimates[-1]
    expected_second = estimates[-2] if len(estimates) >= 2 else 55
    recent_high = recent_highs[-1]

    leader = max(p for p, _c in players)
    gap = leader - my_pos

    if dist <= 100:
        finish_bid = clamp_bid(dist)
        if finish_bid <= expected_high - 2 or finish_bid <= recent_high - 5:
            bid = finish_bid
        else:
            bid = min(finish_bid, expected_high - random.randint(3, 8))
    elif gap > 160:
        bid = max(expected_second + 6, expected_high - random.randint(1, 4))
        bid += random.randint(-2, 5)
    elif gap > 70:
        bid = max(expected_second + 2, expected_high - random.randint(3, 7))
        bid += random.randint(-3, 3)
    elif my_pos >= leader - 20 and my_pos >= 760:
        bid = min(expected_second + 1, expected_high - random.randint(6, 12))
        bid += random.randint(-2, 2)
    else:
        bid = expected_second + random.randint(0, 6)
        bid = min(bid, expected_high - random.randint(2, 7))
        bid += random.randint(-3, 3)

    if round_no <= 1 and dist > 100:
        bid = 63 + random.randint(-3, 5)

    if dist > 100:
        bid = max(bid, 34)

    return clamp_bid(bid)


while True:
    try:
        line = input()
    except EOFError:
        break

    if not line.strip():
        continue

    values = [int(x) for x in line.split()]
    players = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]

    update_memory(players)
    print(choose_bid(players), flush=True)

    last_players = players
    round_no += 1
