"""Local Icarus referee: my compiled bot (player 0) vs scripted opponents."""
import random
import subprocess
import sys

TARGET = 999
BOT = ["./icarus_test.exe"]


def run_match(opp_strats, seed, verbose=False):
    rng = random.Random(seed)
    proc = subprocess.Popen(BOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    pos = [0, 0, 0, 0]
    cum = [0, 0, 0, 0]
    dead = [False, False, False, False]
    state = [{} for _ in range(4)]

    for rnd in range(60):
        bids = [0, 0, 0, 0]
        # player 0 = my bot
        line = " ".join(str(v) for i in range(4) for v in (pos[i], cum[i]))
        proc.stdin.write(line + "\n")
        proc.stdin.flush()
        out = proc.stdout.readline().strip()
        try:
            bids[0] = max(1, min(100, int(out)))
        except ValueError:
            dead[0] = True
            pos[0] = cum[0] = 0
        # opponents
        for i in range(1, 4):
            if dead[i]:
                continue
            b = opp_strats[i - 1](pos, cum, i, rng, state[i])
            bids[i] = max(1, min(100, int(b)))

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
        if verbose:
            print(f"r{rnd}: bids={bids} blocked={sorted(blocked)} pos={pos}")
        if any(pos[i] >= TARGET for i in active):
            break

    try:
        proc.stdin.close()
        proc.wait(timeout=2)
    except Exception:
        proc.kill()

    order = sorted(range(4), key=lambda i: -pos[i])
    pts = [0, 0, 0, 0]
    rank = 0
    i = 0
    vals = [pos[o] for o in order]
    # pessimistic: tied players get the worse rank
    j = 0
    while j < 4:
        k = j
        while k + 1 < 4 and vals[k + 1] == vals[j]:
            k += 1
        worse = k  # 0=1st .. 3=4th
        for t in range(j, k + 1):
            pts[order[t]] = [3, 2, 1, 0][worse]
        j = k + 1
    return pos, pts


# ---- opponent strategies -------------------------------------------------
def uniform_bot(pos, cum, i, rng, st):
    return rng.randint(1, 100)


def constant_bot(val):
    def f(pos, cum, i, rng, st):
        return val
    return f


def smart_bot(pos, cum, i, rng, st):
    """Bid just under a guessed crowd level; rush the finish."""
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
    return rng.randint(68, 88)


def aggressive_bot(pos, cum, i, rng, st):
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
    return rng.randint(82, 99)


def cautious_bot(pos, cum, i, rng, st):
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
    return rng.randint(40, 60)


def snipe_bot(pos, cum, i, rng, st):
    """Bid just below the highest opponent's last-round implied move."""
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
    others = [pos[j] for j in range(4) if j != i]
    return max(30, min(95, 70 + (max(others) - pos[i]) // 20))


def br_bot(pos, cum, i, rng, st):
    """Best-responder: bid just under the predicted second-highest rival."""
    dist = TARGET - pos[i]
    if dist <= 100:
        return dist
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


SUITES = {
    "all-random": [uniform_bot, uniform_bot, uniform_bot],
    "all-smart": [smart_bot, smart_bot, smart_bot],
    "all-aggressive": [aggressive_bot, aggressive_bot, aggressive_bot],
    "all-cautious": [cautious_bot, cautious_bot, cautious_bot],
    "mixed": [smart_bot, aggressive_bot, cautious_bot],
    "mixed2": [uniform_bot, smart_bot, snipe_bot],
    "constants": [constant_bot(70), constant_bot(85), constant_bot(55)],
    "two-aggressive": [aggressive_bot, aggressive_bot, cautious_bot],
    "all-bestresp": [br_bot, br_bot, br_bot],
    "br-mix": [br_bot, smart_bot, aggressive_bot],
    "br-snipe": [br_bot, snipe_bot, uniform_bot],
}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    if len(sys.argv) > 2 and sys.argv[2] == "-v":
        suite = sys.argv[3] if len(sys.argv) > 3 else "mixed"
        sd = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        pos, pts = run_match(SUITES[suite], seed=sd, verbose=True)
        print("final pos", pos, "pts", pts)
        sys.exit(0)

    grand = 0
    grand_n = 0
    for name, strats in SUITES.items():
        tot_pts = 0
        wins = 0
        for s in range(n):
            pos, pts = run_match(strats, seed=1000 + s)
            tot_pts += pts[0]
            if pts[0] == 3:
                wins += 1
        avg = tot_pts / n
        grand += tot_pts
        grand_n += n
        print(f"{name:16s}  avg_pts={avg:.3f}  outright_wins={wins}/{n} "
              f"({100*wins/n:.0f}%)")
    print(f"{'OVERALL':16s}  avg_pts={grand/grand_n:.3f}  (3=win, 0=last)")
