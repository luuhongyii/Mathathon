"""Head-to-head: our tuned Icarus bot vs the codex_version bot.

Both bots are deterministic, so a fixed 4-bot lineup yields a single match.
To get a distribution we fill the other two seats with varied "zoo" bots from
the training suite and vary their seed -- this mimics a real 4-team lobby.
We rotate which seat each contender holds to remove seat bias, and report
average tournament points (3=win, 2, 1, 0) for our bot vs the codex bot.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))

from icarus_rl_train import (  # noqa: E402
    LearnedBot, Params, clamp_bid, TARGET,
    uniform_bot, smart_bot, aggressive_bot, cautious_bot, snipe_bot, br_bot,
)


# --- faithful port of codex_version/icarus_game.py -------------------------
class CodexBot:
    def __init__(self):
        self.hist = [[] for _ in range(4)]
        self.alive = [True] * 4
        self.zero = [0] * 4
        self.last_pos = [0] * 4
        self.last_cum = [0] * 4
        self.have = False
        self.block_bias = 0

    @staticmethod
    def _median(xs):
        if not xs:
            return 0
        s = sorted(xs)
        return s[len(s) // 2]

    @staticmethod
    def _spread(xs):
        if not xs:
            return 100
        return max(xs) - min(xs)

    def _update(self, pos, cum):
        if not self.have:
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
                self.zero[i] += 1
                if self.zero[i] >= 2:
                    self.alive[i] = False
                    self.hist[i].clear()
                continue
            self.zero[i] = 0
            if 1 <= dc <= 100:
                self.hist[i].append(dc)
                if len(self.hist[i]) > 12:
                    del self.hist[i][0]

    def _predict(self, i, pos_i):
        h = self.hist[i]
        dist = TARGET - pos_i
        if 1 <= dist <= 100:
            return dist
        if not h:
            return 64
        r = h[-5:]
        m = self._median(r)
        if len(r) >= 3 and self._spread(r[-3:]) <= 2:
            return clamp_bid(round(sum(r[-3:]) / 3))
        if len(r) >= 4:
            d = r[-1] - r[-3]
            if abs(d) <= 12:
                return clamp_bid(r[-1] + d // 2)
        return clamp_bid(m)

    def _choose(self, pos, cum):
        my_dist = TARGET - pos[0]
        opp = [i for i in range(1, 4) if self.alive[i]]
        if not opp:
            return 1
        preds = sorted(((self._predict(i, pos[i]), i) for i in opp), reverse=True)
        top = preds[0][0]
        second = preds[1][0] if len(preds) > 1 else 1
        if all(not self.hist[i] for i in opp):
            return 63
        danger_finish = any(1 <= TARGET - pos[i] <= 100 for i in opp)
        if 1 <= my_dist <= 100:
            if top > my_dist + 1 or danger_finish:
                return clamp_bid(my_dist)
            setup = clamp_bid(min(my_dist - 1, max(1, top - 1)))
            return setup if setup >= 1 else clamp_bid(my_dist)
        top_hist = self.hist[preds[0][1]]
        top_reliable = (len(top_hist) >= 2
                        and self._spread(top_hist[-min(4, len(top_hist)):]) <= 10)
        if top >= 12 and top_reliable:
            b = top - 1 - self.block_bias
            if b <= second - 4 and top - second <= 8:
                b = second - 1
            return clamp_bid(b)
        if top - second <= 10 and top >= 10:
            return clamp_bid(top - 2 - self.block_bias)
        if top >= 70:
            return clamp_bid(top - 2 - self.block_bias)
        if second >= 45:
            return clamp_bid(second - 1)
        return 52

    def choose(self, pos, cum):
        if self.have:
            my_bid = cum[0] - self.last_cum[0]
            blocked = my_bid > 0 and pos[0] == self.last_pos[0]
            if blocked:
                self.block_bias = min(18, self.block_bias + 3)
            else:
                self.block_bias = max(0, self.block_bias - 1)
        self._update(pos, cum)
        bid = clamp_bid(self._choose(pos, cum))
        self.last_pos = list(pos)
        self.last_cum = list(cum)
        self.have = True
        return bid


# --- strategy adapters (see itself as index 0) -----------------------------
def reindex(pos, cum, i):
    order = [i] + [j for j in range(4) if j != i]
    return [pos[j] for j in order], [cum[j] for j in order]


def ours_strategy(pos, cum, i, rng, st):
    bot = st.get("bot")
    if bot is None:
        bot = LearnedBot(Params())
        st["bot"] = bot
    rp, rc = reindex(pos, cum, i)
    return bot.choose(rp, rc)


def codex_strategy(pos, cum, i, rng, st):
    bot = st.get("bot")
    if bot is None:
        bot = CodexBot()
        st["bot"] = bot
    rp, rc = reindex(pos, cum, i)
    return bot.choose(rp, rc)


ZOO = [uniform_bot, smart_bot, aggressive_bot, cautious_bot, snipe_bot, br_bot]


def play(strategies, seed):
    """Run one 4-player Icarus match; return tournament points per seat."""
    rng = random.Random(seed)
    pos, cum = [0, 0, 0, 0], [0, 0, 0, 0]
    dead = [False] * 4
    state = [{} for _ in range(4)]
    for _ in range(60):
        bids = [0, 0, 0, 0]
        for i in range(4):
            if not dead[i]:
                bids[i] = clamp_bid(int(strategies[i](pos, cum, i, rng, state[i])))
        active = [i for i in range(4) if not dead[i]]
        for i in active:
            cum[i] += bids[i]
        hi = max(bids[i] for i in active)
        tied = [i for i in active if bids[i] == hi]
        low = min(cum[i] for i in tied)
        blocked = {i for i in tied if cum[i] == low}
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
    return pts


def main():
    n_seeds = 300
    our_pts = codex_pts = 0
    our_wins = codex_wins = 0
    matches = 0
    for s in range(n_seeds):
        rng = random.Random(10_000 + s)
        z1, z2 = rng.choice(ZOO), rng.choice(ZOO)
        # Rotate the 4 seats so neither contender has a fixed-seat advantage.
        for rot in range(4):
            line = [ours_strategy, codex_strategy, z1, z2]
            line = line[rot:] + line[:rot]
            pts = play(line, 20_000 + s * 7 + rot)
            our_seat = line.index(ours_strategy)
            cod_seat = line.index(codex_strategy)
            our_pts += pts[our_seat]
            codex_pts += pts[cod_seat]
            our_wins += pts[our_seat] == 3
            codex_wins += pts[cod_seat] == 3
            matches += 1

    print(f"matches: {matches}  (our bot + codex bot + 2 varied zoo bots)")
    print(f"  OUR  bot : avg_pts={our_pts / matches:.3f}  "
          f"outright_wins={our_wins}/{matches} ({100*our_wins/matches:.0f}%)")
    print(f"  CODEX bot: avg_pts={codex_pts / matches:.3f}  "
          f"outright_wins={codex_wins}/{matches} ({100*codex_wins/matches:.0f}%)")
    print(f"  (fair share = 1.500; head-to-head our_pts-codex_pts = "
          f"{(our_pts - codex_pts) / matches:+.3f}/match)")

    # Direct duel: 2 of ours vs 2 codex, no zoo bots.
    duel_our = duel_cod = 0
    for s in range(n_seeds):
        for rot in range(4):
            line = [ours_strategy, codex_strategy, ours_strategy, codex_strategy]
            line = line[rot:] + line[:rot]
            pts = play(line, 50_000 + s * 7 + rot)
            for i in range(4):
                if line[i] is ours_strategy:
                    duel_our += pts[i]
                else:
                    duel_cod += pts[i]
    dm = n_seeds * 4
    print(f"\n2-ours vs 2-codex ({dm} matches, 2 seats each):")
    print(f"  OUR  total avg_pts/seat={duel_our / (dm * 2):.3f}")
    print(f"  CODEX total avg_pts/seat={duel_cod / (dm * 2):.3f}")


if __name__ == "__main__":
    main()
