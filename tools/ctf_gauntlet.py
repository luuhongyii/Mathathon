"""Realistic opponent gauntlet for the Capture-the-Flag bot.

The local random/smart sims are too weak — they say 20-0 while we lose real
games. This gauntlet is built from opponents actually observed in real games:
  - RunnerBot   : pure solo attacker (tungtung, Neumannism's runner)
  - CamperBot   : parks in the oasis and stalls (the deadlock trigger)
  - SmartBot    : ctf_sim's strong BFS runner+guard split
A candidate executable plays each opponent TEAM over many boards; the script
reports win/draw/loss so changes are measured against realistic opposition
instead of being reactive to the latest game log.

Usage:
    python tools/ctf_gauntlet.py [path-to-exe]
"""
import os
import sys
import subprocess
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CTF_BOT", "placeholder")
import ctf_sim as sim

SIZE = 29
MOVES = (("u", 0, -1), ("d", 0, 1), ("l", -1, 0), ("r", 1, 0), ("s", 0, 0))


def obs_of(board):
    return {(i % SIZE, i // SIZE) for i, c in enumerate(board) if c == "#"}


def bfs(obs, sources):
    dist = {}
    q = deque()
    for s in sources:
        if s not in obs:
            dist[s] = 0
            q.append(s)
    while q:
        x, y = q.popleft()
        for _, dx, dy in MOVES[:4]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in obs
                    and (nx, ny) not in dist):
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist


def greedy_step(x, y, obs, field, enemies, blue, chase=None):
    """Pick the move minimising `field` (or Chebyshev to `chase`), avoiding
    cells where we would be caught."""
    eterr = "R" if blue else "B"
    best, best_key = "s", None
    for mv, dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if mv != "s" and (not (0 <= nx < SIZE and 0 <= ny < SIZE)
                          or (nx, ny) in obs):
            continue
        if chase is not None:
            d = max(abs(nx - chase[0]), abs(ny - chase[1]))
        else:
            d = field.get((nx, ny), 10 ** 9)
        unsafe = 0
        if sim.territory(nx, ny) == eterr:
            for ex, ey in enemies:
                if max(abs(ex - nx), abs(ey - ny)) <= 1:
                    unsafe = 1
                    break
        key = (unsafe, d)
        if best_key is None or key < best_key:
            best_key, best = key, mv
    return best


class RunnerBot:
    """Pure solo attacker: beeline the enemy flag, carry home, refill when
    low. Models tungtung / Neumannism's runner."""

    def __init__(self, board, pid):
        self.obs = obs_of(board)
        self.blue = pid < 2
        eflag = (28, 28) if self.blue else (0, 0)
        terr = "B" if self.blue else "R"
        self.d_eflag = bfs(self.obs, [eflag])
        self.d_home = bfs(self.obs, [(x, y) for x in range(SIZE)
                                     for y in range(SIZE)
                                     if sim.territory(x, y) == terr])
        self.d_oasis = bfs(self.obs, [(x, y) for x in range(12, 17)
                                      for y in range(12, 17)])

    def move(self, state):
        n = [int(v) for v in state.split()]
        x, y, h, carry = n[:4]
        if x < 0:
            return "s"
        enemies = [(n[o], n[o + 1]) for o in (8, 12) if n[o] >= 0]
        field = self.d_home if carry else self.d_eflag
        do = self.d_oasis.get((x, y), 10 ** 9)
        if not sim.in_oasis(x, y) and do < 10 ** 8 and h < 2 * do + 16:
            field = self.d_oasis
        return greedy_step(x, y, self.obs, field, enemies, self.blue)


class HunterBot:
    """Attacker that drops back to chase the enemy flag-carrier — models a
    runner that also defends (Neumannism plugging the exit)."""

    def __init__(self, board, pid):
        self.obs = obs_of(board)
        self.blue = pid < 2
        eflag = (28, 28) if self.blue else (0, 0)
        terr = "B" if self.blue else "R"
        self.d_eflag = bfs(self.obs, [eflag])
        self.d_home = bfs(self.obs, [(x, y) for x in range(SIZE)
                                     for y in range(SIZE)
                                     if sim.territory(x, y) == terr])

    def move(self, state):
        n = [int(v) for v in state.split()]
        x, y, h, carry = n[:4]
        if x < 0:
            return "s"
        mate = n[4:8]
        enemies = [(n[o], n[o + 1]) for o in (8, 12) if n[o] >= 0]
        ecar = [(n[o], n[o + 1]) for o in (8, 12)
                if n[o] >= 0 and n[o + 3] == 1]
        if carry:
            return greedy_step(x, y, self.obs, self.d_home, enemies, self.blue)
        if ecar:
            return greedy_step(x, y, self.obs, None, enemies, self.blue,
                               chase=ecar[0])
        return greedy_step(x, y, self.obs, self.d_eflag, enemies, self.blue)


class GuardBot:
    """Holds near its own flag and intercepts the nearest enemy attacker —
    models a real defender (Ovon1, Neumannism's P0). Paired with a runner
    this is the '1 attacker + 1 active defender' team that beat us."""

    def __init__(self, board, pid):
        self.obs = obs_of(board)
        self.blue = pid < 2
        self.flag = (0, 0) if self.blue else (28, 28)
        self.d_flag = bfs(self.obs, [self.flag])

    def move(self, state):
        n = [int(v) for v in state.split()]
        x, y, h, carry = n[:4]
        if x < 0:
            return "s"
        enemies = [(n[o], n[o + 1]) for o in (8, 12) if n[o] >= 0]
        ecar = [(n[o], n[o + 1]) for o in (8, 12)
                if n[o] >= 0 and n[o + 3] == 1]
        if ecar:
            return greedy_step(x, y, self.obs, None, enemies, self.blue,
                               chase=ecar[0])
        tgt, best = None, 10 ** 9
        for ex, ey in enemies:
            d = self.d_flag.get((ex, ey), 10 ** 9)
            if d < best:
                best, tgt = d, (ex, ey)
        if tgt is not None and best <= 20:
            return greedy_step(x, y, self.obs, None, enemies, self.blue,
                               chase=tgt)
        return greedy_step(x, y, self.obs, self.d_flag, enemies, self.blue)


class CamperBot:
    """Walks to a fixed oasis cell and sits — the stall/deadlock trigger."""

    def __init__(self, board, pid):
        self.obs = obs_of(board)
        self.tx, self.ty = (16, 12) if pid % 2 == 0 else (12, 16)

    def move(self, state):
        n = [int(v) for v in state.split()]
        x, y = n[:2]
        if x < 0 or (x, y) == (self.tx, self.ty):
            return "s"
        best, bk = "s", 10 ** 9
        for mv, dx, dy in MOVES[:4]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                d = abs(nx - self.tx) + abs(ny - self.ty)
                if d < bk:
                    bk, best = d, mv
        return best


class ExeProc:
    def __init__(self, exe, board):
        self.p = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        self.p.stdin.write(board + "\n")
        self.p.stdin.flush()

    def move(self, state):
        self.p.stdin.write(state + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline().strip()
        return line if line in ("u", "d", "l", "r", "s") else None

    def close(self):
        try:
            self.p.stdin.close()
            self.p.kill()
        except Exception:
            pass


# each opponent team is a (slot-2 class, slot-3 class) pair
OPP = {
    "runner": (RunnerBot, RunnerBot),
    "hunter": (HunterBot, HunterBot),
    "guarded": (RunnerBot, GuardBot),   # the team type that beat us
    "camper": (CamperBot, CamperBot),
    "smart": (sim.SmartBot, sim.SmartBot),
}


def boards(n):
    import random
    out = []
    for g in range(n):
        if g % 3 == 0:
            out.append(sim.REAL_BOARD)
        else:
            out.append(sim.gen_board(random.Random(7000 + g)))
    return out


def run_matchup(exe, opp_name, board_list):
    tally = {"win": 0, "draw": 0, "loss": 0, "dq": 0}
    c2, c3 = OPP[opp_name]
    for board in board_list:
        bots = [ExeProc(exe, board), ExeProc(exe, board),
                c2(board, 2), c3(board, 3)]
        result = sim.run_game(board, bots)
        for b in bots:
            if isinstance(b, ExeProc):
                b.close()
        out = result[0]
        if out == "blue":
            tally["win"] += 1
        elif out == "red":
            tally["loss"] += 1
        elif out == "dq":
            tally["dq"] += 1
        else:
            tally["draw"] += 1
    return tally


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "submission", "capture_the_flag.exe")
    board_list = boards(15)
    print("gauntlet:", os.path.basename(exe))
    total = {"win": 0, "draw": 0, "loss": 0, "dq": 0}
    for name in ("runner", "hunter", "guarded", "camper", "smart"):
        t = run_matchup(exe, name, board_list)
        for k in total:
            total[k] += t[k]
        print("  vs %-8s W%-3d D%-3d L%-3d DQ%-2d"
              % (name, t["win"], t["draw"], t["loss"], t["dq"]))
    print("  %s" % ("-" * 30))
    print("  TOTAL    W%-3d D%-3d L%-3d DQ%-2d"
          % (total["win"], total["draw"], total["loss"], total["dq"]))
    score = total["win"] * 2 + total["draw"]
    print("  score: %d / %d" % (score, len(board_list) * len(OPP) * 2))


if __name__ == "__main__":
    main()
