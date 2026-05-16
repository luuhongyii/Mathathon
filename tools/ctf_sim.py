"""Capture the Flag local simulator - validates submission/capture_the_flag.py.

Runs the real game rules, driving bots over stdio exactly like the platform.
Usage:
    python tools/ctf_sim.py            # my bot (blue) vs random (red), many games
    python tools/ctf_sim.py mirror     # my bot vs my bot
    python tools/ctf_sim.py hard       # vs greedy runner + guard
    python tools/ctf_sim.py smart      # vs stronger rule bot (danger-aware)
"""
import os
import random
import subprocess
import sys

SIZE = 29
BOT = os.environ.get(
    "CTF_BOT",
    os.path.join(os.path.dirname(__file__), "..", "submission", "capture_the_flag.py"),
)


def in_oasis(x, y):
    return 12 <= x <= 16 and 12 <= y <= 16


def territory(x, y):
    if in_oasis(x, y):
        return "N"
    if y <= 13:
        return "B"
    if y >= 15:
        return "R"
    return "N"


REAL_BOARD = (
    ".............#...#.......#............#...#...#...#...#.##........."
    "#...........#.....#..###.###.#...#...#####.....#..............#...."
    ".............###.###...#...#............................#........"
    "......###.#####.#####.###.....##.........#...................##..."
    "....#.....###.###.###.#..........#.................#..#######.#.#."
    "......###.#.###............#...........#......###.#.#.#.#.....#.#."
    "#.###........#.#.#.#.....#.#.#.#........###.#.#.#.....#.#.#.#.###."
    ".....#...........#............###.#.###.......#.#.#######..#......"
    "...........#..........#.###.###.###.....#.......##................"
    "...#.........##.....###.#####.#####.###..............#..........."
    ".................#...#...###.###..................#.............."
    "#.....#####...#...#.###.###..#.....#...........#.........##.#...#."
    "..#...#...#............#.......#...#.............")


def _connected(board):
    obs = {(i % SIZE, i // SIZE) for i, c in enumerate(board) if c == "#"}
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        x, y = stack.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < SIZE and 0 <= ny < SIZE
                    and (nx, ny) not in obs and (nx, ny) not in seen):
                seen.add((nx, ny))
                stack.append((nx, ny))
    return all(c in seen for c in ((28, 28), (28, 0), (0, 28)))


def gen_board(rng, density=0.14):
    """Random obstacles, regenerated until all four corners are connected
    (the real platform board is dense but structured/connected)."""
    for _ in range(200):
        cells = ["."] * (SIZE * SIZE)
        for i in range(SIZE * SIZE):
            x, y = i % SIZE, i // SIZE
            if (x, y) in ((0, 0), (28, 28), (28, 0), (0, 28)):
                continue
            if in_oasis(x, y):
                continue
            if rng.random() < density:
                cells[i] = "#"
        board = "".join(cells)
        if _connected(board):
            return board
    return "." * (SIZE * SIZE)


class Proc:
    """One bot subprocess (one player)."""

    def __init__(self, board):
        cmd = BOT.split() if isinstance(BOT, str) and " " in BOT else [BOT]
        if BOT.endswith(".py"):
            cmd = [sys.executable, BOT]
        self.p = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
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


class RandomBot:
    """Safe-random opponent: never walks into a wall/edge."""

    def __init__(self, board):
        self.obs = {(i % SIZE, i // SIZE)
                    for i, c in enumerate(board) if c == "#"}

    def move(self, state):
        nums = [int(n) for n in state.split()]
        x, y = nums[0], nums[1]
        opts = []
        for mv, dx, dy in (("u", 0, -1), ("d", 0, 1), ("l", -1, 0),
                           ("r", 1, 0), ("s", 0, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                opts.append(mv)
        return random.choice(opts or ["s"])


class GreedyFieldBot:
    """Rule-based opponent for stress tests: one runner and one home guard."""

    def __init__(self, board, mode):
        self.obs = {(i % SIZE, i // SIZE)
                    for i, c in enumerate(board) if c == "#"}
        self.mode = mode
        self.adj = {}
        for y in range(SIZE):
            for x in range(SIZE):
                if (x, y) in self.obs:
                    continue
                ns = []
                for mv, dx, dy in (("u", 0, -1), ("d", 0, 1),
                                   ("l", -1, 0), ("r", 1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                        ns.append((mv, nx, ny))
                self.adj[(x, y)] = ns

    def _target(self, nums):
        x, y, _, carrying = nums[:4]
        blue = y <= 14
        if self.mode == "runner":
            if carrying:
                return (14, 13) if blue else (14, 15)
            return (28, 28) if blue else (0, 0)
        return (27, 27) if blue else (1, 1)

    def move(self, state):
        nums = [int(n) for n in state.split()]
        x, y = nums[0], nums[1]
        if x < 0:
            return "s"
        tx, ty = self._target(nums)
        best = ("s", abs(x - tx) + abs(y - ty), random.random())
        for mv, nx, ny in self.adj.get((x, y), ()):
            key = (mv, abs(nx - tx) + abs(ny - ty), random.random())
            if (key[1], key[2]) < (best[1], best[2]):
                best = key
        return best[0]


class SmartBot:
    """Stress opponent: BFS fields, 1-ply catch model, runner/guard split."""

    def __init__(self, board, player_id):
        self.pid = player_id
        self.blue = player_id < 2
        self.runner = (player_id % 2 == 0)
        self.obs = {(i % SIZE, i // SIZE)
                    for i, c in enumerate(board) if c == "#"}
        self.adj = {}
        self.free = {}
        for y in range(SIZE):
            for x in range(SIZE):
                if (x, y) in self.obs:
                    continue
                i = y * SIZE + x
                self.free[i] = True
                ns = []
                for mv, dx, dy in (("u", 0, -1), ("d", 0, 1),
                                   ("l", -1, 0), ("r", 1, 0)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                        ns.append((mv, nx, ny, ny * SIZE + nx))
                self.adj[i] = ns
        if self.blue:
            self.my_flag = 0
            self.enemy_flag = 28 * SIZE + 28
            self.my_terr, self.enemy_terr = "B", "R"
        else:
            self.my_flag = 28 * SIZE + 28
            self.enemy_flag = 0
            self.my_terr, self.enemy_terr = "R", "B"
        self.d_enemyflag = self._bfs([self.enemy_flag])
        self.d_myflag = self._bfs([self.my_flag])
        self.d_myterr = self._bfs(
            [i for i in self.free
             if territory(i % SIZE, i // SIZE) == self.my_terr])
        oasis = [i for i in self.free if in_oasis(i % SIZE, i // SIZE)]
        self.d_oasis = self._bfs(oasis)
        guard = []
        for i in self.free:
            x, y = i % SIZE, i // SIZE
            if self.my_terr == "B":
                if y in (12, 13) and 2 <= x <= 26:
                    guard.append(i)
            elif y in (15, 16) and 2 <= x <= 26:
                guard.append(i)
        self.d_guard = self._bfs(guard or [self.my_flag])

    def _bfs(self, sources):
        dist = {i: 10 ** 9 for i in self.free}
        q = []
        for s in sources:
            if s in dist and dist[s] == 10 ** 9:
                dist[s] = 0
                q.append(s)
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            nd = dist[c] + 1
            for _, _, _, nb in self.adj.get(c, ()):
                if dist[nb] > nd:
                    dist[nb] = nd
                    q.append(nb)
        return dist

    def _enemy_steps(self, ex, ey):
        yield ex, ey
        for _, dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = ex + dx, ey + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in self.obs:
                yield nx, ny

    def _caught_after(self, nx, ny, enemies):
        if territory(nx, ny) != self.enemy_terr:
            return False
        for ex, ey, _, _ in enemies:
            best = 99
            for px, py in self._enemy_steps(ex, ey):
                best = min(best, max(abs(px - nx), abs(py - ny)))
            if best <= 1:
                return True
        return False

    def _flag_guarded(self, my_i, enemies):
        if not enemies:
            return False
        my_d = self.d_enemyflag.get(my_i, 10 ** 9)
        eb = 10 ** 9
        for ex, ey, _, _ in enemies:
            eb = min(eb, self.d_enemyflag.get(ey * SIZE + ex, 10 ** 9))
        return eb <= my_d + 1

    def _unsafe(self, nx, ny, enemies, use_1ply):
        if use_1ply:
            return self._caught_after(nx, ny, enemies)
        if territory(nx, ny) != self.enemy_terr:
            return False
        for ex, ey, _, _ in enemies:
            if max(abs(ex - nx), abs(ey - ny)) <= 2:
                for px, py in self._enemy_steps(ex, ey):
                    if max(abs(px - nx), abs(py - ny)) <= 1:
                        return True
        return False

    def move(self, state):
        nums = [int(n) for n in state.split()]
        x, y, h, carrying = nums[:4]
        if x < 0:
            return "s"
        mate = nums[4:8]
        mate_alive = mate[0] >= 0
        mate_carry = mate_alive and mate[3] == 1
        enemies = []
        for off in (8, 12):
            ex, ey = nums[off], nums[off + 1]
            if ex >= 0:
                enemies.append((ex, ey, nums[off + 2], nums[off + 3]))
        my_i = y * SIZE + x
        enemy_carrier = next((e for e in enemies if e[3] == 1), None)
        use_1ply = bool(enemies) and (
            carrying or enemy_carrier is not None
            or territory(x, y) == self.enemy_terr)

        do = self.d_oasis.get(my_i, 10 ** 9)
        need_home = self.d_myterr.get(my_i, 10 ** 9)
        refilling = False
        if mate_alive and in_oasis(mate[0], mate[1]) and mate[2] >= 108 and h >= 75:
            refilling = False
        elif in_oasis(x, y):
            refilling = h < (100 if carrying else 112)
        elif carrying and do < 10 ** 8:
            refilling = h < need_home + 26
        elif self.runner and not carrying and do < 10 ** 8:
            nf = self.d_enemyflag.get(my_i, 10 ** 9)
            refilling = h < nf + need_home + 20

        guarded = (self.runner and not carrying
                   and self._flag_guarded(my_i, enemies))

        if refilling and not in_oasis(x, y):
            tgt = self.d_oasis
        elif guarded:
            tgt = self.d_oasis
        elif carrying:
            tgt = self.d_myterr
        elif enemy_carrier and not self.runner:
            tgt = self._bfs([enemy_carrier[1] * SIZE + enemy_carrier[0]])
        elif mate_carry and not self.runner:
            best = 10 ** 9
            blocker = None
            for e in enemies:
                d = self.d_myflag.get(e[1] * SIZE + e[0], 10 ** 9)
                if d < best:
                    best, blocker = d, e
            if blocker:
                tgt = self._bfs([blocker[1] * SIZE + blocker[0]])
            else:
                tgt = self.d_guard
        elif self.runner and not carrying:
            tgt = self.d_enemyflag
        elif enemies:
            best = 10 ** 9
            intr = None
            for e in enemies:
                if territory(e[0], e[1]) == self.my_terr:
                    d = self.d_myflag.get(e[1] * SIZE + e[0], 10 ** 9)
                    if d < best:
                        best, intr = d, e
            if intr is None:
                for e in enemies:
                    d = self.d_myflag.get(e[1] * SIZE + e[0], 10 ** 9)
                    if d < best:
                        best, intr = d, e
            tgt = (self._bfs([intr[1] * SIZE + intr[0]]) if intr
                   else self.d_guard)
        else:
            tgt = self.d_guard

        here_bad = self._unsafe(x, y, enemies, use_1ply)
        opts = []
        for mv, nx, ny, ni in self.adj.get(my_i, ()):
            d = tgt.get(ni, 10 ** 9)
            if d == 10 ** 9:
                d = 9000 + abs(nx - x) + abs(ny - y)
            bad = self._unsafe(nx, ny, enemies, use_1ply)
            safe = 0 if bad else 1
            if here_bad:
                mc = min(max(abs(ex - nx), abs(ey - ny)) for ex, ey, _, _ in enemies)
                opts.append((safe, mc, -d, random.random(), mv))
            else:
                opts.append((safe, -d, random.random(), mv))
        opts.append((1, 0, random.random(), "s") if not here_bad
                    else (1, 99, 0, random.random(), "s"))
        opts.sort(reverse=True)
        return opts[0][-1]


DELTA = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0), "s": (0, 0)}
START = [(0, 0), (0, 0), (28, 28), (28, 28)]
RESPAWN = [(28, 0), (28, 0), (0, 28), (0, 28)]
HOME = ["B", "B", "R", "R"]
FLAG_INIT = {"B": (0, 0), "R": (28, 28)}
MATE = {0: 1, 1: 0, 2: 3, 3: 2}
ENEMIES = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}


def run_game(board, bots, verbose=False):
    obs = {(i % SIZE, i // SIZE) for i, c in enumerate(board) if c == "#"}
    px = list(START)
    pos = [list(START[i]) for i in range(4)]
    hyd = [140] * 4
    carry = [False] * 4              # carrying the enemy flag
    alive = [True] * 4
    respawn = [0] * 4
    flag = {"B": [0, 0], "R": [28, 28]}   # current flag positions
    flag_at = {"B": None, "R": None}      # player idx carrying, or None

    def state_for(p):
        order = [p, MATE[p], ENEMIES[p][0], ENEMIES[p][1]]
        out = []
        for q in order:
            if alive[q]:
                out += [pos[q][0], pos[q][1], hyd[q], 1 if carry[q] else 0]
            else:
                out += [-1, -1, 0, 0]
        return " ".join(str(n) for n in out)

    for turn in range(512):
        moves = []
        for p in range(4):
            if not alive[p]:
                moves.append("s")
                continue
            mv = bots[p].move(state_for(p))
            if mv is None:
                return ("dq", p)            # invalid output = disqualified
            moves.append(mv)

        # apply movement
        for p in range(4):
            if not alive[p]:
                continue
            dx, dy = DELTA[moves[p]]
            nx, ny = pos[p][0] + dx, pos[p][1] + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and (nx, ny) not in obs:
                pos[p] = [nx, ny]

        # flag pickup (lower index wins ties)
        for team in ("B", "R"):
            if flag_at[team] is not None:
                continue
            fx, fy = flag[team]
            for p in range(4):
                if alive[p] and HOME[p] != team and not carry[p] \
                        and pos[p] == [fx, fy]:
                    flag_at[team] = p
                    carry[p] = True
                    break

        # hydration
        for p in range(4):
            if not alive[p]:
                continue
            x, y = pos[p]
            hyd[p] -= 2 if territory(x, y) == HOME[p] else 1
            if in_oasis(x, y):
                hyd[p] = 140

        # deaths
        dead = set()
        for p in range(4):
            if not alive[p]:
                continue
            if hyd[p] <= 0:
                dead.add(p)
                continue
            x, y = pos[p]
            enemy_terr = "R" if HOME[p] == "B" else "B"
            if territory(x, y) == enemy_terr:
                for q in range(4):
                    if alive[q] and HOME[q] != HOME[p]:
                        if max(abs(pos[q][0] - x), abs(pos[q][1] - y)) <= 1:
                            dead.add(p)
                            break
        for p in dead:
            alive[p] = False
            if carry[p]:
                carry[p] = False
                team = "R" if HOME[p] == "B" else "B"
                flag[team] = list(FLAG_INIT[team])
                flag_at[team] = None
            pos[p] = [-1, -1]
            hyd[p] = 0
            respawn[p] = 30

        # carried flags follow the carrier
        for team in ("B", "R"):
            if flag_at[team] is not None and alive[flag_at[team]]:
                flag[team] = list(pos[flag_at[team]])

        # win check
        winners = set()
        for p in range(4):
            if alive[p] and carry[p] and territory(*pos[p]) == HOME[p]:
                winners.add(HOME[p])
        if winners:
            if "B" in winners and "R" in winners:
                return ("draw", turn)
            return ("blue" if "B" in winners else "red", turn)

        # respawns
        for p in range(4):
            if not alive[p]:
                respawn[p] -= 1
                if respawn[p] <= 0:
                    alive[p] = True
                    pos[p] = list(RESPAWN[p])
                    hyd[p] = 140
                    carry[p] = False

    return ("draw", 512)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    mirror = mode == "mirror"
    hard = mode == "hard"
    smart = mode == "smart"
    games = 20
    tally = {"blue": 0, "red": 0, "draw": 0, "dq": 0}
    for g in range(games):
        rng = random.Random(1000 + g)
        random.seed(2000 + g)
        board = REAL_BOARD if g % 4 == 0 else gen_board(rng)
        bots = []
        for p in range(4):
            if p < 2 or mirror:
                bots.append(Proc(board))
            elif smart:
                bots.append(SmartBot(board, p))
            elif hard:
                bots.append(GreedyFieldBot(board, "guard" if p == 2 else "runner"))
            else:
                bots.append(RandomBot(board))
        result = run_game(board, bots)
        for b in bots:
            if isinstance(b, Proc):
                b.close()
        outcome = result[0]
        tally[outcome] = tally.get(outcome, 0) + 1
        print("game %2d: %-6s %s" % (g, outcome, result[1]))
    print("-" * 30)
    print("tally:", tally)


if __name__ == "__main__":
    main()
