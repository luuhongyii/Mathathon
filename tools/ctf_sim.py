"""Capture the Flag local simulator - validates submission/capture_the_flag.py.

Runs the real game rules, driving bots over stdio exactly like the platform.
Usage:
    python tools/ctf_sim.py            # my bot (blue) vs random (red), many games
    python tools/ctf_sim.py mirror     # my bot vs my bot
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
    mirror = len(sys.argv) > 1 and sys.argv[1] == "mirror"
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
