"""Debug a single greed_bench seed: dump the board + both trails at death."""
import random
import subprocess
import sys
from pathlib import Path

GRID = 32
DIRS = {'u': (0, -1), 'd': (0, 1), 'l': (-1, 0), 'r': (1, 0)}
STARTS = [(8, 8), (23, 23)]
ROOT = Path(__file__).resolve().parents[1]
EX = sys.executable
RND = [EX, str(ROOT / "tools" / "greed_random.py")]
BOT = [EX, str(ROOT / "submission" / "snaky_greed.py")]

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 23
# bench: even seed -> bot is slot 0
bot_slot = 0 if seed % 2 == 0 else 1
cmd0, cmd1 = (BOT, RND) if bot_slot == 0 else (RND, BOT)

rng = random.Random(seed)
grid = [[rng.randint(1, 9) for _ in range(GRID)] for _ in range(GRID)]
gl = " ".join(str(grid[y][x]) for y in range(GRID) for x in range(GRID))
c0 = cmd0 + [str(seed)]
c1 = cmd1 + [str(seed)]
procs = [subprocess.Popen(c, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1) for c in (c0, c1)]
pos = [list(STARTS[0]), list(STARTS[1])]
claimed = {STARTS[0]: 0, STARTS[1]: 1}
score = [1, 1]
alive = [True, True]
history = []


def send(p, s):
    procs[p].stdin.write(s + "\n"); procs[p].stdin.flush()


def recv(p):
    line = procs[p].stdout.readline()
    return line.strip() if line else ""


rounds = 0
death_round = [None, None]
while rounds < 4000 and (alive[0] or alive[1]):
    for p in (0, 1):
        if not alive[p]:
            continue
        o = 1 - p
        if rounds == 0:
            send(p, gl)
        send(p, f"{pos[p][0]} {pos[p][1]} {pos[o][0]} {pos[o][1]}")
    moves = [recv(p) if alive[p] else None for p in (0, 1)]
    history.append((rounds, [tuple(pos[0]), tuple(pos[1])], list(moves)))
    paths = [[], []]
    for p in (0, 1):
        if not alive[p] or moves[p] not in DIRS:
            continue
        dx, dy = DIRS[moves[p]]
        ax, ay = pos[p][0] + dx, pos[p][1] + dy
        dist = grid[ay][ax] if 0 <= ax < GRID and 0 <= ay < GRID else 1
        for k in range(1, dist + 1):
            paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))
    maxlen = max(len(paths[0]), len(paths[1]))
    done = [not (alive[p] and moves[p] in DIRS) for p in (0, 1)]
    for p in (0, 1):
        if alive[p] and moves[p] not in DIRS:
            alive[p] = False
            death_round[p] = rounds
    for k in range(maxlen):
        cell = {}
        for p in (0, 1):
            if done[p]:
                continue
            if k >= len(paths[p]):
                done[p] = True
                continue
            cell[p] = paths[p][k]
        deaths = set()
        for p, c in cell.items():
            cx, cy = c
            if not (0 <= cx < GRID and 0 <= cy < GRID) or c in claimed:
                deaths.add(p)
        if len(cell) == 2 and cell[0] == cell[1]:
            deaths.update((0, 1))
        for p, c in cell.items():
            if p in deaths:
                alive[p] = False
                done[p] = True
                if death_round[p] is None:
                    death_round[p] = rounds
            else:
                claimed[c] = p
                pos[p] = list(c)
                score[p] += 1
        if all(done):
            break
    rounds += 1

for p in procs:
    try:
        p.stdin.close(); p.wait(timeout=2)
    except Exception:
        p.kill()

print(f"seed {seed}  bot_slot={bot_slot}  score={score}  death_round={death_round}  rounds={rounds}")
print("last moves:")
for r, ps, mv in history[-int(sys.argv[2]) if len(sys.argv) > 2 else -6:]:
    print(f"  r{r:3d}: pos={ps} moves={mv}")

# board dump
sym = {None: '.'}
g = [['.'] * GRID for _ in range(GRID)]
for (x, y), p in claimed.items():
    g[y][x] = 'A' if p == 0 else 'B'
for p in (0, 1):
    x, y = pos[p]
    g[y][x] = '0' if p == 0 else '1'
print("  (A/0 = slot0, B/1 = slot1, bot is slot", bot_slot, ")")
for row in g:
    print(' ' + ''.join(row))
