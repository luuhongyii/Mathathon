import os
import sys
from collections import deque

GRID = 32
DIRS = [('u', 0, -1), ('d', 0, 1), ('l', -1, 0), ('r', 1, 0)]

# Leaf room signal. The depth-6 search maximises total claimed steps; without
# a real room term it happily scores its way into a self-enclosure (drawing a
# loop with its own trail) that kills it just past the search horizon. The
# leaf flood-fill estimates free space reachable after 6 plies -- it must be
# (a) capped high enough to tell "boxed into 30 cells" from "open board" and
# (b) weighted enough to outweigh the few extra points a trap line promises.
ROOM_CAP = int(os.environ.get("SG_ROOMCAP", "400"))
ROOM_W = float(os.environ.get("SG_ROOMW", "0.3"))

# Survival horizon extension. The depth-6 search only rejects death within 6
# plies; a self-enclosure (looping our own trail) often seals one or two
# plies later and is invisible to both the search and the flood-fill (the
# loop is still open at the leaf). At each leaf we run a cheap greedy
# playout: if it cannot survive SURV_CAP more plies, the leaf is walking into
# a trap and is penalised hard -- survival outranks the few points the trap
# line promised.
SURV_CAP = int(os.environ.get("SG_SURVCAP", "12"))
SURV_PEN = float(os.environ.get("SG_SURVPEN", "8.0"))

# Penalty for a depth-6 leaf cell with <=1 legal move (one forced jump from
# death). Catches corner traps the survival playout wiggles out of.
MOB_PEN = float(os.environ.get("SG_MOBPEN", "30.0"))


def main():
    first = sys.stdin.readline()
    if not first:
        return
    digits = first.split()
    grid = [[1] * GRID for _ in range(GRID)]
    for idx in range(min(len(digits), GRID * GRID)):
        grid[idx // GRID][idx % GRID] = int(digits[idx])

    pos = sys.stdin.readline().split()
    while len(pos) < 4:
        extra = sys.stdin.readline()
        if not extra:
            return
        pos += extra.split()
    mx, my, ex, ey = (int(v) for v in pos[:4])

    claimed = [[False] * GRID for _ in range(GRID)]
    claimed[my][mx] = True
    claimed[ey][ex] = True

    # reusable stamped visited grid for flood fill
    visited = [[0] * GRID for _ in range(GRID)]
    stamp = [0]

    def in_grid(x, y):
        return 0 <= x < GRID and 0 <= y < GRID

    def simulate(x, y, dx, dy):
        """Return (alive, steps, nx, ny, path) for moving from (x,y) in dir."""
        ax, ay = x + dx, y + dy
        dist = grid[ay][ax] if in_grid(ax, ay) else 1
        path = []
        for k in range(1, dist + 1):
            cx, cy = x + dx * k, y + dy * k
            if not in_grid(cx, cy) or claimed[cy][cx]:
                return (False, k - 1, x + dx * (k - 1), y + dy * (k - 1), path)
            path.append((cx, cy))
        return (True, dist, x + dx * dist, y + dy * dist, path)

    CAP = ROOM_CAP

    def free_space(x, y):
        stamp[0] += 1
        s = stamp[0]
        seen = 0
        q = deque()
        for _, dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if in_grid(nx, ny) and not claimed[ny][nx] and visited[ny][nx] != s:
                visited[ny][nx] = s
                q.append((nx, ny))
        while q and seen < CAP:
            cx, cy = q.popleft()
            seen += 1
            for _, dx, dy in DIRS:
                nx, ny = cx + dx, cy + dy
                if in_grid(nx, ny) and not claimed[ny][nx] and visited[ny][nx] != s:
                    visited[ny][nx] = s
                    q.append((nx, ny))
        return seen

    def jump_space(x, y):
        """Jump-aware reachable-cell count from (x,y), capped at CAP.

        The plain flood-fill above counts a cell as reachable through 1-step
        adjacency, but in this game the jump distance is FORCED by the digit:
        a 30-cell pocket whose digits all overshoot its walls is not 30 cells
        of room, it is a death trap. This BFS walks the real move graph --
        every node is a cell you can actually land on -- so a corner region
        the bot cannot navigate without crashing reads as the dead end it is.
        Path cells are treated as free (an optimistic but jump-honest count)."""
        stamp[0] += 1
        s = stamp[0]
        seen = 0
        q = deque([(x, y)])
        visited[y][x] = s
        while q and seen < CAP:
            cx, cy = q.popleft()
            seen += 1
            for _, dx, dy in DIRS:
                alive, steps, nx, ny, _ = simulate(cx, cy, dx, dy)
                if alive and steps > 0 and visited[ny][nx] != s:
                    visited[ny][nx] = s
                    q.append((nx, ny))
        return seen

    def mobility(x, y):
        """Number of directions from (x,y) that are a legal, non-zero move.
        In this game the jump distance is forced by the digit, so near a wall
        a large digit overshoots the board -> that direction is lethal. A cell
        with mobility <=1 is one forced move from death."""
        m = 0
        for _, dx, dy in DIRS:
            alive, steps, _, _, _ = simulate(x, y, dx, dy)
            if alive and steps > 0:
                m += 1
        return m

    def surv_run(x, y, cap):
        """Greedy (max-steps) survival playout from (x,y). Returns
        (plies_survived capped at `cap`, free space around the END cell).

        The ply count alone misses corner traps -- a snake can wiggle a full
        cap-length playout inside a small pocket and still be doomed. The
        end-cell free space catches that: 'survived 12 plies but ended boxed
        into 20 cells' is a trap. This pushes the room estimate ~SURV_CAP
        plies past the depth-6 search leaf."""
        marked = []
        cx, cy = x, y
        plies = 0
        while plies < cap:
            best = None
            for _, dx, dy in DIRS:
                alive, steps, nx, ny, path = simulate(cx, cy, dx, dy)
                if alive and steps > 0 and (best is None or steps > best[0]):
                    best = (steps, nx, ny, path)
            if best is None:
                break
            for px, py in best[3]:
                claimed[py][px] = True
                marked.append((px, py))
            cx, cy = best[1], best[2]
            plies += 1
        end_free = free_space(cx, cy)
        for px, py in marked:
            claimed[py][px] = False
        return plies, end_free

    def survival_depth(x, y, cap):
        """Exact root-level escape check.

        The leaf playout is greedy; at the root we can afford a small DFS to
        reject moves that land in a short forced-death corridor behind our own
        trail.
        """
        if cap <= 0:
            return 0
        best = 0
        for _, dx, dy in DIRS:
            alive, steps, nx, ny, path = simulate(x, y, dx, dy)
            if not alive or steps <= 0:
                continue
            for px, py in path:
                claimed[py][px] = True
            d = 1 + survival_depth(nx, ny, cap - 1)
            for px, py in path:
                claimed[py][px] = False
            if d > best:
                best = d
            if best >= cap:
                break
        return best

    def search(x, y, depth):
        """Return (value, direction). Maximizes claimed steps; survival first."""
        best_val = None
        best_dir = None
        any_alive = False
        for name, dx, dy in DIRS:
            alive, steps, nx, ny, path = simulate(x, y, dx, dy)
            if not alive:
                continue
            any_alive = True
            for px, py in path:
                claimed[py][px] = True
            if depth > 1:
                sub_val, _ = search(nx, ny, depth - 1)
                val = steps + sub_val
            else:
                # Room is measured at the END of the survival playout
                # (~SURV_CAP plies on), not at the leaf -- that is what
                # exposes a slowly-closing self-enclosure.
                run, _ = surv_run(nx, ny, SURV_CAP)
                # Room is the jump-aware reachable count: it already reads a
                # corner trap as a dead end, so no separate mobility term is
                # needed.
                val = steps + ROOM_W * jump_space(nx, ny)
                if run < SURV_CAP:
                    val -= (SURV_CAP - run) * SURV_PEN
            for px, py in path:
                claimed[py][px] = False
            if best_val is None or val > best_val:
                best_val, best_dir = val, name
        if not any_alive:
            # every direction is lethal: grab the most points before dying
            for name, dx, dy in DIRS:
                _, steps, _, _, _ = simulate(x, y, dx, dy)
                if best_val is None or steps > best_val:
                    best_val, best_dir = steps, name
            return (best_val - 1000.0, best_dir)
        return (best_val, best_dir)

    def enemy_reach(ox, oy):
        """Cells the opponent can move into this round -> min opponent-steps
        to reach each. Used to avoid first moves whose jump path the opponent
        can claim first (a head-on collision is instant death)."""
        danger = {}
        for _, dx, dy in DIRS:
            ax, ay = ox + dx, oy + dy
            dist = grid[ay][ax] if in_grid(ax, ay) else 1
            for k in range(1, dist + 1):
                cx, cy = ox + dx * k, oy + dy * k
                if not in_grid(cx, cy) or claimed[cy][cx]:
                    break
                if (cx, cy) not in danger or k < danger[(cx, cy)]:
                    danger[(cx, cy)] = k
        return danger

    def decide(mx, my, ex, ey):
        """Root move choice: depth-6 search value, then drop first moves whose
        jump path crosses a cell the opponent reaches first (would collide)."""
        danger = enemy_reach(ex, ey)
        cands = []          # (value, name, collide)
        forced = []         # (steps, name) -- only if every move is lethal
        for name, dx, dy in DIRS:
            alive, steps, nx, ny, path = simulate(mx, my, dx, dy)
            forced.append((steps, name))
            if not alive:
                continue
            for px, py in path:
                claimed[py][px] = True
            sub, _ = search(nx, ny, 5)
            for px, py in path:
                claimed[py][px] = False
            val = steps + sub
            surv = survival_depth(nx, ny, 8)
            if surv < 8:
                val -= (8 - surv) * 120.0
            collide = False
            for i, c in enumerate(path, start=1):
                d = danger.get(c)
                if d is not None and d <= i:
                    collide = True
                    break
            cands.append((val, name, collide, surv))
        if not cands:
            return max(forced)[1]
        clean = [c for c in cands if not c[2]]
        pool = clean if clean else cands
        return max(pool)[1]

    def claim_path(ox, oy, nx, ny):
        sx = (nx > ox) - (nx < ox)
        sy = (ny > oy) - (ny < oy)
        cx, cy = ox, oy
        while (cx, cy) != (nx, ny):
            cx += sx
            cy += sy
            if in_grid(cx, cy):
                claimed[cy][cx] = True

    d = decide(mx, my, ex, ey)
    sys.stdout.write(d + "\n")
    sys.stdout.flush()

    pm, pe = (mx, my), (ex, ey)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split()
        if len(parts) < 4:
            break
        mx, my, ex, ey = (int(v) for v in parts[:4])
        claim_path(pm[0], pm[1], mx, my)
        claim_path(pe[0], pe[1], ex, ey)
        if in_grid(mx, my):
            claimed[my][mx] = True
        if in_grid(ex, ey):
            claimed[ey][ex] = True
        pm, pe = (mx, my), (ex, ey)
        d = decide(mx, my, ex, ey)
        sys.stdout.write(d + "\n")
        sys.stdout.flush()


main()
