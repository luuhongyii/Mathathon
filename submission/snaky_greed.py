import sys
from collections import deque

GRID = 32
DIRS = [('u', 0, -1), ('d', 0, 1), ('l', -1, 0), ('r', 1, 0)]


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

    CAP = 60

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
                val = steps + 0.05 * free_space(nx, ny)
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

    def claim_path(ox, oy, nx, ny):
        sx = (nx > ox) - (nx < ox)
        sy = (ny > oy) - (ny < oy)
        cx, cy = ox, oy
        while (cx, cy) != (nx, ny):
            cx += sx
            cy += sy
            if in_grid(cx, cy):
                claimed[cy][cx] = True

    _, d = search(mx, my, 6)
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
        _, d = search(mx, my, 6)
        sys.stdout.write(d + "\n")
        sys.stdout.flush()


main()
