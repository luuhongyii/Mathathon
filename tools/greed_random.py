import random
import sys

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
    mx, my, ex, ey = (int(v) for v in pos[:4])
    claimed = {(mx, my), (ex, ey)}

    def in_grid(x, y):
        return 0 <= x < GRID and 0 <= y < GRID

    def safe(x, y, dx, dy):
        ax, ay = x + dx, y + dy
        dist = grid[ay][ax] if in_grid(ax, ay) else 1
        for k in range(1, dist + 1):
            cx, cy = x + dx * k, y + dy * k
            if not in_grid(cx, cy) or (cx, cy) in claimed:
                return False
        return True

    def claim_path(ox, oy, nx, ny):
        sx = (nx > ox) - (nx < ox)
        sy = (ny > oy) - (ny < oy)
        cx, cy = ox, oy
        while (cx, cy) != (nx, ny):
            cx += sx
            cy += sy
            claimed.add((cx, cy))

    def choose(x, y):
        opts = [n for n, dx, dy in DIRS if safe(x, y, dx, dy)]
        return random.choice(opts) if opts else 'u'

    print(choose(mx, my), flush=True)
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
        claimed.add((mx, my))
        claimed.add((ex, ey))
        pm, pe = (mx, my), (ex, ey)
        print(choose(mx, my), flush=True)


main()
