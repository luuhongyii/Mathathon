"""Simulate Territory Wars to check the bot survives and claims a lot of
territory against simple opponents, and never makes an illegal move."""

import random
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))

from territory_wars import TerritoryBot, SIZE  # noqa: E402

# Standard mapping the simulator uses to apply emitted letters.
LETTER = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
CORNERS = [(0, 0), (SIZE - 1, 0), (0, SIZE - 1), (SIZE - 1, SIZE - 1)]


def safe_moves(pos, wall):
    out = []
    for letter, (dx, dy) in LETTER.items():
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < SIZE and 0 <= ny < SIZE and not wall[ny][nx]:
            out.append((letter, nx, ny))
    return out


def greedy_opponent(idx, positions, wall, rng):
    """Pick the safe move with the most reachable free space (flood fill)."""
    moves = safe_moves(positions[idx], wall)
    if not moves:
        return "u"
    best, best_room = None, -1
    for letter, nx, ny in moves:
        seen = {(nx, ny)}
        q = deque([(nx, ny)])
        while q:
            x, y = q.popleft()
            for ex, ey in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if (0 <= ex < SIZE and 0 <= ey < SIZE
                        and not wall[ey][ex] and (ex, ey) not in seen):
                    seen.add((ex, ey))
                    q.append((ex, ey))
        if len(seen) > best_room:
            best_room, best = len(seen), letter
    return best


def random_opponent(idx, positions, wall, rng):
    moves = safe_moves(positions[idx], wall)
    return rng.choice(moves)[0] if moves else "u"


def simulate(opponents, seed=0, max_turns=2000):
    """opponents: list of 3 strategy fns. Player 0 is our TerritoryBot."""
    rng = random.Random(seed)
    wall = [[False] * SIZE for _ in range(SIZE)]
    heads = list(CORNERS)
    alive = [True, True, True, True]
    claimed = [1, 1, 1, 1]
    for (x, y) in heads:
        wall[y][x] = True
    bot = TerritoryBot()

    illegal = []
    for _turn in range(max_turns):
        if sum(alive) <= 1:
            break
        targets = [None, None, None, None]
        for i in range(4):
            if not alive[i]:
                continue
            ordered = [heads[i]] + [heads[j] for j in range(4) if j != i]
            if i == 0:
                bot.observe(ordered)
                letter = bot.decide(ordered)
            else:
                letter = opponents[i - 1](0, ordered, wall, rng)
            dx, dy = LETTER.get(letter, (0, -1))
            targets[i] = (heads[i][0] + dx, heads[i][1] + dy)

        # Did our bot still have a safe move this turn? (to tell an avoidable
        # blunder apart from an unavoidable trapped death)
        bot_had_escape = bool(safe_moves(heads[0], wall)) if alive[0] else False

        # Resolve simultaneously: off-board / into a wall / head-on collision.
        for i in range(4):
            if not alive[i]:
                continue
            tx, ty = targets[i]
            if not (0 <= tx < SIZE and 0 <= ty < SIZE) or wall[ty][tx]:
                alive[i] = False
                if i == 0 and bot_had_escape:
                    illegal.append(("blunder", targets[i], _turn))
        for i in range(4):
            if not alive[i]:
                continue
            if sum(1 for j in range(4)
                   if alive[j] and targets[j] == targets[i]) > 1:
                alive[i] = False
        for i in range(4):
            if alive[i]:
                tx, ty = targets[i]
                wall[ty][tx] = True
                heads[i] = (tx, ty)
                claimed[i] += 1

    return claimed, alive, illegal


def test_territory_bot():
    # vs 3 greedy opponents
    for seed in range(4):
        claimed, alive, illegal = simulate(
            [greedy_opponent] * 3, seed=seed)
        assert not illegal, f"bot made an illegal move: {illegal}"
        print(f"vs greedy   seed={seed}  claimed={claimed}  "
              f"alive={alive}  -> our cells={claimed[0]}")
        assert claimed[0] >= 30, "bot should claim a real chunk of territory"

    # vs 3 random opponents
    for seed in range(4):
        claimed, alive, illegal = simulate(
            [random_opponent] * 3, seed=seed)
        assert not illegal, f"bot made an illegal move: {illegal}"
        print(f"vs random   seed={seed}  claimed={claimed}  alive={alive}")
        assert claimed[0] >= 30

    # mixed field
    claimed, alive, illegal = simulate(
        [greedy_opponent, random_opponent, greedy_opponent], seed=7)
    assert not illegal
    print(f"vs mixed    seed=7  claimed={claimed}  alive={alive}")


if __name__ == "__main__":
    test_territory_bot()
    print("OK")
