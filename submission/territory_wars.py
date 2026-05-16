"""Territory Wars submission - paste this whole file into the Mathathon editor.

Game (Tron / light-cycle): 31x31 board, 4 players from the corners. Each turn
every player steps u/d/l/r and leaves a permanent trail. Entering any claimed
cell or leaving the board kills you. Score = cells claimed.

The judge only sends the four head positions each round - never the trail map -
so we accumulate it ourselves: every cell a head has ever occupied is a wall.

Strategy - adversarial, aggressive territory control:
  * Identify the nearest reachable rival and run an iterative-deepening
    maximin search against it (I move -> rival's worst reply -> ...), with
    alpha-beta pruning, under a per-move time budget. Other opponents are
    treated as static obstacles.
  * Leaf eval = my Voronoi territory - AGGR * rival's Voronoi territory
    (multi-source BFS). The negative term makes the search actively pick
    moves that cut the rival's space, not just grow mine.
  * A rational rival never suicides, so the search ignores opponent moves
    that would kill the opponent (collision into me / running into a wall).
  * Once sealed off from every opponent, switch to pure space-filling with
    a Warnsdorff wall-hug so the whole region gets claimed tightly.

Robustness: the u/d/l/r -> (dx,dy) map is self-calibrated from observed
moves (handles a flipped y-axis); round 1 plays a guaranteed-safe horizontal
move. Never writes to stderr (the platform treats stderr as a forfeit).
"""

import sys
import time
from collections import deque

SIZE = 31
INF = 1 << 30
BIG = 10 ** 6                       # win/lose magnitude, dwarfs any eval
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPP_LETTER = {"u": "d", "d": "u", "l": "r", "r": "l"}

MAX_DEPTH = 10                      # ceiling on iterative-deepening plies
MOVE_BUDGET = 0.07                  # wall-clock seconds/move (kept < any limit)
AGGR = 0.6                          # weight on shrinking the rival's space
SURV = 0.02                         # mild nudge to keep my own room open
COLLIDE_W = 10.0                    # penalty for a same-turn collision risk


class TerritoryBot:
    def __init__(self):
        self.wall = [[False] * SIZE for _ in range(SIZE)]
        # Standard screen mapping; corrected from observed moves at runtime.
        self.delta = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
        self.prev_me = None
        self.last_move = None
        self.prev_all = None
        self.dead = [False, False, False, False]
        self.round = 0
        self._others_dist = None     # BFS grid for non-rival opponents
        self._deadline = 0.0         # iterative-deepening wall-clock cutoff
        self._timed_out = False

    # -- state update ------------------------------------------------------
    def observe(self, positions):
        self.round += 1
        for (x, y) in positions:
            if 0 <= x < SIZE and 0 <= y < SIZE:
                self.wall[y][x] = True
        # Calibrate direction -> delta from the move we actually made.
        me = positions[0]
        if self.prev_me is not None and self.last_move is not None:
            d = (me[0] - self.prev_me[0], me[1] - self.prev_me[1])
            if d != (0, 0):
                self.delta[self.last_move] = d
                self.delta[OPP_LETTER[self.last_move]] = (-d[0], -d[1])
        # A live player must move every round; an unchanged head means death.
        if self.prev_all is not None:
            for i in range(4):
                if positions[i] == self.prev_all[i]:
                    self.dead[i] = True
                x, y = positions[i]
                if not (0 <= x < SIZE and 0 <= y < SIZE):
                    self.dead[i] = True
        self.prev_all = list(positions)

    # -- decision ----------------------------------------------------------
    def decide(self, positions):
        me = positions[0]
        # Round 1: a guaranteed-safe inward horizontal move. r/l are certain,
        # so this never gambles on the (unknown until observed) y-axis.
        if self.round <= 1:
            return self._emit(me, (1, 0) if me[0] == 0 else (-1, 0))

        my_moves = self._legal(me, frozenset())
        if not my_moves:
            return self._emit(me, (1, 0))         # boxed in - doomed anyway

        opps = [positions[i] for i in range(1, 4)
                if not self.dead[i]
                and 0 <= positions[i][0] < SIZE
                and 0 <= positions[i][1] < SIZE]

        # Who can still reach me? Pick the closest as the search rival.
        reach = self._bfs([me], frozenset())
        rival, rdist = None, INF
        for o in opps:
            d = min((reach[oy][ox] for ox, oy in self._nbrs(o)),
                    default=INF)
            if d < rdist:
                rival, rdist = o, d

        # Sealed off from every opponent -> just fill our region tightly.
        if rival is None:
            return self._emit(me, self._fill_move(me, my_moves))

        others = [o for o in opps if o != rival]
        self._others_dist = self._bfs(others, frozenset()) if others else None

        # Candidate first moves (never initiate a collision into the rival).
        roots = [mv for mv in my_moves
                 if (me[0] + mv[0], me[1] + mv[1]) != rival]
        if not roots:
            return self._emit(me, my_moves[0])    # only move is suicide
        # Other (un-searched) opponents may step onto a cell this same turn.
        penalty = {mv: COLLIDE_W * self._collision_risk(
                       (me[0] + mv[0], me[1] + mv[1]), others)
                   for mv in roots}

        # Iterative deepening under a wall-clock budget: keep the best move
        # from the last fully-completed depth, so we never overrun the limit.
        # Each depth searches the previous depth's best move first, which
        # sharpens alpha-beta pruning and lets us reach deeper in the budget.
        self._deadline = time.perf_counter() + MOVE_BUDGET
        order = {mv: 0.0 for mv in roots}
        best = roots[0]
        for depth in range(2, MAX_DEPTH + 1):
            self._timed_out = False
            cur, cur_score = None, -INF
            for mv in sorted(roots, key=order.get, reverse=True):
                np = (me[0] + mv[0], me[1] + mv[1])
                # raw - penalty must beat cur_score, so alpha = cur + penalty.
                alpha = -INF if cur is None else cur_score + penalty[mv]
                score = self._search(np, rival, frozenset([np]),
                                     depth - 1, True, alpha, INF)
                if self._timed_out:
                    break
                score -= penalty[mv]
                order[mv] = score
                if score > cur_score:
                    cur_score, cur = score, mv
            if self._timed_out:
                break                              # discard partial depth
            best = cur
            if time.perf_counter() > self._deadline:
                break
        return self._emit(me, best)

    # -- adversarial search ------------------------------------------------
    def _search(self, my_pos, opp_pos, extra, depth, opp_to_move, alpha, beta):
        if self._timed_out or time.perf_counter() > self._deadline:
            self._timed_out = True
            return 0.0                            # discarded by the caller
        if depth <= 0:
            return self._eval(my_pos, opp_pos, extra)

        if opp_to_move:
            # Minimiser. A rational rival never walks into a wall or suicides
            # into my head, so those moves are dropped (treated as us winning).
            best = INF
            moved = False
            for (dx, dy) in self._legal(opp_pos, extra):
                op = (opp_pos[0] + dx, opp_pos[1] + dy)
                if op == my_pos:
                    continue                      # mutual death - rival avoids
                moved = True
                v = self._search(my_pos, op, extra | {op},
                                 depth - 1, False, alpha, beta)
                if v < best:
                    best = v
                if best < beta:
                    beta = best
                if beta <= alpha:
                    break
            return BIG if not moved else best     # rival trapped -> great
        else:
            # Maximiser (me).
            best = -INF
            moved = False
            for (dx, dy) in self._legal(my_pos, extra):
                np = (my_pos[0] + dx, my_pos[1] + dy)
                if np == opp_pos:
                    continue                      # I'd die in the collision
                moved = True
                v = self._search(np, opp_pos, extra | {np},
                                 depth - 1, True, alpha, beta)
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if beta <= alpha:
                    break
            return -BIG if not moved else best     # I'm trapped -> disaster

    def _eval(self, my_pos, opp_pos, extra):
        """Voronoi territory differential at a leaf, plus a survival nudge."""
        md = self._bfs([my_pos], extra)
        od = self._bfs([opp_pos], extra)
        xd = self._others_dist
        mine = rival = my_room = 0
        for y in range(SIZE):
            mr, orr = md[y], od[y]
            xr = xd[y] if xd else None
            for x in range(SIZE):
                m = mr[x]
                if m < INF:
                    my_room += 1
                o = orr[x]
                xv = xr[x] if xr is not None else INF
                if m < o and m < xv and m < INF:
                    mine += 1
                elif o < m and o < xv and o < INF:
                    rival += 1
        return mine - AGGR * rival + SURV * my_room

    # -- endgame space filling --------------------------------------------
    def _fill_move(self, me, my_moves):
        """No opponent can reach us: claim our whole pocket. Maximise the
        space still reachable, breaking ties by hugging walls (Warnsdorff)
        so no cell gets stranded."""
        best, best_score = my_moves[0], -INF
        for mv in my_moves:
            np = (me[0] + mv[0], me[1] + mv[1])
            room = sum(row.count(True)  # reachable cells from np
                       for row in self._reach_mask(np))
            score = room * 1000 - self._exits(np[0], np[1])
            if score > best_score:
                best_score, best = score, mv
        return best

    def _reach_mask(self, start):
        seen = [[False] * SIZE for _ in range(SIZE)]
        q = deque([start])
        seen[start[1]][start[0]] = True
        while q:
            x, y = q.popleft()
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < SIZE and 0 <= ny < SIZE
                        and not seen[ny][nx] and not self.wall[ny][nx]):
                    seen[ny][nx] = True
                    q.append((nx, ny))
        return seen

    # -- helpers -----------------------------------------------------------
    def _legal(self, pos, extra):
        out = []
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (0 <= nx < SIZE and 0 <= ny < SIZE
                    and not self.wall[ny][nx] and (nx, ny) not in extra):
                out.append((dx, dy))
        return out

    def _nbrs(self, pos):
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                yield nx, ny

    def _exits(self, x, y):
        n = 0
        for nx, ny in self._nbrs((x, y)):
            if not self.wall[ny][nx]:
                n += 1
        return n

    def _collision_risk(self, cell, opps):
        """Expected number of opponents stepping onto `cell` this turn,
        assuming each picks uniformly among its free exits."""
        risk = 0.0
        for (ox, oy) in opps:
            if abs(ox - cell[0]) + abs(oy - cell[1]) == 1:
                k = self._exits(ox, oy)
                if k:
                    risk += 1.0 / k
        return risk

    def _bfs(self, sources, extra):
        dist = [[INF] * SIZE for _ in range(SIZE)]
        q = deque()
        for (x, y) in sources:
            if 0 <= x < SIZE and 0 <= y < SIZE and dist[y][x] == INF:
                dist[y][x] = 0
                q.append((x, y))
        while q:
            x, y = q.popleft()
            nd = dist[y][x] + 1
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if (0 <= nx < SIZE and 0 <= ny < SIZE
                        and dist[ny][nx] == INF and not self.wall[ny][nx]
                        and (nx, ny) not in extra):
                    dist[ny][nx] = nd
                    q.append((nx, ny))
        return dist

    def _emit(self, me, choice):
        self.prev_me = me
        inv = {v: k for k, v in self.delta.items()}
        letter = inv.get(choice) or {(1, 0): "r", (-1, 0): "l",
                                     (0, 1): "d", (0, -1): "u"}[choice]
        self.last_move = letter
        return letter


def main():
    bot = TerritoryBot()
    for raw in sys.stdin:
        nums = []
        for tok in raw.split():
            try:
                nums.append(int(tok))
            except ValueError:
                pass
        if len(nums) < 8:
            continue
        positions = [(nums[2 * i], nums[2 * i + 1]) for i in range(4)]
        bot.observe(positions)
        sys.stdout.write(bot.decide(positions) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
