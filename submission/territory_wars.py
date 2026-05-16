"""Territory Wars submission - paste this whole file into the Mathathon editor.

Game (Tron / light-cycle): 31x31 board, 4 players from the corners. Each turn
every player steps u/d/l/r and leaves a permanent trail; entering any claimed
cell or the edge kills you. Score = cells claimed.

HARD CONSTRAINT: the platform allows only ~500ms of compute for the WHOLE
game (~512 turns) - about 1ms per move. So this is a fast 1-ply greedy bot,
NO game-tree search:

  * For each safe step, one multi-source BFS per move gives
      - Voronoi territory: cells we reach before any opponent,
      - room: free cells still reachable (anti self-trap).
  * Survival gate: prefer collision-free moves that keep real room - never
    gamble into a collision, never flee one into a dead end.
  * Tie-break by hugging walls (Warnsdorff) to fill space tightly.
  * Time governor: tracks cumulative compute and, if the budget runs low,
    drops to an instant safe-move heuristic - so the bot NEVER times out,
    even on a machine slower than the one it was tuned on.

Robustness: u/d/l/r -> (dx,dy) is self-calibrated from observed moves; round
one plays a guaranteed-safe horizontal move. Never writes to stderr (the
platform treats stderr as a forfeit).
"""

import sys
import time

SIZE = 31
N = SIZE * SIZE
INF = 1 << 29
TOTAL_BUDGET = 0.40                 # compute seconds for the whole game
MOVES = (("u", 0, -1), ("d", 0, 1), ("l", -1, 0), ("r", 1, 0))
OPP = {"u": "d", "d": "u", "l": "r", "r": "l"}
LETTER = {(1, 0): "r", (-1, 0): "l", (0, 1): "d", (0, -1): "u"}

# Precomputed 4-neighbour index lists for every cell (flat y*SIZE+x indexing).
ADJ = []
for _y in range(SIZE):
    for _x in range(SIZE):
        _nb = []
        for _dx, _dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _nx, _ny = _x + _dx, _y + _dy
            if 0 <= _nx < SIZE and 0 <= _ny < SIZE:
                _nb.append(_ny * SIZE + _nx)
        ADJ.append(tuple(_nb))


class TerritoryBot:
    def __init__(self):
        self.board = bytearray(N)          # 0 = free, 1 = occupied trail
        self.delta = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
        self.prev_me = None
        self.last_move = None
        self.prev_all = None
        self.dead = [False, False, False, False]
        self.round = 0
        self.spent = 0.0                   # cumulative compute used so far
        self.worst_full = 0.003            # worst full-move cost seen

    # -- state update ------------------------------------------------------
    def observe(self, positions):
        self.round += 1
        for (x, y) in positions:
            if 0 <= x < SIZE and 0 <= y < SIZE:
                self.board[y * SIZE + x] = 1
        me = positions[0]
        if self.prev_me is not None and self.last_move is not None:
            d = (me[0] - self.prev_me[0], me[1] - self.prev_me[1])
            if d != (0, 0):
                self.delta[self.last_move] = d
                self.delta[OPP[self.last_move]] = (-d[0], -d[1])
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
        t0 = time.perf_counter()
        move = self._choose(positions)
        self.spent += time.perf_counter() - t0
        return move

    def _choose(self, positions):
        me = positions[0]
        # Round 1: a guaranteed-safe inward horizontal move (r/l never depend
        # on the y-axis sign, which we have not observed yet).
        if self.round <= 1:
            return self._emit(me, (1, 0) if me[0] == 0 else (-1, 0))

        mx, my = me
        roots = []                         # (dx, dy, cell_index) legal moves
        for _, dx, dy in MOVES:
            nx, ny = mx + dx, my + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                idx = ny * SIZE + nx
                if not self.board[idx]:
                    roots.append((dx, dy, idx))
        if not roots:
            return self._emit(me, (1, 0))   # boxed in - doomed anyway

        opps = [positions[i] for i in range(1, 4)
                if not self.dead[i]
                and 0 <= positions[i][0] < SIZE
                and 0 <= positions[i][1] < SIZE]

        # Time governor: if completing a full move could overrun the whole-
        # game budget, play an instant safe move instead. This degrades
        # gracefully on a slow machine rather than ever timing out.
        if self.spent + self.worst_full * 1.4 + 0.003 > TOTAL_BUDGET:
            return self._emit(me, self._cheap(roots, opps))

        t0 = time.perf_counter()
        choice = self._full(me, roots, opps)
        self.worst_full = max(self.worst_full, time.perf_counter() - t0)
        return self._emit(me, choice)

    # -- full 1-ply greedy evaluation -------------------------------------
    def _full(self, me, roots, opps):
        opp_dist = None
        if opps:
            opp_dist = self._bfs([oy * SIZE + ox for ox, oy in opps])

        room = {}
        voronoi = {}
        for dx, dy, c in roots:
            rm, vr = self._flood_eval(c, opp_dist)
            room[(dx, dy)] = rm
            voronoi[(dx, dy)] = vr

        risk = {}
        for dx, dy, c in roots:
            cx, cy = c % SIZE, c // SIZE
            r = 0.0
            for ox, oy in opps:
                if abs(ox - cx) + abs(oy - cy) == 1:   # opp could step here
                    k = self._exits(oy * SIZE + ox)
                    if k:
                        r += 1.0 / k
            risk[(dx, dy)] = r

        # Survival gate: if a collision-free move keeps real room, choose
        # only among those - never gamble a collision, never flee into a
        # pocket. Otherwise fall back to all moves (collision priced in).
        roomy = max(room.values())
        gate = [(dx, dy, c) for dx, dy, c in roots
                if risk[(dx, dy)] == 0.0 and room[(dx, dy)] >= 0.5 * roomy]
        candidates = gate if gate else roots

        best, best_score = None, -1e18
        for dx, dy, c in candidates:
            mv = (dx, dy)
            # Voronoi dominates; room is the survival tiebreak; wall-hug
            # (fewer open exits) fills tightly once moves are otherwise tied.
            score = (voronoi[mv] * 100000.0 + room[mv] * 100.0
                     - self._exits(c) - risk[mv] * 200000.0)
            if score > best_score:
                best_score, best = score, mv
        return best

    def _flood_eval(self, start, opp_dist):
        """BFS from `start` over free cells. Returns (room, voronoi):
        room = reachable free cells, voronoi = those we reach before any
        opponent. Counting is folded into the BFS to avoid an O(N) scan."""
        board = self.board
        dist = [INF] * N
        dist[start] = 0
        q = [start]
        head = 0
        if opp_dist is None:
            while head < len(q):
                c = q[head]
                head += 1
                for nb in ADJ[c]:
                    if dist[nb] == INF and not board[nb]:
                        dist[nb] = 0
                        q.append(nb)
            return len(q), len(q)
        vor = 1 if opp_dist[start] > 0 else 0
        while head < len(q):
            c = q[head]
            head += 1
            nd = dist[c] + 1
            for nb in ADJ[c]:
                if dist[nb] == INF and not board[nb]:
                    dist[nb] = nd
                    q.append(nb)
                    if nd < opp_dist[nb]:
                        vor += 1
        return len(q), vor

    # -- instant fallback (time governor) ---------------------------------
    def _cheap(self, roots, opps):
        """O(1) safe move: collision-free if possible, else most open exits -
        used only when the compute budget is nearly exhausted."""
        best, best_key = None, None
        for dx, dy, c in roots:
            cx, cy = c % SIZE, c // SIZE
            safe = 1
            for ox, oy in opps:
                if abs(ox - cx) + abs(oy - cy) == 1:
                    safe = 0
                    break
            key = (safe, self._exits(c))
            if best_key is None or key > best_key:
                best_key, best = key, (dx, dy)
        return best

    # -- helpers -----------------------------------------------------------
    def _bfs(self, sources):
        """Multi-source BFS over free cells; returns a flat distance array."""
        board = self.board
        dist = [INF] * N
        q = []
        for s in sources:
            if dist[s] == INF:
                dist[s] = 0
                q.append(s)
        head = 0
        while head < len(q):
            c = q[head]
            head += 1
            nd = dist[c] + 1
            for nb in ADJ[c]:
                if dist[nb] == INF and not board[nb]:
                    dist[nb] = nd
                    q.append(nb)
        return dist

    def _exits(self, idx):
        board = self.board
        n = 0
        for nb in ADJ[idx]:
            if not board[nb]:
                n += 1
        return n

    def _emit(self, me, choice):
        self.prev_me = me
        inv = {v: k for k, v in self.delta.items()}
        letter = inv.get(choice) or LETTER[choice]
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
            # Malformed line - never stay silent (silence = forfeit).
            sys.stdout.write((bot.last_move or "r") + "\n")
            sys.stdout.flush()
            continue
        positions = [(nums[2 * i], nums[2 * i + 1]) for i in range(4)]
        bot.observe(positions)
        sys.stdout.write(bot.decide(positions) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
