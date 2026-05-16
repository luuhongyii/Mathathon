"""Capture the Flag - Mathathon submission. Paste this whole file into the editor.

Game: 29x29 grid, two teams of two. Grab the ENEMY flag and carry it into your
own territory to win. You get caught (die, respawn after 30 turns) if you stand
in enemy territory next to a living opponent. Hydration drains every turn
(2/turn at home, 1/turn elsewhere); the central 5x5 oasis refills it to 140 and
is also a safe haven (neutral - nobody can be caught there).

Strategy
  * One bot instance per player; the teammate runs this same code (a clone).
    Roles split with NO communication: a lexicographic tiebreak on positions,
    committed the moment the two clones diverge. One ATTACKS (goes for the
    enemy flag), one DEFENDS (guards the home front, intercepts intruders/carriers).
  * Movement = greedy descent of BFS distance fields. Static fields (enemy
    flag, our flag, our territory, oasis) are computed once at round 0; at
    most one extra BFS per round (chasing a moving enemy). Cheap and safe on
    any time budget.
  * Danger model: a cell in enemy territory within Chebyshev 1 of a living
    enemy is where you get caught - avoided. If progress stalls too long the
    attacker turns "desperate" and pushes through anyway (a draw beats a
    timid stalemate; a death just costs a respawn, not a forfeit).
  * Refuels at the oasis when hydration would not otherwise last.

Robustness: never writes stderr (the platform treats stderr as a forfeit);
every round prints exactly one of u/d/l/r/s and flushes; all decision logic is
wrapped so a bug emits the last move instead of crashing.
"""
import sys
import random

SIZE = 29
N = SIZE * SIZE
INF = 1 << 29
DIRS = (("u", 0, -1), ("d", 0, 1), ("l", -1, 0), ("r", 1, 0))


def in_oasis(x, y):
    return 12 <= x <= 16 and 12 <= y <= 16


def territory(x, y):
    """'B' = blue home, 'R' = red home, 'N' = neutral (oasis + row 14)."""
    if 12 <= x <= 16 and 12 <= y <= 16:
        return "N"
    if y <= 13:
        return "B"
    if y >= 15:
        return "R"
    return "N"


def bfs(sources, adj):
    """Multi-source BFS over free cells; returns a flat distance array."""
    dist = [INF] * N
    q = []
    for s in sources:
        if 0 <= s < N and dist[s] == INF:
            dist[s] = 0
            q.append(s)
    head = 0
    while head < len(q):
        c = q[head]
        head += 1
        nd = dist[c] + 1
        for nb in adj[c]:
            if dist[nb] == INF:
                dist[nb] = nd
                q.append(nb)
    return dist


class Bot:
    def __init__(self):
        self.ready = False
        self.free = None
        self.adj = None
        self.blue = True
        self.my_flag = 0
        self.enemy_flag = 0
        self.my_terr = "B"
        self.enemy_terr = "R"
        self.d_enemyflag = None
        self.d_myterr = None
        self.d_oasis = None
        self.d_myflag = None
        self.d_guard = None
        self.role = None            # committed 'A' / 'D'
        self.refilling = False
        self.stall = 0
        self.last_tgt_id = None
        self.last_tgt_val = INF
        self.last_move = "s"

    # ---- one-time setup -------------------------------------------------
    def setup_board(self, board_text):
        free = bytearray(N)
        L = len(board_text)
        for i in range(N):
            free[i] = 0 if (i < L and board_text[i] == "#") else 1
        adj = [()] * N
        for y in range(SIZE):
            for x in range(SIZE):
                i = y * SIZE + x
                if not free[i]:
                    continue
                nb = []
                for _, dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < SIZE and 0 <= ny < SIZE and free[ny * SIZE + nx]:
                        nb.append(ny * SIZE + nx)
                adj[i] = tuple(nb)
        self.free = free
        self.adj = adj

    def setup_team(self, mx, my):
        # Blue starts/respawns at y=0, red at y=28 -> y tells the team.
        self.blue = (my <= 14)
        if self.blue:
            self.my_flag = 0                       # (0, 0)
            self.enemy_flag = 28 * SIZE + 28       # (28, 28)
            self.my_terr, self.enemy_terr = "B", "R"
        else:
            self.my_flag = 28 * SIZE + 28
            self.enemy_flag = 0
            self.my_terr, self.enemy_terr = "R", "B"
        free, adj = self.free, self.adj
        self.d_enemyflag = bfs([self.enemy_flag], adj)
        self.d_myflag = bfs([self.my_flag], adj)
        self.d_oasis = bfs(
            [y * SIZE + x for y in range(12, 17) for x in range(12, 17)
             if free[y * SIZE + x]], adj)
        self.d_myterr = bfs(
            [y * SIZE + x for y in range(SIZE) for x in range(SIZE)
             if free[y * SIZE + x] and territory(x, y) == self.my_terr], adj)
        guard = []
        for y in range(SIZE):
            for x in range(SIZE):
                i = y * SIZE + x
                if not free[i]:
                    continue
                if self.my_terr == "B":
                    if y in (12, 13) and 2 <= x <= 26:
                        guard.append(i)
                elif y in (15, 16) and 2 <= x <= 26:
                    guard.append(i)
        self.d_guard = bfs(guard or [self.my_flag], adj)
        self.ready = True

    # ---- helpers --------------------------------------------------------
    def _danger(self, x, y, enemies):
        """True if cell (x,y) is where I could be caught next turn."""
        if territory(x, y) != self.enemy_terr:
            return False
        for ex, ey, _, _ in enemies:
            if max(abs(ex - x), abs(ey - y)) <= 1:
                return True
        return False

    @staticmethod
    def _min_cheb(x, y, enemies):
        m = 99
        for ex, ey, _, _ in enemies:
            d = max(abs(ex - x), abs(ey - y))
            if d < m:
                m = d
        return m

    # ---- per-round decision --------------------------------------------
    def decide(self, v):
        players = [(v[i], v[i + 1], v[i + 2], v[i + 3]) for i in range(0, 16, 4)]
        mx, my, mh, mf = players[0]
        if mx < 0:                                  # dead / respawning
            self.stall = 0
            self.refilling = False
            return "s"
        if not self.ready:
            self.setup_team(mx, my)

        mate = players[1]
        mate_alive = mate[0] >= 0
        enemies = [e for e in (players[2], players[3]) if e[0] >= 0]
        my_idx = my * SIZE + mx
        i_carry = (mf == 1)
        mate_carry = mate_alive and mate[3] == 1
        carrier = next((e for e in enemies if e[3] == 1), None)

        # ----- role assignment (no communication) -----
        chase_field = None
        provisional = False
        if i_carry:
            role = "A"
        elif carrier is not None:
            chase_field = bfs([carrier[1] * SIZE + carrier[0]], self.adj)
            my_d = chase_field[my_idx]
            mate_d = (chase_field[mate[1] * SIZE + mate[0]]
                      if mate_alive else INF)
            mine_closer = (my_d < mate_d or
                           (my_d == mate_d and (mx, my) < (mate[0], mate[1])))
            role = "D" if (not mate_alive or mine_closer) else "A"
        elif mate_carry:
            role = "D"
        elif not mate_alive:
            role = "A"
        elif self.role is not None:
            role = self.role                        # committed - sticky
        elif (mx, my) != (mate[0], mate[1]):
            role = "A" if (mx, my) < (mate[0], mate[1]) else "D"
            self.role = role
        else:
            role = "A"                              # both on same cell yet
            provisional = True

        # ----- pick the target distance field -----
        do = self.d_oasis[my_idx]
        if in_oasis(mx, my):
            self.refilling = False
        elif do != INF and mh < 2 * do + 14:
            self.refilling = True

        if self.refilling and not in_oasis(mx, my):
            tgt, tgt_id, goal = self.d_oasis, "oasis", (14, 14)
        elif role == "A":
            if i_carry:
                tgt, tgt_id = self.d_myterr, "home"
                goal = (self.my_flag % SIZE, self.my_flag // SIZE)
            else:
                tgt, tgt_id = self.d_enemyflag, "eflag"
                goal = (self.enemy_flag % SIZE, self.enemy_flag // SIZE)
        else:                                       # defender
            intruder = carrier
            if intruder is None:
                best = INF
                for e in enemies:
                    if territory(e[0], e[1]) == self.my_terr:
                        ds = self.d_myflag[e[1] * SIZE + e[0]]
                        if ds < best:
                            best, intruder = ds, e
            if intruder is not None:
                if chase_field is not None and intruder is carrier:
                    tgt = chase_field
                else:
                    tgt = bfs([intruder[1] * SIZE + intruder[0]], self.adj)
                tgt_id, goal = "chase", (intruder[0], intruder[1])
            else:
                tgt, tgt_id = self.d_guard, "guard"
                goal = (14, 13 if self.blue else 15)

        # ----- stall tracking -> desperate push -----
        cur = tgt[my_idx]
        if tgt_id != self.last_tgt_id or cur < self.last_tgt_val or cur == 0:
            self.stall = 0
        else:
            self.stall += 1
        self.last_tgt_id, self.last_tgt_val = tgt_id, cur
        desperate = self.stall > 25

        # ----- candidate moves -----
        cands = []
        for letter, dx, dy in DIRS:
            nx, ny = mx + dx, my + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and self.free[ny * SIZE + nx]:
                cands.append((letter, nx, ny))
        cands.append(("s", mx, my))

        # Provisional clones share a cell - move randomly (but not backwards)
        # so the two diverge; positions then commit the lexicographic roles.
        if provisional:
            use_man = (cur == INF)
            gx, gy = goal
            forward = []
            for letter, nx, ny in cands:
                ni = ny * SIZE + nx
                dv = abs(nx - gx) + abs(ny - gy) if use_man else tgt[ni]
                if dv <= cur:
                    forward.append(letter)
            self.last_move = random.choice(forward or [c[0] for c in cands])
            return self.last_move

        # ----- score candidates -----
        use_man = (cur == INF)
        gx, gy = goal
        here_danger = self._danger(mx, my, enemies)
        best, best_key = "s", None
        for letter, nx, ny in cands:
            ni = ny * SIZE + nx
            if use_man:
                dval = abs(nx - gx) + abs(ny - gy)
            else:
                dv = tgt[ni]
                dval = dv if dv != INF else 9000 + abs(nx - gx) + abs(ny - gy)
            danger = self._danger(nx, ny, enemies) and not desperate
            terr_pen = (role == "D" and territory(nx, ny) == self.enemy_terr)
            safe = 0 if (danger or terr_pen) else 1
            if here_danger:                         # escape first
                key = (safe, self._min_cheb(nx, ny, enemies), -dval,
                       random.random())
            else:                                   # progress first
                key = (safe, -dval, random.random())
            if best_key is None or key > best_key:
                best_key, best = key, letter
        self.last_move = best
        return best


def main():
    bot = Bot()
    first = True
    for raw in sys.stdin:
        if first:
            first = False
            bot.setup_board(raw.strip())
            continue
        if not raw.strip():
            continue
        nums = []
        for tok in raw.split():
            try:
                nums.append(int(tok))
            except ValueError:
                pass
        if len(nums) < 16:
            sys.stdout.write((bot.last_move or "s") + "\n")
            sys.stdout.flush()
            continue
        try:
            move = bot.decide(nums[:16])
        except Exception:
            move = bot.last_move or "s"
        sys.stdout.write(move + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
