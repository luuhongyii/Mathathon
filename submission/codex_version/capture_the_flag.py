"""Capture the Flag - Mathathon submission. Paste this whole file into the editor.

Game: 29x29 grid, two teams of two. Grab the ENEMY flag and carry it into your
own territory to win. You get caught (die, respawn after 30 turns) if you stand
in enemy territory next to a living opponent. Hydration drains every turn
(2/turn at home, 1/turn elsewhere); the central 5x5 oasis refills it to 140 and
is also a safe haven (neutral - nobody can be caught there).

Strategy (balanced build: aggressive pathing + selective safety)
  * Lexicographic role split: one attacks, one defends (no communication).
  * Greedy BFS fields; cheap route interception when defending.
  * _danger_next: avoid cells an enemy can reach in one step and catch you.
  * Dynamic flag staging only when a real enemy can contest their flag.
  * Defender escorts carriers, camps the home flag when threatened, otherwise
    guards the mid-line.
  * No deep search / 1-ply simulation (keeps games fast, fewer stale draws).

Robustness: never writes stderr; always prints u/d/l/r/s and flushes.
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
    if 12 <= x <= 16 and 12 <= y <= 16:
        return "N"
    if y <= 13:
        return "B"
    if y >= 15:
        return "R"
    return "N"


def bfs(sources, adj):
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
        self.d_enemyterr = None
        self.d_myterr = None
        self.d_homegate = None
        self.d_enemygate = None
        self.d_oasis = None
        self.d_myflag = None
        self.d_guard = None
        self.d_midguard = None
        self.d_flagring = None
        self.oasis_to_home = 0
        self.role = None
        self.refilling = False
        self.stall = 0
        self.guarded_wait = 0
        self.last_tgt_id = None
        self.last_tgt_val = INF
        self.last_move = "s"

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
        self.blue = (my <= 14)
        if self.blue:
            self.my_flag = 0
            self.enemy_flag = 28 * SIZE + 28
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
        self.d_enemyterr = bfs(
            [y * SIZE + x for y in range(SIZE) for x in range(SIZE)
             if free[y * SIZE + x] and territory(x, y) == self.enemy_terr],
            adj)
        homegate, enemygate = [], []
        for y in range(SIZE):
            for x in range(SIZE):
                i = y * SIZE + x
                if not free[i]:
                    continue
                if self.my_terr == "B":
                    if y in (12, 13):
                        homegate.append(i)
                    if y in (15, 16):
                        enemygate.append(i)
                else:
                    if y in (15, 16):
                        homegate.append(i)
                    if y in (12, 13):
                        enemygate.append(i)
        self.d_homegate = bfs(homegate or [self.my_flag], adj)
        self.d_enemygate = bfs(enemygate or [self.enemy_flag], adj)
        fx, fy = self.my_flag % SIZE, self.my_flag // SIZE
        oasis, guard, midguard, ring = [], [], [], []
        for y in range(SIZE):
            for x in range(SIZE):
                i = y * SIZE + x
                if not free[i]:
                    continue
                if in_oasis(x, y):
                    oasis.append(i)
                cheb = max(abs(x - fx), abs(y - fy))
                if 1 <= cheb <= 2 and territory(x, y) == self.my_terr:
                    ring.append(i)
                if self.my_terr == "B":
                    if y in (12, 13) and 2 <= x <= 26:
                        guard.append(i)
                    if y in (10, 11, 12) and 4 <= x <= 24:
                        midguard.append(i)
                elif y in (15, 16) and 2 <= x <= 26:
                    guard.append(i)
                    if y in (16, 17, 18) and 4 <= x <= 24:
                        midguard.append(i)
        self.d_guard = bfs(guard or [self.my_flag], adj)
        self.d_midguard = bfs(midguard or guard or [self.my_flag], adj)
        self.d_flagring = bfs(ring or [self.my_flag], adj)
        self.oasis_to_home = 0
        for i in oasis:
            d = self.d_myterr[i]
            if d != INF and d > self.oasis_to_home:
                self.oasis_to_home = d
        self.ready = True

    def _danger(self, x, y, enemies):
        if territory(x, y) != self.enemy_terr:
            return False
        for ex, ey, _, _ in enemies:
            if max(abs(ex - x), abs(ey - y)) <= 1:
                return True
        return False

    def _danger_next(self, x, y, enemies):
        if territory(x, y) != self.enemy_terr:
            return False
        if self._danger(x, y, enemies):
            return True
        for ex, ey, _, _ in enemies:
            if max(abs(ex - x), abs(ey - y)) > 2:
                continue
            for _, dx, dy in DIRS:
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < SIZE and 0 <= ny < SIZE and self.free[ny * SIZE + nx]:
                    if max(abs(nx - x), abs(ny - y)) <= 1:
                        return True
        return False

    def _enemy_effectively_dead(self, e, mate, mate_alive):
        ex, ey, eh, _ = e
        if 0 < eh <= 3:
            return True
        if mate_alive and territory(ex, ey) == self.my_terr:
            return max(abs(mate[0] - ex), abs(mate[1] - ey)) <= 1
        return False

    def _flag_guarded(self, my_idx, enemies, mate, mate_alive):
        if not enemies:
            return False
        my_dist = self.d_enemyflag[my_idx]
        enemy_best = INF
        for e in enemies:
            if self._enemy_effectively_dead(e, mate, mate_alive):
                continue
            ex, ey, eh, _ = e
            d = self.d_enemyflag[ey * SIZE + ex]
            if 0 < eh < 30:
                d += (30 - eh) // 4
            if d < enemy_best:
                enemy_best = d
        return enemy_best != INF and enemy_best <= my_dist

    def _escort_field(self, mate):
        mate_i = mate[1] * SIZE + mate[0]
        dm = self.d_myflag[mate_i]
        sources = []
        for i in range(N):
            if not self.free[i]:
                continue
            d = self.d_myflag[i]
            if d == INF or d > dm:
                continue
            x, y = i % SIZE, i // SIZE
            if territory(x, y) not in (self.my_terr, "N"):
                continue
            if dm - d <= 10:
                sources.append(i)
        if not sources:
            sources = [mate_i]
        return bfs(sources, self.adj)

    def _intercept_field(self, intr, goal_field, my_field):
        intr_i = intr[1] * SIZE + intr[0]
        intr_field = bfs([intr_i], self.adj)
        goal_dist = goal_field[intr_i]
        best_cell = -1
        best_remaining = -1
        for i in range(N):
            if not self.free[i]:
                continue
            enemy_here = intr_field[i]
            enemy_rest = goal_field[i]
            if enemy_here == INF or enemy_rest == INF:
                continue
            if goal_dist != INF and enemy_here + enemy_rest > goal_dist + 1:
                continue
            mine_here = my_field[i]
            if mine_here == INF or mine_here > enemy_here:
                continue
            if enemy_rest > best_remaining:
                best_remaining = enemy_rest
                best_cell = i
        if best_cell < 0:
            return intr_field, (intr[0], intr[1])
        return bfs([best_cell], self.adj), (best_cell % SIZE, best_cell // SIZE)

    def _update_refill(self, mx, my, mh, my_idx, i_carry, mate, mate_alive):
        if mate_alive and in_oasis(mate[0], mate[1]) and mate[2] >= 105 and mh >= 72:
            self.refilling = False
            return
        do = self.d_oasis[my_idx]
        if in_oasis(mx, my):
            self.refilling = mh < (95 if i_carry else 108)
            return
        if do == INF:
            self.refilling = False
            return
        if i_carry:
            home_direct = self.d_myterr[my_idx]
            via_oasis = do + self.oasis_to_home
            can_go_home = mh >= home_direct + 24
            can_reach_oasis = mh >= do + 4
            oasis_helps = via_oasis <= home_direct + 16
            self.refilling = (not can_go_home and can_reach_oasis
                              and oasis_helps)
        else:
            self.refilling = mh < 2 * do + 12

    @staticmethod
    def _min_cheb(x, y, enemies):
        m = 99
        for ex, ey, _, _ in enemies:
            d = max(abs(ex - x), abs(ey - y))
            if d < m:
                m = d
        return m

    def decide(self, v):
        players = [(v[i], v[i + 1], v[i + 2], v[i + 3]) for i in range(0, 16, 4)]
        mx, my, mh, mf = players[0]
        if mx < 0:
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
        enemy_near_home = any(self.d_myflag[e[1] * SIZE + e[0]] <= 14
                              for e in enemies)
        mate_primary_attack = False
        if mate_alive and not i_carry and not mate_carry and carrier is None:
            my_ef = self.d_enemyflag[my_idx]
            mate_i = mate[1] * SIZE + mate[0]
            mate_ef = self.d_enemyflag[mate_i]
            mate_primary_attack = (mate_ef != INF and
                                   (mate_ef + 8 < my_ef or mate_ef <= 12))

        chase_field = None
        provisional = False
        if i_carry:
            role = "A"
        elif mate_carry:
            role = "D"
        elif carrier is not None:
            chase_field = bfs([carrier[1] * SIZE + carrier[0]], self.adj)
            my_d = chase_field[my_idx]
            mate_d = (chase_field[mate[1] * SIZE + mate[0]]
                      if mate_alive else INF)
            mine_closer = (my_d < mate_d or
                           (my_d == mate_d and (mx, my) < (mate[0], mate[1])))
            role = "D" if (not mate_alive or mine_closer) else "A"
        elif mate_primary_attack:
            role = "D"
        elif not mate_alive:
            role = "D" if enemy_near_home else "A"
        elif self.role is not None:
            role = self.role
        elif (mx, my) != (mate[0], mate[1]):
            role = "A" if (mx, my) < (mate[0], mate[1]) else "D"
            self.role = role
        else:
            role = "A"
            provisional = True

        self._update_refill(mx, my, mh, my_idx, i_carry, mate, mate_alive)
        carrier_emergency = (carrier is not None and not i_carry and
                             not mate_carry)
        continue_steal = (carrier_emergency and
                          self.d_enemyflag[my_idx] <= 6 and mh >= 44)

        flag_guarded = (role == "A" and not i_carry
                        and self._flag_guarded(my_idx, enemies, mate,
                                               mate_alive))
        if flag_guarded:
            self.guarded_wait += 1
        else:
            self.guarded_wait = 0
        stage_guarded = flag_guarded and self.guarded_wait <= 8

        if self.refilling and not in_oasis(mx, my):
            tgt, tgt_id, goal = self.d_oasis, "oasis", (14, 14)
        elif stage_guarded:
            tgt, tgt_id, goal = self.d_oasis, "stage", (14, 14)
        elif role == "A" and carrier_emergency and not continue_steal:
            if in_oasis(mx, my) or mh >= 80:
                tgt, tgt_id = self.d_midguard, "resetguard"
                goal = (14, 13 if self.blue else 15)
            else:
                tgt, tgt_id = self.d_oasis, "reset"
                goal = (14, 14)
        elif role == "A":
            if i_carry:
                gate_blocked = (self.d_homegate[my_idx] <= 8 and
                                any(self.d_homegate[e[1] * SIZE + e[0]] <= 3
                                    for e in enemies))
                exit_trap = (territory(mx, my) == self.enemy_terr and
                             not in_oasis(mx, my) and
                             self.d_homegate[my_idx] > 5 and
                             any(e[3] == 0 and
                                 max(abs(e[0] - mx), abs(e[1] - my)) <= 6
                                 for e in enemies))
                if (gate_blocked or exit_trap) and not in_oasis(mx, my):
                    tgt, tgt_id = self.d_oasis, "reroute"
                    goal = (14, 14)
                else:
                    tgt, tgt_id = self.d_myterr, "home"
                    goal = (self.my_flag % SIZE, self.my_flag // SIZE)
            else:
                tgt, tgt_id = self.d_enemyflag, "eflag"
                goal = (self.enemy_flag % SIZE, self.enemy_flag // SIZE)
        else:
            if mate_carry:
                tgt = self._escort_field(mate)
                tgt_id, goal = "escort", (
                    self.my_flag % SIZE, self.my_flag // SIZE)
                if enemies:
                    mate_field = bfs([mate[1] * SIZE + mate[0]], self.adj)
                    mate_home = self.d_myterr[mate[1] * SIZE + mate[0]]
                    best = INF
                    blocker = None
                    for e in enemies:
                        eidx = e[1] * SIZE + e[0]
                        ds = mate_field[eidx]
                        if ds == INF:
                            continue
                        if (territory(mate[0], mate[1]) == self.enemy_terr
                                and e[3] == 0 and ds <= 8):
                            ds -= 14
                        if self.d_homegate[mate[1] * SIZE + mate[0]] <= 10:
                            if self.d_homegate[eidx] <= 4:
                                ds -= 10
                            else:
                                ds += 6
                        if e[3] == 1:
                            catch = bfs([eidx], self.adj)[my_idx]
                            if catch <= self.d_enemyterr[eidx] + 3:
                                ds -= 6
                            else:
                                ds += 18
                        if ds < best:
                            best, blocker = ds, e
                    if blocker is not None and best <= mate_home + 20:
                        tgt = bfs([blocker[1] * SIZE + blocker[0]], self.adj)
                        tgt_id, goal = "block", (blocker[0], blocker[1])
            else:
                enemy_near_flag = INF
                for e in enemies:
                    d = self.d_myflag[e[1] * SIZE + e[0]]
                    if d < enemy_near_flag:
                        enemy_near_flag = d

                anchor_defense = (mate_primary_attack and carrier is None
                                  and enemy_near_flag > 9)
                if anchor_defense:
                    if in_oasis(mx, my) and mh >= 120:
                        tgt, tgt_id = self.d_midguard, "anchor"
                        goal = (14, 13 if self.blue else 15)
                    else:
                        tgt, tgt_id = self.d_oasis, "anchorfill"
                        goal = (14, 14)
                else:
                    intruder = carrier
                    if intruder is None:
                        best = INF
                        for e in enemies:
                            if territory(e[0], e[1]) == self.my_terr:
                                ds = self.d_myflag[e[1] * SIZE + e[0]]
                                if ds < best:
                                    best, intruder = ds, e
                        if intruder is None:
                            for e in enemies:
                                approaching = (11 <= e[1] <= 16 if self.my_terr == "B"
                                               else 12 <= e[1] <= 17)
                                ds = self.d_myflag[e[1] * SIZE + e[0]]
                                if approaching and ds < best:
                                    best, intruder = ds, e
                    if carrier is None and enemy_near_flag <= 9:
                        tgt, tgt_id = self.d_flagring, "campflag"
                        goal = (self.my_flag % SIZE, self.my_flag // SIZE)
                    elif intruder is not None:
                        goal_field = (self.d_enemyterr if intruder is carrier
                                      else self.d_myflag)
                        my_field = bfs([my_idx], self.adj)
                        tgt, goal = self._intercept_field(intruder, goal_field,
                                                          my_field)
                        tgt_id = "intercept"
                    else:
                        tgt, tgt_id = (self.d_midguard if enemies else self.d_guard,
                                       "midguard" if enemies else "guard")
                        goal = (14, 13 if self.blue else 15)

        cur = tgt[my_idx]
        if tgt_id != self.last_tgt_id or cur < self.last_tgt_val or cur == 0:
            self.stall = 0
        else:
            self.stall += 1
        self.last_tgt_id, self.last_tgt_val = tgt_id, cur
        stall_lim = 16 if i_carry else 25
        desperate = self.stall > stall_lim and not flag_guarded

        cands = []
        for letter, dx, dy in DIRS:
            nx, ny = mx + dx, my + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE and self.free[ny * SIZE + nx]:
                cands.append((letter, nx, ny))
        cands.append(("s", mx, my))

        if provisional:
            use_man = (cur == INF)
            gx, gy = goal
            forward, lateral = [], []
            for letter, nx, ny in cands:
                if letter == "s":
                    continue
                ni = ny * SIZE + nx
                dv = abs(nx - gx) + abs(ny - gy) if use_man else tgt[ni]
                if dv <= cur:
                    forward.append(letter)
                    if nx != mx:
                        lateral.append(letter)
            self.last_move = random.choice(lateral or forward or [c[0] for c in cands])
            return self.last_move

        use_man = (cur == INF)
        gx, gy = goal
        here_danger = self._danger_next(mx, my, enemies)
        best, best_key = "s", None
        for letter, nx, ny in cands:
            ni = ny * SIZE + nx
            if use_man:
                dval = abs(nx - gx) + abs(ny - gy)
            else:
                dv = tgt[ni]
                dval = dv if dv != INF else 9000 + abs(nx - gx) + abs(ny - gy)
            immediate = self._danger(nx, ny, enemies)
            lookahead = self._danger_next(nx, ny, enemies)
            danger = immediate or (lookahead and not desperate)
            terr_pen = (role == "D" and territory(nx, ny) == self.enemy_terr)
            safe = 0 if (danger or terr_pen) else 1
            enemy_clear = (9 if not enemies
                           else min(9, self._min_cheb(nx, ny, enemies)))
            if here_danger:
                key = (safe, self._min_cheb(nx, ny, enemies), -dval,
                       random.random())
            elif i_carry:
                key = (safe, -dval, enemy_clear, random.random())
            else:
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
