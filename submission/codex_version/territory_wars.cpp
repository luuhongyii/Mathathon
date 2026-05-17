// Territory Wars submission. The judge is LINUX - submit this .cpp SOURCE
// file (the platform compiles it). Do NOT submit a Windows .exe.
//
// Local test build:  g++ -std=c++17 -O2 territory_wars.cpp -o tw
//
// Game (Tron / light-cycle): 31x31 board, 4 players from the corners. Each
// turn every player steps u/d/l/r and claims cells. Your own territory remains
// passable; enemy territory and the edge are unsafe. Score = cells claimed.
// The judge sends only the four head positions per line - we accumulate owner.
//
// Strategy - iterative-deepening maximin search vs the nearest rival, with a
// CHAMBER-AWARE space evaluation:
//   * leaf eval = articulation-point "tree of chambers" estimate of the truly
//     usable space inside my Voronoi territory. Plain flood-fill over-counts:
//     it can't see that snaking up one column and down the next walls you into
//     a tiny region. Chamber analysis decomposes the space at its cut-points,
//     so once you enter a chamber through a narrow neck only that branch
//     counts - the bot stops boxing itself in.
//   * survival gate: prefer collision-free first moves that keep real room
//   * each move's thinking time is (remaining budget)/(remaining turns), so
//     the whole 512-turn game stays well under the ~500ms total budget
//   * if sealed off from every opponent: greedy wall-hug space-filling
// (Tried a1k0n's edge-count term and a red/black parity bound - measured: edge
// counting hurt in this 4-player absolute-eval setting, parity was neutral.
// Kept the plain chamber cell count, which tested strongest.)
// Never writes to stderr (the platform treats stderr as a forfeit).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

using Clock = std::chrono::steady_clock;

static const int SIZE = 31;
static const int N = SIZE * SIZE;
static const int INF = 1 << 29;
static const double INF_D = 1e18;
static const double BIG_D = 1e9;            // terminal (trapped) magnitude

static const double TOTAL_BUDGET = 0.34;    // compute seconds for whole game
                                             // (~160ms margin under the limit)
static const int    MAX_TURNS    = 512;     // game length (for time slicing)
static const int    MAX_DEPTH    = 12;      // iterative-deepening ceiling
static const double COLLIDE_W    = 600.0;   // residual collision penalty

static signed char owner[N];                // -1 = unclaimed, 0..3 = owner
static int adj[N][4];
static int adjn[N];
static int q[N];
static int dist_me[N], dist_op[N], dist_ot[N], dist_tmp[N];

// Chamber-analysis scratch. disc/artic/vis2 use a per-leaf generation stamp
// instead of a full-array memset (a 961-int memset x3 per search leaf is
// pure overhead): a cell is "current" only when its *_gen entry equals
// `cgen`. Bumping cgen logically clears all three in O(1).
static char inregion[N];                     // cells the estimate may use
static int  disc[N], low[N];                 // Tarjan discovery / low-link
static int  disc_gen[N];                     // generation of disc[] entry
static char artic[N];                        // articulation point?
static int  artic_gen[N];                    // generation of artic[] entry
static char vis2[N];                         // chamber-flood visited
static int  vis2_gen[N];                     // generation of vis2[] entry
static int  cq[N];                           // chamber-flood queue
static int  timer_;
static int  cgen = 0;                        // current chamber-analysis stamp

// Stamp-aware accessors: treat a stale entry as the cleared default.
static inline int  DISC(int i)  { return disc_gen[i] == cgen ? disc[i] : 0; }
static inline void SETDISC(int i, int v) { disc_gen[i] = cgen; disc[i] = v; }
static inline int  ARTIC(int i) { return artic_gen[i] == cgen ? artic[i] : 0; }
static inline void SETARTIC(int i) { artic_gen[i] = cgen; artic[i] = 1; }
static inline int  VIS2(int i)  { return vis2_gen[i] == cgen ? vis2[i] : 0; }
static inline void SETVIS2(int i) { vis2_gen[i] = cgen; vis2[i] = 1; }

// Move index: 0=u 1=d 2=l 3=r. dxs/dys map a letter to a board step and are
// self-calibrated from observed moves (handles a flipped y-axis).
static const char LET[4] = {'u', 'd', 'l', 'r'};
static int dxs[4] = {0, 0, -1, 1};
static int dys[4] = {-1, 1, 0, 0};
static const int OPPMOVE[4] = {1, 0, 3, 2};

// Per-game state.
static int g_round = 0;
static int prev_me_x = -1, prev_me_y = -1;
static int last_move = -1;
static int prev_all[8];
static bool have_prev = false;
static bool dead[4] = {false, false, false, false};
static double spent = 0.0;

// Search state.
static Clock::time_point g_deadline;
static bool g_timed_out = false;

static void init_adj() {
    const int ddx[4] = {1, -1, 0, 0}, ddy[4] = {0, 0, 1, -1};
    for (int y = 0; y < SIZE; y++)
        for (int x = 0; x < SIZE; x++) {
            int idx = y * SIZE + x, n = 0;
            for (int k = 0; k < 4; k++) {
                int nx = x + ddx[k], ny = y + ddy[k];
                if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE)
                    adj[idx][n++] = ny * SIZE + nx;
            }
            adjn[idx] = n;
        }
}

static bool passable(int idx, int player) {
    return owner[idx] < 0 || owner[idx] == player;
}

static int cell_exits(int idx, int player) {
    int n = 0;
    for (int k = 0; k < adjn[idx]; k++)
        if (passable(adj[idx][k], player)) n++;
    return n;
}

// Multi-source BFS over free cells; distances written into d[].
static void bfs(const int* src, int ns, int player, int* d) {
    for (int i = 0; i < N; i++) d[i] = INF;
    int head = 0, tail = 0;
    for (int i = 0; i < ns; i++)
        if (d[src[i]] == INF) { d[src[i]] = 0; q[tail++] = src[i]; }
    while (head < tail) {
        int c = q[head++], nd = d[c] + 1;
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (d[nb] == INF && passable(nb, player)) {
                d[nb] = nd;
                q[tail++] = nb;
            }
        }
    }
}

static int reach_count(int start, int player) {
    int head = 0, tail = 0;
    static int seen[N], stamp = 0;
    ++stamp;
    seen[start] = stamp;
    q[tail++] = start;
    while (head < tail) {
        int c = q[head++];
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (seen[nb] != stamp && passable(nb, player)) {
                seen[nb] = stamp;
                q[tail++] = nb;
            }
        }
    }
    return tail;
}

static int unclaimed_reach_count(int start, int player) {
    int head = 0, tail = 0, gain = 0;
    static int seen[N], stamp = 100000;
    ++stamp;
    seen[start] = stamp;
    q[tail++] = start;
    while (head < tail) {
        int c = q[head++];
        if (owner[c] < 0) gain++;
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (seen[nb] != stamp && passable(nb, player)) {
                seen[nb] = stamp;
                q[tail++] = nb;
            }
        }
    }
    return gain;
}

// Step distance from `start` to the nearest unclaimed cell, over cells
// passable to `player`. 0 if `start` itself is unclaimed, INF if none is
// reachable. Used to break oscillation ties: when no adjacent cell is fresh,
// steer toward the closest distant unclaimed cell instead of bouncing.
static int dist_to_unclaimed(int start, int player) {
    if (owner[start] < 0) return 0;
    static int seen[N], stamp = 300000;
    static int dep[N];
    ++stamp;
    int head = 0, tail = 0;
    seen[start] = stamp;  dep[start] = 0;
    q[tail++] = start;
    while (head < tail) {
        int c = q[head++], nd = dep[c] + 1;
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (seen[nb] == stamp || !passable(nb, player)) continue;
            if (owner[nb] < 0) return nd;
            seen[nb] = stamp;  dep[nb] = nd;
            q[tail++] = nb;
        }
    }
    return INF;
}

// -- chamber-aware usable-space estimate --------------------------------
// Tarjan articulation points over the subgraph induced by inregion[].
// Explicit-stack iterative DFS: the board has up to 961 cells, so a
// recursive DFS could overflow the call stack and forfeit the move.
static int tj_parent[N];      // DFS parent of each node
static int tj_kidx[N];        // next neighbour index to visit for each node
static int tj_children[N];    // root-child count (only meaningful at root)
static int tj_stack[N];       // explicit DFS stack
static void tarjan(int root, int parent_unused) {
    (void)parent_unused;
    int sp = 0;
    SETDISC(root, ++timer_);  low[root] = timer_;
    tj_parent[root] = -1;
    tj_kidx[root] = 0;
    tj_children[root] = 0;
    tj_stack[sp++] = root;
    while (sp > 0) {
        int u = tj_stack[sp - 1];
        if (tj_kidx[u] < adjn[u]) {
            int v = adj[u][tj_kidx[u]++];
            if (!inregion[v] || v == tj_parent[u]) continue;
            if (DISC(v) == 0) {                  // tree edge: descend
                if (tj_parent[u] == -1) tj_children[u]++;
                SETDISC(v, ++timer_);  low[v] = timer_;
                tj_parent[v] = u;
                tj_kidx[v] = 0;
                tj_children[v] = 0;
                tj_stack[sp++] = v;
            } else if (DISC(v) < low[u]) {       // back edge
                low[u] = DISC(v);
            }
        } else {                                 // done with u: pop, update
            sp--;
            int par = tj_parent[u];
            if (par != -1) {
                if (low[u] < low[par]) low[par] = low[u];
                if (tj_parent[par] != -1 && low[u] >= DISC(par))
                    SETARTIC(par);
            }
        }
    }
    if (tj_children[root] > 1) SETARTIC(root);
}

// One chamber = cells reachable from `start` without crossing an articulation
// point. We fill it, then exit through the single best articulation point.
// Explicit-stack iterative form (the recursive version could nest as deep as
// the number of chambers on a 961-cell board and overflow the call stack).
//
// Each frame is one chamber: `size` is its own cell count, `best` is the
// largest value among the disjoint sub-regions reached through its exits.
// The recursive contract `return size + best` is preserved by, on pop,
// folding `size + best` into the parent frame's `best`.
struct ChFrame {
    int size;        // cells in this chamber
    int best;        // best child sub-region value so far
    int ne;          // number of exit-seed cells
    int ei;          // next exit-seed index to expand
    int seeds[32];   // exit-seed start cells (one per disjoint sub-region)
};
static ChFrame ch_stack[N];

// Flood one chamber from `start`; records its size and exit-seeds in `f`.
static void chamber_fill(int start, ChFrame& f) {
    int head = 0, tail = 0;
    cq[tail++] = start;
    SETVIS2(start);
    f.size = 0; f.best = 0; f.ne = 0; f.ei = 0;
    int exits[32], ne = 0;
    while (head < tail) {
        int c = cq[head++];
        f.size++;
        if (c != start && ARTIC(c)) {            // chamber boundary
            if (ne < 32) exits[ne++] = c;
            continue;                            // do not expand past it
        }
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (inregion[nb] && !VIS2(nb)) { SETVIS2(nb); cq[tail++] = nb; }
        }
    }
    // Each unvisited cell beyond an exit seeds one disjoint sub-region.
    for (int e = 0; e < ne; e++) {
        int a = exits[e];
        for (int k = 0; k < adjn[a]; k++) {
            int nb = adj[a][k];
            if (inregion[nb] && !VIS2(nb) && f.ne < 32)
                f.seeds[f.ne++] = nb;
        }
    }
}

static int chamber_rec(int start) {
    int sp = 0;
    chamber_fill(start, ch_stack[sp]);
    sp++;
    int result = 0;
    while (sp > 0) {
        ChFrame& f = ch_stack[sp - 1];
        if (f.ei < f.ne) {
            int seed = f.seeds[f.ei++];
            if (VIS2(seed)) continue;            // already consumed
            chamber_fill(seed, ch_stack[sp]);    // descend into sub-region
            sp++;
        } else {                                 // chamber done: fold up
            int total = f.size + f.best;
            sp--;
            if (sp > 0) {
                if (total > ch_stack[sp - 1].best)
                    ch_stack[sp - 1].best = total;
            } else {
                result = total;
            }
        }
    }
    return result;
}

// Estimate of the cells we can actually still cover starting at `start`,
// confined to the cells flagged in inregion[] (set by the caller).
static int chamber_space(int start) {
    ++cgen;                  // O(1) logical clear of disc/artic/vis2
    timer_ = 0;
    tarjan(start, -1);
    return chamber_rec(start);
}

// Leaf evaluation: chamber-aware usable space inside my Voronoi region (the
// cells I reach strictly before every opponent).
static double evaluate(int mypos, int rivpos, int rival_player, bool have_rival) {
    int s[1];
    s[0] = mypos;
    bfs(s, 1, 0, dist_me);
    if (have_rival) { s[0] = rivpos; bfs(s, 1, rival_player, dist_op); }
    int room = 0, myv = 0, myunclaimed = 0;
    for (int i = 0; i < N; i++) {
        int dm = dist_me[i];
        if (dm >= INF) { inregion[i] = 0; continue; }
        room++;
        int dr = have_rival ? dist_op[i] : INF;
        bool mine = (dm < dr && dm < dist_ot[i]);
        if (mine) {
            myv++;
            if (owner[i] < 0) myunclaimed++;  // fresh territory I can paint
        }
        inregion[i] = mine ? 1 : 0;
    }
    inregion[mypos] = 1;
    // Score is cells *claimed*, but chamber_space/myv/room only reward
    // reachable space. Add an explicit term for the unclaimed cells inside my
    // Voronoi region so the search prefers lines that actually paint new
    // territory now over lines that merely retread cells I already own.
    return (double)chamber_space(mypos) + 0.90 * (double)myunclaimed
           + 0.12 * (double)myv + 0.02 * (double)room;
}

// Maximin search: I maximise, the nearest rival minimises. Other opponents
// are static. Moves are simulated by marking/unmarking the shared board.
static double maximin(int mypos, int rivpos, int rival_player,
                      int depth, bool opp_turn,
                      double alpha, double beta) {
    if (g_timed_out || Clock::now() > g_deadline) {
        g_timed_out = true;
        return 0.0;                          // discarded by the caller
    }
    if (depth == 0)
        return evaluate(mypos, rivpos, rival_player, true);

    if (opp_turn) {                          // minimiser (the rival)
        double best = INF_D;
        bool moved = false;
        for (int k = 0; k < adjn[rivpos]; k++) {
            int op = adj[rivpos][k];
            if (!passable(op, rival_player) || op == mypos) continue;
            moved = true;
            signed char old = owner[op];
            if (old < 0) owner[op] = (signed char)rival_player;
            double v = maximin(mypos, op, rival_player, depth - 1, false,
                               alpha, beta);
            owner[op] = old;
            if (v < best) best = v;
            if (best < beta) beta = best;
            if (beta <= alpha) break;
        }
        return moved ? best : BIG_D;         // rival trapped -> great for us
    } else {                                 // maximiser (me)
        double best = -INF_D;
        bool moved = false;
        for (int k = 0; k < adjn[mypos]; k++) {
            int np = adj[mypos][k];
            if (!passable(np, 0) || np == rivpos) continue;
            moved = true;
            signed char old = owner[np];
            if (old < 0) owner[np] = 0;
            double v = maximin(np, rivpos, rival_player, depth - 1, true,
                               alpha, beta);
            owner[np] = old;
            if (v > best) best = v;
            if (best > alpha) alpha = best;
            if (beta <= alpha) break;
        }
        return moved ? best : -BIG_D;        // I'm trapped -> disaster
    }
}

static void observe(const int* p) {
    g_round++;
    for (int i = 0; i < 4; i++) {
        int x = p[2 * i], y = p[2 * i + 1];
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE)
            owner[y * SIZE + x] = (signed char)i;
    }
    int mx = p[0], my = p[1];
    if (prev_me_x >= 0 && last_move >= 0) {
        int dx = mx - prev_me_x, dy = my - prev_me_y;
        if (dx != 0 || dy != 0) {
            dxs[last_move] = dx;  dys[last_move] = dy;
            int o = OPPMOVE[last_move];
            dxs[o] = -dx;  dys[o] = -dy;
        }
    }
    if (have_prev) {
        for (int i = 0; i < 4; i++) {
            if (p[2 * i] == prev_all[2 * i] &&
                p[2 * i + 1] == prev_all[2 * i + 1])
                dead[i] = true;                  // a live player must move
            int x = p[2 * i], y = p[2 * i + 1];
            if (!(x >= 0 && x < SIZE && y >= 0 && y < SIZE))
                dead[i] = true;
        }
    }
    for (int i = 0; i < 8; i++) prev_all[i] = p[i];
    have_prev = true;
}

// Convert a desired board step (ddx,ddy) to the letter that produces it.
static char emit(int mx, int my, int ddx, int ddy) {
    prev_me_x = mx;  prev_me_y = my;
    int m = -1;
    for (int k = 0; k < 4; k++)
        if (dxs[k] == ddx && dys[k] == ddy) { m = k; break; }
    if (m < 0)
        m = (ddx == 1) ? 3 : (ddx == -1) ? 2 : (ddy == 1) ? 1 : 0;
    last_move = m;
    return LET[m];
}

static char decide(const int* p) {
    int mx = p[0], my = p[1];

    // dist_ot[] is read unconditionally inside evaluate() but only filled in
    // the rival branch below. Initialise it to INF up front so a leftover
    // value from a previous turn can never leak into the eval.
    for (int i = 0; i < N; i++) dist_ot[i] = INF;

    if (g_round <= 1)
        return emit(mx, my, (mx == 0) ? 1 : -1, 0);

    // Legal first moves (true board space; emit() resolves letters). Order
    // u,d,l,r so symmetric ties break deterministically.
    const int TRY[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    int rdx[4], rdy[4], rc[4], nr = 0;
    for (int k = 0; k < 4; k++) {
        int nx = mx + TRY[k][0], ny = my + TRY[k][1];
        if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE &&
            passable(ny * SIZE + nx, 0)) {
            rdx[nr] = TRY[k][0];  rdy[nr] = TRY[k][1];
            rc[nr] = ny * SIZE + nx;  nr++;
        }
    }
    if (nr == 0) return emit(mx, my, 1, 0);          // boxed in - doomed

    int opx[3], opy[3], opc[3], opid[3], no = 0;
    for (int i = 1; i < 4; i++) {
        if (dead[i]) continue;
        int x = p[2 * i], y = p[2 * i + 1];
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            opx[no] = x;  opy[no] = y;  opc[no] = y * SIZE + x;
            opid[no] = i; no++;
        }
    }

    // Reachability flood from us: pick the nearest opponent as the rival.
    int mecell = my * SIZE + mx;
    int src[1] = {mecell};
    bfs(src, 1, 0, dist_me);
    int rival = -1, rbest = INF;
    for (int o = 0; o < no; o++) {
        int dmin = INF;
        for (int k = 0; k < adjn[opc[o]]; k++) {
            int d = dist_me[adj[opc[o]][k]];
            if (d < dmin) dmin = d;
        }
        if (dmin <= rbest) { rbest = dmin; rival = o; }
    }

    // Per move: chamber-aware room (over all free cells) and collision risk.
    for (int i = 0; i < N; i++) inregion[i] = passable(i, 0) ? 1 : 0;
    int chamber_room[4], roomy = 0, reach_room[4], reachy = 0;
    int gain_room[4], gainy = 0, d2u[4];
    double risk[4];
    for (int r = 0; r < nr; r++) {
        chamber_room[r] = chamber_space(rc[r]);
        reach_room[r] = reach_count(rc[r], 0);
        gain_room[r] = unclaimed_reach_count(rc[r], 0);
        if (chamber_room[r] > roomy) roomy = chamber_room[r];
        if (reach_room[r] > reachy) reachy = reach_room[r];
        if (gain_room[r] > gainy) gainy = gain_room[r];
        int cx = rc[r] % SIZE, cy = rc[r] / SIZE;
        double rk = 0.0;
        for (int o = 0; o < no; o++)
            if (passable(rc[r], opid[o]) &&
                abs(opx[o] - cx) + abs(opy[o] - cy) == 1) {
                int k = cell_exits(opc[o], opid[o]);
                if (k > 0) rk += 1.0 / k;
            }
        risk[r] = rk;
        d2u[r] = dist_to_unclaimed(rc[r], 0);
    }

    // Progress gate: if a safe move expands into a new cell without throwing
    // away most reachable gain, prefer it. This stops loops inside our own
    // territory while still allowing a retreat through owned cells when all
    // frontiers are blocked or a direct frontier is a tiny pocket.
    int gate[4], ng = 0;
    for (int r = 0; r < nr; r++)
        if (risk[r] == 0.0 && owner[rc[r]] < 0 &&
            (gain_room[r] >= 0.45 * gainy || reach_room[r] >= 0.35 * reachy))
            gate[ng++] = r;

    // Survival gate: if no progress move qualifies, keep collision-free moves
    // that preserve room. Otherwise keep all moves (collision priced in).
    if (ng == 0) {
        for (int r = 0; r < nr; r++)
            if (risk[r] == 0.0 &&
                (chamber_room[r] >= 0.5 * roomy || reach_room[r] >= 0.35 * reachy))
                gate[ng++] = r;
    }
    if (ng == 0)
        for (int r = 0; r < nr; r++) gate[ng++] = r;

    // No rival reachable -> our own region: greedy wall-hug space filling.
    if (rival < 0 || rbest == INF) {
        int best = gate[0];  double bs = -INF_D;
        for (int gi = 0; gi < ng; gi++) {
            int r = gate[gi];
            double sc = gain_room[r] * 5000.0
                        + reach_room[r] * 80.0
                        + chamber_room[r] * 25.0
                        + ((owner[rc[r]] < 0) ? 250.0 : 0.0)
                        - cell_exits(rc[r], 0)
                        - risk[r] * COLLIDE_W
                        - (double)(d2u[r] < 60 ? d2u[r] : 60) * 60.0;
            if (sc > bs) { bs = sc; best = r; }
        }
        return emit(mx, my, rdx[best], rdy[best]);
    }

    // Other (non-rival) opponents: static Voronoi sources for the eval.
    for (int i = 0; i < N; i++) dist_ot[i] = INF;
    for (int o = 0; o < no; o++)
        if (o != rival) {
            int otsrc[1] = {opc[o]};
            bfs(otsrc, 1, opid[o], dist_tmp);
            for (int i = 0; i < N; i++)
                if (dist_tmp[i] < dist_ot[i]) dist_ot[i] = dist_tmp[i];
        }

    int rivalcell = opc[rival];
    int rival_player = opid[rival];

    // Time slice: spread the remaining whole-game budget over the remaining
    // turns, so the full 512-turn game stays under TOTAL_BUDGET.
    int rem_turns = MAX_TURNS - g_round;
    if (rem_turns < 1) rem_turns = 1;
    double slice = (TOTAL_BUDGET - spent) / rem_turns;
    if (slice < 0.0005) slice = 0.0005;
    g_deadline = Clock::now() +
        std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(slice));

    // Iterative-deepening maximin over the gated moves; keep the best move
    // from the last fully-completed depth.
    int order[4];
    for (int i = 0; i < ng; i++) order[i] = gate[i];
    int best_root = gate[0];
    for (int depth = 2; depth <= MAX_DEPTH; depth++) {
        g_timed_out = false;
        int cur = -1;  double cur_score = -INF_D;
        for (int gi = 0; gi < ng; gi++) {
            int r = order[gi];
            signed char old = owner[rc[r]];
            if (old < 0) owner[rc[r]] = 0;
            double sc = maximin(rc[r], rivalcell, rival_player, depth - 1, true,
                                -INF_D, INF_D);
            owner[rc[r]] = old;
            if (g_timed_out) break;
            sc -= COLLIDE_W * risk[r];
            sc += gain_room[r] * 8.0 + ((old < 0) ? 120.0 : 0.0);
            sc -= (double)(d2u[r] < 60 ? d2u[r] : 60) * 4.0;  // toward frontier
            sc -= 0.001 * cell_exits(rc[r], 0);  // wall-hug tie-break
            if (sc > cur_score) { cur_score = sc; cur = r; }
        }
        if (g_timed_out) break;                  // discard partial depth
        best_root = cur;
        // Search the winner first next depth (sharper alpha-beta pruning).
        for (int gi = 0; gi < ng; gi++)
            if (order[gi] == cur) {
                for (int j = gi; j > 0; j--) order[j] = order[j - 1];
                order[0] = cur;
                break;
            }
        if (Clock::now() > g_deadline) break;
    }

    for (int r = 0; r < nr; r++)
        if (r == best_root)
            return emit(mx, my, rdx[r], rdy[r]);
    return emit(mx, my, rdx[0], rdy[0]);
}

int main() {
    init_adj();
    memset(owner, -1, sizeof owner);
    int p[8];
    while (scanf("%d %d %d %d %d %d %d %d", &p[0], &p[1], &p[2], &p[3],
                 &p[4], &p[5], &p[6], &p[7]) == 8) {
        observe(p);
        Clock::time_point t0 = Clock::now();
        char mv = decide(p);
        spent += std::chrono::duration<double>(Clock::now() - t0).count();
        putchar(mv);
        putchar('\n');
        fflush(stdout);
    }
    return 0;
}
