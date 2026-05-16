// Territory Wars submission. Compile STATIC and submit the .exe:
//
//   g++ -std=c++17 -O2 -static territory_wars.cpp -o territory_wars.exe
//
// Game (Tron / light-cycle): 31x31 board, 4 players from the corners. Each
// turn every player steps u/d/l/r and leaves a permanent trail; entering any
// claimed cell or the edge kills you. Score = cells claimed. The judge sends
// only the four head positions per line - we accumulate the trail map.
//
// The platform allows ~500ms of compute for the WHOLE game. C++ is fast
// enough to spend that on real search, so this bot runs an iterative-
// deepening maximin search against the nearest rival:
//   * leaf eval = my Voronoi territory - AGGR * rival's Voronoi territory
//     (negative term => actively cut the rival's space), via multi-source BFS
//   * alpha-beta pruning; a rational rival never suicides, so the search
//     ignores opponent moves that would kill the opponent
//   * survival gate: prefer collision-free first moves that keep real room
//   * each move's thinking time is (remaining budget)/(remaining turns), so
//     the whole 512-turn game stays comfortably under the 500ms total
//   * if sealed off from every opponent: greedy wall-hug space-filling
// Never writes to stderr (the platform treats stderr as a forfeit).

#include <cstdio>
#include <cstdlib>
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
static const double AGGR         = 0.0;     // cutting the rival backfires in a
                                             // 4-player free-for-all (the other
                                             // two expand freely) - measured
static const double SURV         = 0.02;    // mild own-room nudge
static const double COLLIDE_W    = 600.0;   // residual collision penalty

static int board[N];                        // 0 = free, 1 = occupied trail
static int adj[N][4];
static int adjn[N];
static int q[N];
static int dist_me[N], dist_op[N], dist_ot[N];

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

static int cell_exits(int idx) {
    int n = 0;
    for (int k = 0; k < adjn[idx]; k++)
        if (board[adj[idx][k]] == 0) n++;
    return n;
}

// Multi-source BFS over free cells; distances written into d[].
static void bfs(const int* src, int ns, int* d) {
    for (int i = 0; i < N; i++) d[i] = INF;
    int head = 0, tail = 0;
    for (int i = 0; i < ns; i++)
        if (d[src[i]] == INF) { d[src[i]] = 0; q[tail++] = src[i]; }
    while (head < tail) {
        int c = q[head++], nd = d[c] + 1;
        for (int k = 0; k < adjn[c]; k++) {
            int nb = adj[c][k];
            if (d[nb] == INF && board[nb] == 0) {
                d[nb] = nd;
                q[tail++] = nb;
            }
        }
    }
}

static int reach_count(int start) {
    int s[1] = {start};
    bfs(s, 1, dist_op);
    int n = 0;
    for (int i = 0; i < N; i++)
        if (dist_op[i] < INF) n++;
    return n;
}

// Leaf evaluation: Voronoi territory differential (mine - AGGR*rival's),
// plus a mild survival nudge. dist_ot (other opponents) is precomputed.
static double evaluate(int mypos, int rivpos, bool have_rival) {
    int s[1];
    s[0] = mypos;
    bfs(s, 1, dist_me);
    if (!have_rival) {
        int room = 0, myv = 0;
        for (int i = 0; i < N; i++)
            if (dist_me[i] < INF) {
                room++;
                if (dist_me[i] < dist_ot[i]) myv++;
            }
        return (double)myv + SURV * room;
    }
    s[0] = rivpos;
    bfs(s, 1, dist_op);
    int room = 0, myv = 0, rivv = 0;
    for (int i = 0; i < N; i++) {
        int dm = dist_me[i], dr = dist_op[i], dt = dist_ot[i];
        if (dm < INF) room++;
        if (dm < dr && dm < dt && dm < INF) myv++;
        else if (dr < dm && dr < dt && dr < INF) rivv++;
    }
    return (double)myv - AGGR * (double)rivv + SURV * room;
}

// Maximin search: I maximise, the nearest rival minimises. Other opponents
// are static. Moves are simulated by marking/unmarking the shared board.
static double maximin(int mypos, int rivpos, int depth, bool opp_turn,
                      double alpha, double beta) {
    if (g_timed_out || Clock::now() > g_deadline) {
        g_timed_out = true;
        return 0.0;                          // discarded by the caller
    }
    if (depth == 0)
        return evaluate(mypos, rivpos, true);

    if (opp_turn) {                          // minimiser (the rival)
        double best = INF_D;
        bool moved = false;
        for (int k = 0; k < adjn[rivpos]; k++) {
            int op = adj[rivpos][k];
            if (board[op] != 0 || op == mypos) continue;  // wall / suicide
            moved = true;
            board[op] = 1;
            double v = maximin(mypos, op, depth - 1, false, alpha, beta);
            board[op] = 0;
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
            if (board[np] != 0 || np == rivpos) continue; // wall / collision
            moved = true;
            board[np] = 1;
            double v = maximin(np, rivpos, depth - 1, true, alpha, beta);
            board[np] = 0;
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
            board[y * SIZE + x] = 1;
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
    if (g_round <= 1)
        return emit(mx, my, (mx == 0) ? 1 : -1, 0);

    // Legal first moves (true board space; emit() resolves letters). Order
    // u,d,l,r so symmetric ties break deterministically.
    const int TRY[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    int rdx[4], rdy[4], rc[4], nr = 0;
    for (int k = 0; k < 4; k++) {
        int nx = mx + TRY[k][0], ny = my + TRY[k][1];
        if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE &&
            board[ny * SIZE + nx] == 0) {
            rdx[nr] = TRY[k][0];  rdy[nr] = TRY[k][1];
            rc[nr] = ny * SIZE + nx;  nr++;
        }
    }
    if (nr == 0) return emit(mx, my, 1, 0);          // boxed in - doomed

    int opx[3], opy[3], opc[3], no = 0;
    for (int i = 1; i < 4; i++) {
        if (dead[i]) continue;
        int x = p[2 * i], y = p[2 * i + 1];
        if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
            opx[no] = x;  opy[no] = y;  opc[no] = y * SIZE + x;  no++;
        }
    }

    // Reachability flood from us: pick the nearest opponent as the rival.
    int mecell = my * SIZE + mx;
    int src[1] = {mecell};
    bfs(src, 1, dist_me);
    int rival = -1, rbest = INF;
    for (int o = 0; o < no; o++) {
        int dmin = INF;
        for (int k = 0; k < adjn[opc[o]]; k++) {
            int d = dist_me[adj[opc[o]][k]];
            if (d < dmin) dmin = d;
        }
        if (dmin < rbest) { rbest = dmin; rival = o; }
    }

    // Room (reachable free cells) and same-turn collision risk per move.
    int room[4], roomy = 0;
    double risk[4];
    for (int r = 0; r < nr; r++) {
        room[r] = reach_count(rc[r]);
        if (room[r] > roomy) roomy = room[r];
        int cx = rc[r] % SIZE, cy = rc[r] / SIZE;
        double rk = 0.0;
        for (int o = 0; o < no; o++)
            if (abs(opx[o] - cx) + abs(opy[o] - cy) == 1) {
                int k = cell_exits(opc[o]);
                if (k > 0) rk += 1.0 / k;
            }
        risk[r] = rk;
    }

    // Survival gate: if a collision-free move keeps real room, search only
    // those. Otherwise keep all moves (collision priced into the score).
    int gate[4], ng = 0;
    for (int r = 0; r < nr; r++)
        if (risk[r] == 0.0 && room[r] >= 0.5 * roomy) gate[ng++] = r;
    if (ng == 0)
        for (int r = 0; r < nr; r++) gate[ng++] = r;

    // No rival reachable -> our own region: greedy wall-hug space filling.
    if (rival < 0 || rbest == INF) {
        int best = gate[0];  double bs = -INF_D;
        for (int gi = 0; gi < ng; gi++) {
            int r = gate[gi];
            double sc = room[r] * 1000.0 - cell_exits(rc[r])
                        - risk[r] * COLLIDE_W;
            if (sc > bs) { bs = sc; best = r; }
        }
        return emit(mx, my, rdx[best], rdy[best]);
    }

    // Other (non-rival) opponents: static Voronoi sources for the eval.
    int otsrc[3], notc = 0;
    for (int o = 0; o < no; o++)
        if (o != rival) otsrc[notc++] = opc[o];
    if (notc > 0) bfs(otsrc, notc, dist_ot);
    else for (int i = 0; i < N; i++) dist_ot[i] = INF;

    int rivalcell = opc[rival];

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
            board[rc[r]] = 1;
            double sc = maximin(rc[r], rivalcell, depth - 1, true,
                                -INF_D, INF_D);
            board[rc[r]] = 0;
            if (g_timed_out) break;
            sc -= COLLIDE_W * risk[r];
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
