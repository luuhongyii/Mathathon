// Greed (snaky-greed) bot -- C++ port of submission/snaky_greed.py.
//
// Faithful port of the current Python bot: a depth-6 search that maximises
// total claimed steps (opponent treated as static), with a flood-fill room
// term and a greedy survival playout at each leaf to extend the death
// horizon, plus a root-level opponent-collision filter. Same logic, same
// SG_* env knobs; C++ only for speed.
//
//   g++ -O2 -std=c++17 -o submission/snaky_greed.exe submission/snaky_greed.cpp

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

static const int GRID = 32;
// DIRS order: u, d, l, r -- must match the Python so tie-breaks line up.
static const int DX[4] = {0, 0, -1, 1};
static const int DY[4] = {-1, 1, 0, 0};
static const char NAME[4] = {'u', 'd', 'l', 'r'};

// ---- tunable knobs (env override; defaults match snaky_greed.py) ----------
static int ROOM_CAP;       // flood-fill cap for the leaf room term
static double ROOM_W;      // weight on leaf free space
static int SURV_CAP;       // survival-playout horizon (plies)
static double SURV_PEN;    // penalty per missing survival ply

static double envd(const char *k, double def) {
    const char *v = getenv(k);
    return v ? atof(v) : def;
}
static int envi(const char *k, int def) {
    const char *v = getenv(k);
    return v ? atoi(v) : def;
}

// ---- shared game state -----------------------------------------------------
static int grid[GRID][GRID];
static bool claimed[GRID][GRID];
static int visited[GRID][GRID];   // stamped flood-fill marker
static int stamp = 0;
static int danger[GRID][GRID];    // enemy_reach map (0 = unreachable)

static inline bool in_grid(int x, int y) {
    return 0 <= x && x < GRID && 0 <= y && y < GRID;
}

// ---- simulate --------------------------------------------------------------
struct Sim {
    bool alive;
    int steps, nx, ny;
};

// Move from (x,y) in direction di. Jump distance is the digit on the first
// destination cell; a wall/claimed cell mid-flight stops us short (fatal).
static Sim simulate(int x, int y, int di) {
    int dx = DX[di], dy = DY[di];
    int ax = x + dx, ay = y + dy;
    int dist = in_grid(ax, ay) ? grid[ay][ax] : 1;
    for (int k = 1; k <= dist; ++k) {
        int cx = x + dx * k, cy = y + dy * k;
        if (!in_grid(cx, cy) || claimed[cy][cx])
            return {false, k - 1, x + dx * (k - 1), y + dy * (k - 1)};
    }
    return {true, dist, x + dx * dist, y + dy * dist};
}

// Mark / unmark every cell on a straight jump path (k = 1..steps).
static inline void set_path(int x, int y, int di, int steps, bool v) {
    int dx = DX[di], dy = DY[di];
    for (int k = 1; k <= steps; ++k)
        claimed[y + dy * k][x + dx * k] = v;
}

// ---- leaf room term --------------------------------------------------------
// Free cells reachable from (x,y)'s free neighbours, BFS, counted up to
// ROOM_CAP. (x,y) itself is claimed (the landing cell), so we seed from its
// neighbours -- matching the Python.
static int free_space(int x, int y) {
    ++stamp;
    static int qx[GRID * GRID + 8], qy[GRID * GRID + 8];
    int qh = 0, qt = 0;
    for (int di = 0; di < 4; ++di) {
        int nx = x + DX[di], ny = y + DY[di];
        if (in_grid(nx, ny) && !claimed[ny][nx] && visited[ny][nx] != stamp) {
            visited[ny][nx] = stamp;
            qx[qt] = nx;
            qy[qt] = ny;
            ++qt;
        }
    }
    int seen = 0;
    while (qh < qt && seen < ROOM_CAP) {
        int cx = qx[qh], cy = qy[qh];
        ++qh;
        ++seen;
        for (int di = 0; di < 4; ++di) {
            int nx = cx + DX[di], ny = cy + DY[di];
            if (in_grid(nx, ny) && !claimed[ny][nx] &&
                visited[ny][nx] != stamp) {
                visited[ny][nx] = stamp;
                qx[qt] = nx;
                qy[qt] = ny;
                ++qt;
            }
        }
    }
    return seen;
}

// ---- survival playout ------------------------------------------------------
// Greedy (max-steps) playout from (x,y); returns plies survived, capped at
// `cap`. Mimics how the depth-6 search plays on, so a line that walls itself
// in just past the search leaf yields a short run.
static int surv_run(int x, int y, int cap) {
    static int mkx[GRID * GRID], mky[GRID * GRID];
    int nm = 0;
    int cx = x, cy = y, plies = 0;
    while (plies < cap) {
        int best_steps = -1, bdi = -1, bnx = 0, bny = 0;
        for (int di = 0; di < 4; ++di) {
            Sim s = simulate(cx, cy, di);
            if (s.alive && s.steps > 0 && s.steps > best_steps) {
                best_steps = s.steps;
                bdi = di;
                bnx = s.nx;
                bny = s.ny;
            }
        }
        if (bdi < 0)
            break;
        for (int k = 1; k <= best_steps; ++k) {
            int px = cx + DX[bdi] * k, py = cy + DY[bdi] * k;
            claimed[py][px] = true;
            mkx[nm] = px;
            mky[nm] = py;
            ++nm;
        }
        cx = bnx;
        cy = bny;
        ++plies;
    }
    for (int i = 0; i < nm; ++i)
        claimed[mky[i]][mkx[i]] = false;
    return plies;
}

// ---- depth-6 search --------------------------------------------------------
struct SR {
    double val;
    int di;
};

// Maximise total claimed steps; survival first (a fully-lethal node returns
// best-steps - 1000). Opponent is static (not in the tree).
static SR search(int x, int y, int depth) {
    bool have = false, any_alive = false;
    double best_val = 0.0;
    int best_di = -1;
    for (int di = 0; di < 4; ++di) {
        Sim s = simulate(x, y, di);
        if (!s.alive)
            continue;
        any_alive = true;
        set_path(x, y, di, s.steps, true);
        double val;
        if (depth > 1) {
            val = s.steps + search(s.nx, s.ny, depth - 1).val;
        } else {
            val = s.steps + ROOM_W * free_space(s.nx, s.ny);
            int run = surv_run(s.nx, s.ny, SURV_CAP);
            if (run < SURV_CAP)
                val -= (SURV_CAP - run) * SURV_PEN;
        }
        set_path(x, y, di, s.steps, false);
        if (!have || val > best_val) {
            have = true;
            best_val = val;
            best_di = di;
        }
    }
    if (!any_alive) {
        // every direction is lethal: grab the most points before dying
        int bs = -1, bd = 0;
        bool h2 = false;
        for (int di = 0; di < 4; ++di) {
            Sim s = simulate(x, y, di);
            if (!h2 || s.steps > bs) {
                h2 = true;
                bs = s.steps;
                bd = di;
            }
        }
        return {bs - 1000.0, bd};
    }
    return {best_val, best_di};
}

// ---- opponent reach --------------------------------------------------------
// danger[y][x] = min opponent-steps to reach (x,y) this round (0 = none).
static void enemy_reach(int ox, int oy) {
    memset(danger, 0, sizeof(danger));
    for (int di = 0; di < 4; ++di) {
        int ax = ox + DX[di], ay = oy + DY[di];
        int dist = in_grid(ax, ay) ? grid[ay][ax] : 1;
        for (int k = 1; k <= dist; ++k) {
            int cx = ox + DX[di] * k, cy = oy + DY[di] * k;
            if (!in_grid(cx, cy) || claimed[cy][cx])
                break;
            if (danger[cy][cx] == 0 || k < danger[cy][cx])
                danger[cy][cx] = k;
        }
    }
}

// ---- root move choice ------------------------------------------------------
// depth-6 search value per first move, then drop moves whose jump path
// crosses a cell the opponent reaches first (a head-on collision is death).
static char decide(int mx, int my, int ex, int ey) {
    enemy_reach(ex, ey);
    double cval[4];
    int cdi[4];
    bool ccol[4];
    int nc = 0;
    int fsteps[4];
    for (int di = 0; di < 4; ++di) {
        Sim s = simulate(mx, my, di);
        fsteps[di] = s.steps;
        if (!s.alive)
            continue;
        set_path(mx, my, di, s.steps, true);
        SR sub = search(s.nx, s.ny, 5);
        set_path(mx, my, di, s.steps, false);
        double val = s.steps + sub.val;
        bool collide = false;
        for (int i = 1; i <= s.steps; ++i) {
            int cx = mx + DX[di] * i, cy = my + DY[di] * i;
            int d = danger[cy][cx];
            if (d > 0 && d <= i) {
                collide = true;
                break;
            }
        }
        cval[nc] = val;
        cdi[nc] = di;
        ccol[nc] = collide;
        ++nc;
    }
    if (nc == 0) {
        // every direction lethal: most points before dying
        // (tie-break on direction char, like Python max((steps, name))).
        int bd = 0;
        for (int di = 1; di < 4; ++di)
            if (fsteps[di] > fsteps[bd] ||
                (fsteps[di] == fsteps[bd] && NAME[di] > NAME[bd]))
                bd = di;
        return NAME[bd];
    }
    bool any_clean = false;
    for (int i = 0; i < nc; ++i)
        if (!ccol[i])
            any_clean = true;
    // pick max over (value, name, collide) among the pool (clean if any),
    // matching Python's max() on (val, name, collide) tuples.
    int bi = -1;
    for (int i = 0; i < nc; ++i) {
        if (any_clean && ccol[i])
            continue;
        if (bi < 0) {
            bi = i;
            continue;
        }
        if (cval[i] > cval[bi] ||
            (cval[i] == cval[bi] && NAME[cdi[i]] > NAME[cdi[bi]]) ||
            (cval[i] == cval[bi] && NAME[cdi[i]] == NAME[cdi[bi]] &&
             ccol[i] > ccol[bi]))
            bi = i;
    }
    return NAME[cdi[bi]];
}

// ---- trail bookkeeping -----------------------------------------------------
static void claim_path(int ox, int oy, int nx, int ny) {
    int sx = (nx > ox) - (nx < ox);
    int sy = (ny > oy) - (ny < oy);
    int cx = ox, cy = oy;
    while (cx != nx || cy != ny) {
        cx += sx;
        cy += sy;
        if (in_grid(cx, cy))
            claimed[cy][cx] = true;
    }
}

// ---- main ------------------------------------------------------------------
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    ROOM_CAP = envi("SG_ROOMCAP", 400);
    ROOM_W = envd("SG_ROOMW", 0.3);
    SURV_CAP = envi("SG_SURVCAP", 12);
    SURV_PEN = envd("SG_SURVPEN", 8.0);

    for (int y = 0; y < GRID; ++y)
        for (int x = 0; x < GRID; ++x)
            grid[y][x] = 1;
    // Round 0: 1024 grid digits, then the position line.
    for (int idx = 0; idx < GRID * GRID; ++idx) {
        int v;
        if (!(std::cin >> v))
            return 0;
        grid[idx / GRID][idx % GRID] = v;
    }
    int mx, my, ex, ey;
    if (!(std::cin >> mx >> my >> ex >> ey))
        return 0;

    memset(claimed, 0, sizeof(claimed));
    claimed[my][mx] = true;
    claimed[ey][ex] = true;

    std::cout << decide(mx, my, ex, ey) << "\n";
    std::cout.flush();

    int pmx = mx, pmy = my, pex = ex, pey = ey;
    while (std::cin >> mx >> my >> ex >> ey) {
        claim_path(pmx, pmy, mx, my);
        claim_path(pex, pey, ex, ey);
        if (in_grid(mx, my))
            claimed[my][mx] = true;
        if (in_grid(ex, ey))
            claimed[ey][ex] = true;
        pmx = mx;
        pmy = my;
        pex = ex;
        pey = ey;
        std::cout << decide(mx, my, ex, ey) << "\n";
        std::cout.flush();
    }
    return 0;
}
