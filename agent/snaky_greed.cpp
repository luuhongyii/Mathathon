// Snaky Greed agent: codex C++ core + opening book + phased geometry.
// Bench: python tools/greed_bench.py 40 agent/snaky_greed.exe
// Replay: python tools/greed_replay.py agent/snaky_greed.exe --slot 1
// Never writes to stderr; prints one of u/d/l/r and flushes every turn.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static const int GRID = 32;
static const int N = GRID * GRID;
static const double GAME_BUDGET = 8.0;
static const double PER_MOVE_SOFT = 0.35;
static const int SURV_CAP = 12;
static const double SURV_PEN = 55.0;
static const int INF = 1 << 28;

// Fixed corner lines (SE vs tungtung / NW vs stRWategy-style bots).
static const char SE_OPEN[] = "urddrd";
static const char NW_OPEN[] = "rdrurl";
static const int OPEN_PLIES = 6;
static const int OPEN_EARLY_PLIES = 10;
static const int OPEN_FREE_MIN = 750;

static const char DCH[4] = {'u', 'd', 'l', 'r'};
static const int DX[4] = {0, 0, -1, 1};
static const int DY[4] = {-1, 1, 0, 0};

struct Move {
    bool alive = false;
    int steps = 0;
    int nx = 0, ny = 0;
    std::vector<int> path;
};

struct Bot {
    int grid[GRID][GRID];
    bool claimed[GRID][GRID];
    int mx = -1, my = -1, ex = -1, ey = -1;
    int last_ex = -999, last_ey = -999, enemy_stale = 0;
    int last_edge_side = 0, edge_streak = 0;
    bool have_enemy_seen = false;
    int start_mx = -1, start_my = -1;
    int ply = 0;
    double spent = 0.0;
    Clock::time_point deadline;

    static bool in_grid(int x, int y) {
        return 0 <= x && x < GRID && 0 <= y && y < GRID;
    }

    Move simulate(int x, int y, int dx, int dy) {
        Move m;
        int ax = x + dx, ay = y + dy;
        int dist = in_grid(ax, ay) ? grid[ay][ax] : 1;
        for (int k = 1; k <= dist; k++) {
            int cx = x + dx * k, cy = y + dy * k;
            if (!in_grid(cx, cy) || claimed[cy][cx]) {
                m.alive = false;
                m.steps = k - 1;
                m.nx = x + dx * (k - 1);
                m.ny = y + dy * (k - 1);
                return m;
            }
            m.path.push_back(cy * GRID + cx);
        }
        m.alive = true;
        m.steps = dist;
        m.nx = x + dx * dist;
        m.ny = y + dy * dist;
        return m;
    }

    std::vector<Move> moves_from(int x, int y) {
        std::vector<Move> out;
        if (!in_grid(x, y)) return out;
        for (int d = 0; d < 4; d++) {
            Move m = simulate(x, y, DX[d], DY[d]);
            if (m.alive && m.steps > 0) out.push_back(std::move(m));
        }
        return out;
    }

    void mark_path(const std::vector<int>& path, bool v) {
        for (int c : path) claimed[c / GRID][c % GRID] = v;
    }

    static double opening_scale(int free_cells) {
        if (free_cells >= 820) return 0.0;
        if (free_cells <= 680) return 1.0;
        return (820.0 - free_cells) / 140.0;
    }

  // 0 = opening sprint, 1 = full contest geometry.
    static double phase_geom(int free_cells, int turn) {
        if (turn < 8 && free_cells >= 750) return 0.0;
        if (free_cells >= 900) return 0.0;
        if (free_cells >= 750) return 0.15 * opening_scale(free_cells);
        if (free_cells >= 550) return 0.5 + 0.5 * opening_scale(free_cells);
        return 1.0;
    }

    // Hard-coded corner script (first OPEN_PLIES turns); falls through if illegal.
    char scripted_opening() {
        if (ply > OPEN_PLIES || start_mx < 0)
            return 0;
        const char *script = nullptr;
        int slen = 0;
        if (start_mx >= 18 && start_my >= 18) {
            script = SE_OPEN;
            slen = (int)sizeof(SE_OPEN) - 1;
        } else if (start_mx <= 13 && start_my <= 13) {
            script = NW_OPEN;
            slen = (int)sizeof(NW_OPEN) - 1;
        } else {
            return 0;
        }
        if (ply > slen)
            return 0;
        char want = script[ply - 1];
        for (int d = 0; d < 4; d++) {
            if (DCH[d] != want)
                continue;
            Move m = simulate(mx, my, DX[d], DY[d]);
            if (m.alive && m.steps > 0)
                return DCH[d];
        }
        return 0;
    }

    // Heuristic opening when script is blocked: max jump + structural direction.
    char opening_book(int free_cells) {
        if (ply >= 8) return 0;
        int claimed_cells = GRID * GRID - free_cells;
        if (claimed_cells > 110) return 0;

        char best = 0;
        double best_sc = -1e100;
        const bool nw = (mx < 17 && my < 17);
        const bool se = (mx >= 15 && my >= 15);

        for (int d = 0; d < 4; d++) {
            Move m = simulate(mx, my, DX[d], DY[d]);
            if (!m.alive || m.steps <= 0) continue;
            // Steps dominate (digit jumps); direction is a tie-breaker only.
            double s = m.steps * 12.0;
            if (m.steps >= 6) s += 10.0;

            if (nw) {
                s += 2.5 * (m.nx - mx) + 1.5 * (m.ny - my);
                if (ply == 1) s += 4.0 * (m.ny - my);
                if (ply == 2) s += 3.0 * (m.nx - mx);
            }
            if (se) {
                s += 2.5 * (my - m.ny) + 2.0 * (m.nx - mx);
                if (ply == 2 || ply == 3) s += 4.0 * (m.ny - my);
                if (m.nx >= 28 && m.ny >= 28) s += 8.0;
            }
            if (s > best_sc) {
                best_sc = s;
                best = DCH[d];
            }
        }
        return best;
    }

    static bool path_lethal(const std::vector<int>& path,
                            const std::array<std::pair<int, int>, N>& danger) {
        for (int i = 0; i < (int)path.size(); i++) {
            const auto& hit = danger[path[i]];
            if (hit.first == 1 && hit.second <= i + 1) return true;
        }
        return false;
    }

    int survival_depth(int x, int y, int cap) {
        if (cap <= 0) return 0;
        std::vector<Move> mv = moves_from(x, y);
        if (mv.empty()) return 0;
        int best = 1;
        for (Move& m : mv) {
            mark_path(m.path, true);
            int d = 1 + survival_depth(m.nx, m.ny, cap - 1);
            mark_path(m.path, false);
            best = std::max(best, d);
            if (best >= cap) break;
        }
        return best;
    }

    std::array<std::pair<int, int>, N> enemy_future_reach(int ox, int oy) {
        std::array<std::pair<int, int>, N> danger;
        for (int i = 0; i < N; i++) danger[i] = {INF, INF};
        if (!in_grid(ox, oy)) return danger;
        auto remember = [&](int cell, int turn, int step) {
            std::pair<int, int> val(turn, step);
            if (val < danger[cell]) danger[cell] = val;
        };
        std::vector<Move> ms = moves_from(ox, oy);
        for (const Move& m : ms) {
            for (int i = 0; i < (int)m.path.size(); i++)
                remember(m.path[i], 1, i + 1);
            mark_path(m.path, true);
            std::vector<Move> ms2 = moves_from(m.nx, m.ny);
            for (const Move& m2 : ms2)
                for (int i = 0; i < (int)m2.path.size(); i++)
                    remember(m2.path[i], 2, i + 1);
            mark_path(m.path, false);
        }
        return danger;
    }

    std::array<int, 3> mobility(int x, int y) {
        std::vector<Move> ms = moves_from(x, y);
        int step_sum = 0, best = 0;
        for (const Move& m : ms) {
            step_sum += std::min(m.steps, 6);
            best = std::max(best, m.steps);
        }
        return {(int)ms.size(), step_sum, best};
    }

    double collision_penalty(const std::vector<int>& path,
                             const std::array<std::pair<int, int>, N>& danger) {
        if (path.empty()) return 0.0;
        double worst = 0.0, future = 0.0;
        int end = path.back();
        for (int i = 0; i < (int)path.size(); i++) {
            int cell = path[i];
            auto hit = danger[cell];
            if (hit.first >= INF) continue;
            int turn = hit.first, step = hit.second;
            if (turn == 1) {
                int gap = std::abs((i + 1) - step);
                double val = 95.0 / (gap + 1);
                if (step <= i + 1) val += 25.0 / ((i + 1) - step + 1);
                if (cell == end) val += 35.0;
                worst = std::max(worst, val);
            } else {
                future += 9.0 / (step + 1);
            }
        }
        return worst + std::min(28.0, future);
    }

    double simultaneous_root_score(const Move& mine, const std::vector<Move>& enemy_moves) {
        if (enemy_moves.empty()) return 115.0;
        double worst = 1e100;
        for (const Move& enemy : enemy_moves) {
            bool ma = true, ea = true;
            int ms = 0, es = 0;
            int maxlen = std::max(mine.path.size(), enemy.path.size());
            for (int k = 0; k < maxlen; k++) {
                bool mh = ma && k < (int)mine.path.size();
                bool eh = ea && k < (int)enemy.path.size();
                int mc = mh ? mine.path[k] : -1;
                int ec = eh ? enemy.path[k] : -2;
                bool mdie = false, edie = false;
                if (mh && claimed[mc / GRID][mc % GRID]) mdie = true;
                if (eh && claimed[ec / GRID][ec % GRID]) edie = true;
                if (mh && eh && mc == ec) {
                    mdie = true;
                    edie = true;
                }
                if (!mdie && mh) {
                    // The opponent reaches this cell earlier in the same move.
                    for (int j = 0; j < k && j < (int)enemy.path.size(); j++)
                        if (enemy.path[j] == mc) { mdie = true; break; }
                }
                if (!edie && eh) {
                    for (int j = 0; j < k && j < (int)mine.path.size(); j++)
                        if (mine.path[j] == ec) { edie = true; break; }
                }
                if (mdie) ma = false;
                if (edie) ea = false;
                if (ma && mh) ms++;
                if (ea && eh) es++;
                if ((!ma || !mh) && (!ea || !eh)) break;
            }
            double sc = 2.8 * (ms - es);
            if (!ma) sc -= 260.0;
            if (!ea) sc += 210.0;
            if (ma && ea && mine.path.back() == enemy.path.back()) sc -= 80.0;
            worst = std::min(worst, sc);
        }
        return worst;
    }

    double openness(int x, int y) {
        int runs[4];
        for (int d = 0; d < 4; d++) {
            int cx = x, cy = y, run = 0;
            for (int k = 0; k < 10; k++) {
                cx += DX[d]; cy += DY[d];
                if (!in_grid(cx, cy) || claimed[cy][cx]) break;
                run++;
            }
            runs[d] = run;
        }
        int mn = std::min(std::min(runs[0], runs[1]), std::min(runs[2], runs[3]));
        int sum = runs[0] + runs[1] + runs[2] + runs[3];
        return 2.0 * mn + 0.25 * sum;
    }

    double trap_penalty(int x, int y, const std::array<int, 3>& mob, int rch) {
        double p = 0.0;
        if (rch > 8) {
            if (mob[0] <= 1) p += 82.0;
            else if (mob[0] == 2) p += 18.0;
        }
        int edge = std::min(std::min(x, GRID - 1 - x), std::min(y, GRID - 1 - y));
        int near_edges = (x <= 1) + (x >= GRID - 2) + (y <= 1) + (y >= GRID - 2);
        if (edge == 0 && mob[0] <= 2 && rch > 16) p += 22.0;
        if (near_edges >= 2 && mob[0] <= 2 && rch > 10) p += 36.0;
        return p;
    }

    static int edge_dist(int x, int y) {
        return std::min(std::min(x, GRID - 1 - x), std::min(y, GRID - 1 - y));
    }

    static int edge_side(int x, int y) {
        if (y <= 1) return 1;
        if (y >= GRID - 2) return 2;
        if (x <= 1) return 3;
        if (x >= GRID - 2) return 4;
        return 0;
    }

    int inward_escape_count(int x, int y) {
        int cur_edge = edge_dist(x, y);
        int cnt = 0;
        for (const Move& m : moves_from(x, y)) {
            if (edge_dist(m.nx, m.ny) > cur_edge) cnt++;
        }
        return cnt;
    }

    int near_edge_inward_escape_count(int x, int y) {
        int cnt = 0;
        for (const Move& m : moves_from(x, y)) {
            if (edge_dist(m.nx, m.ny) >= 3) cnt++;
        }
        return cnt;
    }

    static bool near_corner(int x, int y) {
        return (x <= 2 || x >= GRID - 3) && (y <= 2 || y >= GRID - 3);
    }

    void unit_bfs(int sx, int sy, int dist[N]) {
        for (int i = 0; i < N; i++) dist[i] = INF;
        if (!in_grid(sx, sy) || claimed[sy][sx]) return;
        std::queue<int> q;
        int s = sy * GRID + sx;
        dist[s] = 0; q.push(s);
        while (!q.empty()) {
            int c = q.front(); q.pop();
            int x = c % GRID, y = c / GRID;
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (!in_grid(nx, ny) || claimed[ny][nx]) continue;
                int nb = ny * GRID + nx;
                if (dist[nb] == INF) {
                    dist[nb] = dist[c] + 1;
                    q.push(nb);
                }
            }
        }
    }

    std::pair<double, double> voronoi_split(int ax, int ay, int bx, int by) {
        int da[N], db[N];
        unit_bfs(ax, ay, da);
        unit_bfs(bx, by, db);
        double mine = 0.0, enemy = 0.0;
        for (int i = 0; i < N; i++) {
            if (da[i] == INF && db[i] == INF) continue;
            if (da[i] < db[i]) mine += 1.0;
            else if (db[i] < da[i]) enemy += 1.0;
            else { mine += 0.5; enemy += 0.5; }
        }
        return {mine, enemy};
    }

    void jump_claim_dist(int sx, int sy, int dist[N], int max_turns = 12) {
        for (int i = 0; i < N; i++) dist[i] = INF;
        if (!in_grid(sx, sy)) return;
        int seen_end[N];
        for (int i = 0; i < N; i++) seen_end[i] = INF;
        std::queue<std::array<int, 3>> q;
        int s = sy * GRID + sx;
        seen_end[s] = 0; dist[s] = 0; q.push({sx, sy, 0});
        while (!q.empty()) {
            auto cur = q.front(); q.pop();
            int x = cur[0], y = cur[1], turn = cur[2];
            if (turn >= max_turns) continue;
            for (int d = 0; d < 4; d++) {
                Move m = simulate(x, y, DX[d], DY[d]);
                if (!m.alive || m.path.empty()) continue;
                int nt = turn + 1;
                for (int c : m.path) dist[c] = std::min(dist[c], nt);
                int e = m.ny * GRID + m.nx;
                if (nt < seen_end[e]) {
                    seen_end[e] = nt;
                    q.push({m.nx, m.ny, nt});
                }
            }
        }
    }

    std::pair<double, double> jump_voronoi_split(int ax, int ay, int bx, int by) {
        int da[N], db[N];
        jump_claim_dist(ax, ay, da);
        jump_claim_dist(bx, by, db);
        double mine = 0.0, enemy = 0.0;
        for (int i = 0; i < N; i++) {
            if (da[i] == INF && db[i] == INF) continue;
            if (da[i] < db[i]) mine += 1.0;
            else if (db[i] < da[i]) enemy += 1.0;
            else { mine += 0.5; enemy += 0.5; }
        }
        return {mine, enemy};
    }

    double usable_space(int sx, int sy, int cap = 400) {
        if (!in_grid(sx, sy) || claimed[sy][sx]) return 0.0;
        int idx[N], low[N], parent[N], subtree[N], it[N];
        for (int i = 0; i < N; i++) {
            idx[i] = -1; low[i] = 0; parent[i] = -1;
            subtree[i] = 0; it[i] = 0;
        }
        int root = sy * GRID + sx, counter = 0;
        std::vector<int> stack, order;
        stack.push_back(root);
        idx[root] = low[root] = counter++;
        subtree[root] = 1; order.push_back(root);
        while (!stack.empty()) {
            int node = stack.back();
            int x = node % GRID, y = node / GRID;
            bool advanced = false;
            while (it[node] < 4) {
                int d = it[node]++;
                int nx = x + DX[d], ny = y + DY[d];
                if (!in_grid(nx, ny) || claimed[ny][nx]) continue;
                int nb = ny * GRID + nx;
                if (idx[nb] < 0) {
                    parent[nb] = node;
                    idx[nb] = low[nb] = counter++;
                    subtree[nb] = 1; order.push_back(nb);
                    stack.push_back(nb);
                    advanced = true;
                    break;
                } else if (nb != parent[node]) {
                    low[node] = std::min(low[node], idx[nb]);
                }
            }
            if (advanced) continue;
            stack.pop_back();
            int p = parent[node];
            if (p >= 0) {
                low[p] = std::min(low[p], low[node]);
                subtree[p] += subtree[node];
            }
        }
        int comp_size = (int)order.size();
        if (comp_size <= 1) return (double)comp_size;
        double discount = 0.0;
        std::vector<int> root_children;
        for (int node : order) {
            int p = parent[node];
            if (p < 0) continue;
            if (p == root) {
                root_children.push_back(node);
                continue;
            }
            if (low[node] >= idx[p]) {
                int sz = subtree[node], other = comp_size - sz;
                discount += 0.5 * std::min(sz, other);
            }
        }
        if (root_children.size() > 1) {
            std::vector<int> sizes;
            for (int c : root_children) sizes.push_back(subtree[c]);
            std::sort(sizes.begin(), sizes.end(), std::greater<int>());
            int rest = 0;
            for (size_t i = 1; i < sizes.size(); i++) rest += sizes[i];
            discount += 0.5 * rest;
        }
        double val = comp_size - discount;
        if (val < 1.0) val = 1.0;
        return std::min(val, (double)cap);
    }

    double rollout(int x, int y, int enx, int eny) {
        (void)enx; (void)eny;
        std::vector<int> added;
        double total = 0.0;
        int cx = x, cy = y;
        for (int budget = 0; budget < 26; budget++) {
            double best_v = -1e100;
            Move best;
            for (int d = 0; d < 4; d++) {
                Move m = simulate(cx, cy, DX[d], DY[d]);
                if (!m.alive || m.steps == 0) continue;
                mark_path(m.path, true);
                auto mob = mobility(m.nx, m.ny);
                double v = m.steps + 0.55 * openness(m.nx, m.ny)
                         + 2.4 * mob[0] + 0.25 * mob[1];
                mark_path(m.path, false);
                if (v > best_v) { best_v = v; best = std::move(m); }
            }
            if (best_v < -1e90) break;
            mark_path(best.path, true);
            for (int c : best.path) added.push_back(c);
            total += best.steps;
            cx = best.nx; cy = best.ny;
        }
        for (int c : added) claimed[c / GRID][c % GRID] = false;
        return total;
    }

    int reachable(int sx, int sy) {
        if (!in_grid(sx, sy) || claimed[sy][sx]) return 0;
        bool seen[GRID][GRID] = {};
        std::vector<int> st;
        st.push_back(sy * GRID + sx);
        seen[sy][sx] = true;
        int cnt = 0;
        while (!st.empty()) {
            int c = st.back(); st.pop_back(); cnt++;
            int x = c % GRID, y = c / GRID;
            for (int d = 0; d < 4; d++) {
                int nx = x + DX[d], ny = y + DY[d];
                if (in_grid(nx, ny) && !claimed[ny][nx] && !seen[ny][nx]) {
                    seen[ny][nx] = true;
                    st.push_back(ny * GRID + nx);
                }
            }
        }
        return cnt;
    }

    double leaf_value(int nx, int ny, int enx, int eny) {
        double ro = rollout(nx, ny, enx, eny);
        double space = usable_space(nx, ny);
        auto vorp = voronoi_split(nx, ny, enx, eny);
        auto mob = mobility(nx, ny);
        int rch = reachable(nx, ny);
        double vor = vorp.first - vorp.second;
        return ro + 0.72 * space + 0.82 * vor
             + 7.5 * mob[0] + 0.45 * mob[1] + 0.8 * mob[2]
             - trap_penalty(nx, ny, mob, rch);
    }

    double solo_survival_value(int nx, int ny) {
        auto mob = mobility(nx, ny);
        int rch = reachable(nx, ny);
        double space = usable_space(nx, ny, 700);
        double ro = rollout(nx, ny, -1, -1);
        return 2.4 * space + 0.28 * rch + 3.2 * ro
             + 22.0 * mob[0] + 1.1 * mob[1] + 1.6 * mob[2]
             - 1.35 * trap_penalty(nx, ny, mob, rch);
    }

    double second_escape_value(int x, int y) {
        std::vector<Move> next = moves_from(x, y);
        if (next.empty()) return -260.0;
        double best = -1e100;
        for (Move& nm : next) {
            mark_path(nm.path, true);
            auto mob2 = mobility(nm.nx, nm.ny);
            int r2 = reachable(nm.nx, nm.ny);
            double sc = 1.2 * nm.steps + 34.0 * mob2[0] + 0.16 * std::min(r2, 180)
                      + 0.8 * mob2[1] + 1.2 * mob2[2]
                      - 1.15 * trap_penalty(nm.nx, nm.ny, mob2, r2);
            mark_path(nm.path, false);
            best = std::max(best, sc);
        }
        return best;
    }

    int robust_second_escape_count(int x, int y, int base_reach) {
        std::vector<Move> next = moves_from(x, y);
        if (next.empty()) return 0;
        int good = 0;
        int need = std::max(16, std::min(60, base_reach / 4));
        for (Move& nm : next) {
            mark_path(nm.path, true);
            auto mob2 = mobility(nm.nx, nm.ny);
            int r2 = reachable(nm.nx, nm.ny);
            if (r2 >= need && mob2[0] >= 2) good++;
            mark_path(nm.path, false);
        }
        return good;
    }

    double opponent_cut_risk(int x, int y, const std::vector<Move>& enemy_moves, int base_reach) {
        if (enemy_moves.empty() || base_reach <= 0) return 0.0;
        int worst_loss = 0;
        for (const Move& em : enemy_moves) {
            mark_path(em.path, true);
            int after = reachable(x, y);
            mark_path(em.path, false);
            int loss = base_reach - after;
            if (loss > worst_loss) worst_loss = loss;
        }
        if (worst_loss <= 12) return 0.0;
        return std::min(95.0, 0.42 * worst_loss);
    }

    std::vector<Move> ordered_moves(std::vector<Move> moves) {
        std::vector<std::pair<double, int>> scored;
        for (int i = 0; i < (int)moves.size(); i++) {
            Move& m = moves[i];
            mark_path(m.path, true);
            auto mob = mobility(m.nx, m.ny);
            double score = m.steps + 0.35 * openness(m.nx, m.ny)
                         + 2.0 * mob[0] + 0.25 * mob[1] + 0.4 * mob[2];
            mark_path(m.path, false);
            scored.push_back({score, i});
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<Move> out;
        for (auto& p : scored) out.push_back(std::move(moves[p.second]));
        return out;
    }

    double search(int x, int y, int enx, int eny, int depth,
                  double alpha = -1000000.0, double beta = 1000000.0) {
        if (Clock::now() > deadline) return leaf_value(x, y, enx, eny);
        std::vector<Move> my_moves = ordered_moves(moves_from(x, y));
        if (my_moves.empty()) return -1000.0;
        double best_val = -1000000.0;
        for (Move& m : my_moves) {
            mark_path(m.path, true);
            double val;
            if (depth <= 1) {
                val = m.steps + leaf_value(m.nx, m.ny, enx, eny);
            } else {
                std::vector<Move> en_moves = ordered_moves(moves_from(enx, eny));
                if (en_moves.empty()) {
                    val = m.steps + 80.0 + search(m.nx, m.ny, enx, eny,
                                                   depth - 1, alpha - m.steps, beta);
                } else {
                    double worst = 1000000.0;
                    for (Move& em : en_moves) {
                        mark_path(em.path, true);
                        double sub = search(m.nx, m.ny, em.nx, em.ny, depth - 1,
                                            alpha - m.steps, beta - m.steps);
                        mark_path(em.path, false);
                        worst = std::min(worst, sub);
                        if (m.steps + worst <= alpha || Clock::now() > deadline) break;
                    }
                    val = m.steps + worst;
                }
            }
            mark_path(m.path, false);
            best_val = std::max(best_val, val);
            if (best_val >= beta) break;
            alpha = std::max(alpha, best_val);
            if (Clock::now() > deadline) break;
        }
        return best_val > -999999.0 ? best_val : -1000.0;
    }

    void claim_path(int ox, int oy, int nx, int ny) {
        int sx = (nx > ox) - (nx < ox);
        int sy = (ny > oy) - (ny < oy);
        int cx = ox, cy = oy;
        for (int guard = 0; (cx != nx || cy != ny) && guard < 2 * GRID; guard++) {
            cx += sx; cy += sy;
            if (in_grid(cx, cy)) claimed[cy][cx] = true;
        }
    }

    int pick_depth(int free_cells) {
        double left = GAME_BUDGET - spent;
        if (left < 0.6) return 1;
        if (left < 2.0 || free_cells > 520) return 2;
        if (left < 4.5 && free_cells > 260) return 2;
        if (left >= 5.0 && free_cells <= 480) return 4;
        return 3;
    }

    char decide() {
        auto t0 = Clock::now();
        ply++;
        if (!in_grid(mx, my)) return 'u';

        int free_cells = 0;
        for (int y = 0; y < GRID; y++)
            for (int x = 0; x < GRID; x++)
                if (!claimed[y][x]) free_cells++;

        if (char sc = scripted_opening()) return sc;
        if (char ob = opening_book(free_cells)) return ob;

        const bool early_contest =
            free_cells > OPEN_FREE_MIN && ply <= OPEN_EARLY_PLIES;

        if (have_enemy_seen && ex == last_ex && ey == last_ey) enemy_stale++;
        else enemy_stale = 0;
        have_enemy_seen = true;
        last_ex = ex; last_ey = ey;
        bool enemy_active = enemy_stale < 3 && in_grid(ex, ey);
        int model_ex = enemy_active ? ex : -1;
        int model_ey = enemy_active ? ey : -1;
        int cur_side = edge_side(mx, my);
        if (cur_side != 0 && cur_side == last_edge_side) edge_streak++;
        else edge_streak = (cur_side != 0) ? 1 : 0;
        last_edge_side = cur_side;

        auto danger = enemy_active ? enemy_future_reach(ex, ey)
                                   : std::array<std::pair<int, int>, N>{};
        if (!enemy_active)
            for (int i = 0; i < N; i++) danger[i] = {INF, INF};
        int depth = pick_depth(free_cells);
        const bool opening_sprint = (free_cells >= 750 && ply <= 10);
        double left = std::max(0.02, GAME_BUDGET - spent);
        int est_turns_left = std::max(8, free_cells / 8);
        double move_budget = std::min(PER_MOVE_SOFT,
                                      std::max(0.030, 2.0 * left / est_turns_left));
        deadline = t0 + std::chrono::duration_cast<Clock::duration>(
                            std::chrono::duration<double>(move_budget));

        double geom = phase_geom(free_cells, ply);
        struct Cand {
            double val;
            int rch, exits, steps, surv;
            char name;
            bool collide;
        };
        std::vector<Cand> cands;
        std::vector<std::pair<int, char>> forced;
        for (int d = 0; d < 4; d++) {
            Move m = simulate(mx, my, DX[d], DY[d]);
            forced.push_back({m.steps, DCH[d]});
            if (!m.alive || m.steps <= 0) continue;
            std::vector<Move> ems = enemy_active ? moves_from(ex, ey) : std::vector<Move>();
            mark_path(m.path, true);
            auto mob = mobility(m.nx, m.ny);
            int en_steps = 0;
            for (const Move& em : ems) en_steps += std::min(em.steps, 6);
            double val;
            if (!enemy_active) {
                val = 0.15 * m.steps + solo_survival_value(m.nx, m.ny);
            } else {
                val = m.steps + (depth > 1
                    ? search(m.nx, m.ny, model_ex, model_ey, depth - 1)
                    : leaf_value(m.nx, m.ny, model_ex, model_ey));
            }
            int rch = reachable(m.nx, m.ny);
            val += std::min(rch, 150) * 0.12 + 5.5 * mob[0]
                 + 0.35 * mob[1] + 0.7 * mob[2];
            if (opening_sprint) val += 2.0 * m.steps;
            double esc2 = second_escape_value(m.nx, m.ny);
            if (esc2 < 0.0) val += esc2;
            else val += 0.22 * esc2;
            if (geom > 0.0) {
                if (enemy_active && free_cells <= 820 && rch > 24 && mob[0] <= 2) {
                    int robust_escapes = robust_second_escape_count(m.nx, m.ny, rch);
                    if (robust_escapes == 0) val -= geom * 118.0;
                    else if (robust_escapes == 1) val -= geom * 28.0;
                }
                if (enemy_active && free_cells <= 760)
                    val -= geom * opponent_cut_risk(m.nx, m.ny, ems, rch);
                val -= geom * trap_penalty(m.nx, m.ny, mob, rch);
                if (enemy_active) {
                    bool top_bottom = (m.ny == 0 || m.ny == GRID - 1);
                    if (top_bottom && mob[0] <= 2 && rch > 12) val -= geom * 62.0;
                    if (top_bottom && (m.nx <= 1 || m.nx >= GRID - 2) && rch > 8)
                        val -= geom * 34.0;
                    if (edge_dist(m.nx, m.ny) == 0) {
                        int escapes = inward_escape_count(m.nx, m.ny);
                        if (escapes >= 2) val += 20.0 + geom * 12.0;
                        else if (escapes == 0 && rch > 10) val -= geom * 92.0;
                        else if (escapes == 1 && rch > 10) val -= geom * 24.0;
                    }
                    if (near_corner(m.nx, m.ny) && rch > 6) {
                        int escapes = inward_escape_count(m.nx, m.ny);
                        if (escapes < 2) val -= geom * 145.0;
                        else val -= geom * 38.0;
                    }
                    if (edge_dist(m.nx, m.ny) <= 2 && rch > 10) {
                        int deep_escapes = near_edge_inward_escape_count(m.nx, m.ny);
                        if (deep_escapes == 0) val -= geom * 88.0;
                        else if (deep_escapes == 1 && mob[0] <= 2) val -= geom * 18.0;
                    }
                    int next_side = edge_side(m.nx, m.ny);
                    if (edge_streak >= 2 && cur_side != 0 && next_side == cur_side)
                        val -= geom * 42.0 * std::min(edge_streak, 4);
                }
            }
            if (enemy_active && free_cells <= 520 && Clock::now() < deadline) {
                auto jv = jump_voronoi_split(m.nx, m.ny, model_ex, model_ey);
                val += 0.46 * (jv.first - jv.second);
            }
            if (enemy_active) {
                if (ems.empty()) val += 95.0;
                else val += (4 - (int)ems.size()) * 6.0 - 0.18 * en_steps;
                val += simultaneous_root_score(m, ems);
            }
            mark_path(m.path, false);
            val -= (early_contest ? 0.35 : 1.0) * collision_penalty(m.path, danger);

            int surv = survival_depth(m.nx, m.ny, SURV_CAP);
            if (!opening_sprint && surv < SURV_CAP && rch <= 50)
                val -= (SURV_CAP - surv) * SURV_PEN;
            bool collide = path_lethal(m.path, danger);
            cands.push_back({val, rch, mob[0], m.steps, surv, DCH[d], collide});
        }
        spent += std::chrono::duration<double>(Clock::now() - t0).count();
        if (cands.empty()) {
            return std::max_element(forced.begin(), forced.end())->second;
        }

        bool any_clean = false;
        for (const Cand& c : cands)
            if (!c.collide) {
                any_clean = true;
                break;
            }
        if (!early_contest && any_clean && geom >= 0.35 && free_cells < 780) {
            std::vector<Cand> filtered;
            for (const Cand& c : cands)
                if (!c.collide) filtered.push_back(c);
            cands = std::move(filtered);
        }

        int max_surv = cands[0].surv;
        for (const Cand& c : cands) max_surv = std::max(max_surv, c.surv);
        if (max_surv >= 8 && free_cells < 700) {
            std::vector<Cand> safe;
            for (const Cand& c : cands)
                if (c.surv >= max_surv - 2) safe.push_back(c);
            if (!safe.empty()) cands = std::move(safe);
        }

        double best = cands[0].val;
        for (const Cand& c : cands) best = std::max(best, c.val);
        double window = std::max(1.0, std::abs(best) * 0.03);
        Cand pick = cands[0];
        bool have = false;
        for (const Cand& c : cands) {
            if (c.val < best - window) continue;
            if (!have || std::array<int, 4>{c.surv, c.steps, c.exits, c.rch} >
                         std::array<int, 4>{pick.surv, pick.steps, pick.exits, pick.rch}) {
                pick = c;
                have = true;
            }
        }
        return pick.name;
    }
};

static std::vector<std::string> split_ws(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

static bool all_digits(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (c < '0' || c > '9') return false;
    return true;
}

static std::vector<int> grid_values(const std::string& line) {
    std::vector<std::string> parts = split_ws(line);
    std::vector<int> vals;
    for (const std::string& tok : parts) {
        if (all_digits(tok) && (tok.size() > 2 || (parts.size() == 1 && tok.size() > 1))) {
            for (char c : tok) vals.push_back(c - '0');
        } else {
            try {
                size_t used = 0;
                int v = std::stoi(tok, &used);
                if (used == tok.size()) vals.push_back(v);
                else for (char c : tok) if ('0' <= c && c <= '9') vals.push_back(c - '0');
            } catch (...) {
                for (char c : tok) if ('0' <= c && c <= '9') vals.push_back(c - '0');
            }
        }
    }
    return vals;
}

static std::vector<int> parse_ints(const std::string& line) {
    std::vector<int> out;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
        try { out.push_back(std::stoi(tok)); } catch (...) {}
    }
    return out;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bot bot;
    for (int y = 0; y < GRID; y++)
        for (int x = 0; x < GRID; x++) {
            bot.grid[y][x] = 1;
            bot.claimed[y][x] = false;
        }

    std::vector<int> digits, pending;
    std::string line;
    while ((int)digits.size() < N && std::getline(std::cin, line)) {
        std::vector<int> vals = grid_values(line);
        int need = N - (int)digits.size();
        for (int i = 0; i < (int)vals.size() && i < need; i++) digits.push_back(vals[i]);
        if ((int)vals.size() > need) {
            for (int i = need; i < (int)vals.size(); i++) pending.push_back(vals[i]);
            break;
        }
    }
    if ((int)digits.size() < N) return 0;
    for (int i = 0; i < N; i++) bot.grid[i / GRID][i % GRID] = digits[i];

    std::vector<int> pos = pending;
    while ((int)pos.size() < 4 && std::getline(std::cin, line)) {
        std::vector<int> extra = parse_ints(line);
        pos.insert(pos.end(), extra.begin(), extra.end());
    }
    if ((int)pos.size() < 4) return 0;
    bot.mx = pos[0]; bot.my = pos[1]; bot.ex = pos[2]; bot.ey = pos[3];
    bot.start_mx = bot.mx;
    bot.start_my = bot.my;
    if (Bot::in_grid(bot.mx, bot.my)) bot.claimed[bot.my][bot.mx] = true;
    if (Bot::in_grid(bot.ex, bot.ey)) bot.claimed[bot.ey][bot.ex] = true;

    char d = bot.decide();
    std::cout << d << '\n' << std::flush;
    int pmx = bot.mx, pmy = bot.my, pex = bot.ex, pey = bot.ey;
    while (std::getline(std::cin, line)) {
        std::vector<int> v = parse_ints(line);
        if ((int)v.size() < 4) break;
        bot.mx = v[0]; bot.my = v[1]; bot.ex = v[2]; bot.ey = v[3];
        bot.claim_path(pmx, pmy, bot.mx, bot.my);
        bot.claim_path(pex, pey, bot.ex, bot.ey);
        if (Bot::in_grid(bot.mx, bot.my)) bot.claimed[bot.my][bot.mx] = true;
        if (Bot::in_grid(bot.ex, bot.ey)) bot.claimed[bot.ey][bot.ex] = true;
        pmx = bot.mx; pmy = bot.my; pex = bot.ex; pey = bot.ey;
        d = bot.decide();
        std::cout << d << '\n' << std::flush;
    }
    return 0;
}
