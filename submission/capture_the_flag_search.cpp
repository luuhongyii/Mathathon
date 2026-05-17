// Capture the Flag - Mathathon C++ submission (SEARCH build).
// Replaces the pile of greedy heuristics with a forward simulator + a
// depth-limited search over our own moves. The 3 other players are modelled
// with a fast greedy policy (attack; an enemy chases our flag-carrier). One
// search mechanism handles deadlocks, corner traps and defence uniformly:
// it simply推演s the next several turns and avoids lines that end badly.
//
// Submit this source via Submit C++/Binary. Never writes to stderr.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static constexpr int SIZE = 29;
static constexpr int N = SIZE * SIZE;
static constexpr int INF = 1 << 28;

struct Dir {
    int dx, dy;
    char c;
};
static constexpr array<Dir, 5> MV{{
    {0, -1, 'u'}, {0, 1, 'd'}, {-1, 0, 'l'}, {1, 0, 'r'}, {0, 0, 's'}
}};

static int idx(int x, int y) { return y * SIZE + x; }

static bool inOasis(int x, int y) {
    return 12 <= x && x <= 16 && 12 <= y && y <= 16;
}

// 'B' = blue territory, 'R' = red, 'N' = neutral (oasis + the y==14 band).
static char territory(int x, int y) {
    if (inOasis(x, y)) return 'N';
    if (y <= 13) return 'B';
    if (y >= 15) return 'R';
    return 'N';
}

// Full game state for the forward simulator. Player 0/1 = our team, 2/3 =
// enemies. Coordinates are absolute; team colours come from setup.
struct GState {
    array<int, 4> x, y, h, respawn;
    array<bool, 4> alive, carry;   // carry = holds the OTHER team's flag
    array<int, 2> flagx, flagy;    // flag of team 0 (ours) and team 1
    array<int, 2> fcar;            // player carrying that flag, or -1
    int winner = -1;               // -1 none, 0 us, 1 enemy, 2 both
};

struct Bot {
    bool ready = false;
    array<unsigned char, N> freeCell{};
    array<vector<int>, N> adj{};

    bool blue = true;
    char myTerr = 'B', enemyTerr = 'R';
    int myCorner = 0, enemyCorner = 0;
    int myRespawn = 0, enemyRespawn = 0;

    vector<int> dMyFlag, dEnemyFlag, dMyTerr, dEnemyTerr, dOasis;

    char lastMove = 's';
    uint32_t rng = 2463534242u;

    // Search budget as a NODE COUNT, not wall-clock: deterministic and
    // machine-independent, so the per-turn cost is fixed regardless of how
    // fast or slow the judge machine is. Tune NODE_BUDGET to the platform's
    // time limit (~3000 nodes is well under 1 ms/turn).
    static constexpr long long NODE_BUDGET = 3000;
    mutable bool timeUp = false;
    mutable long long nodeCount = 0;

    uint32_t rnd() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    }

    void setupBoard(const string& board) {
        for (int i = 0; i < N; ++i)
            freeCell[i] = (i < (int)board.size() && board[i] == '#') ? 0 : 1;
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                int here = idx(x, y);
                adj[here].clear();
                if (!freeCell[here]) continue;
                for (int d = 0; d < 4; ++d) {
                    int nx = x + MV[d].dx, ny = y + MV[d].dy;
                    if (0 <= nx && nx < SIZE && 0 <= ny && ny < SIZE &&
                        freeCell[idx(nx, ny)])
                        adj[here].push_back(idx(nx, ny));
                }
            }
    }

    vector<int> bfs(const vector<int>& src) {
        vector<int> dist(N, INF);
        queue<int> q;
        for (int s : src)
            if (0 <= s && s < N && freeCell[s] && dist[s] == INF) {
                dist[s] = 0;
                q.push(s);
            }
        while (!q.empty()) {
            int c = q.front();
            q.pop();
            for (int nb : adj[c])
                if (dist[nb] == INF) {
                    dist[nb] = dist[c] + 1;
                    q.push(nb);
                }
        }
        return dist;
    }

    void setupTeam(int my) {
        blue = (my <= 14);
        myTerr = blue ? 'B' : 'R';
        enemyTerr = blue ? 'R' : 'B';
        myCorner = blue ? idx(0, 0) : idx(28, 28);
        enemyCorner = blue ? idx(28, 28) : idx(0, 0);
        myRespawn = blue ? idx(28, 0) : idx(0, 28);
        enemyRespawn = blue ? idx(0, 28) : idx(28, 0);
        dMyFlag = bfs({myCorner});
        dEnemyFlag = bfs({enemyCorner});
        vector<int> mt, et, oa;
        for (int y = 0; y < SIZE; ++y)
            for (int x = 0; x < SIZE; ++x) {
                int p = idx(x, y);
                if (!freeCell[p]) continue;
                if (territory(x, y) == myTerr) mt.push_back(p);
                if (territory(x, y) == enemyTerr) et.push_back(p);
                if (inOasis(x, y)) oa.push_back(p);
            }
        dMyTerr = bfs(mt);
        dEnemyTerr = bfs(et);
        dOasis = bfs(oa);
        ready = true;
    }

    static int teamOf(int p) { return p < 2 ? 0 : 1; }
    char homeChar(int t) const { return t == 0 ? myTerr : enemyTerr; }

    // Would a player of team `t` standing on (x,y) be caught? (On the enemy
    // team's home soil, Chebyshev-1 of a living opponent.)
    bool catchable(const GState& s, int t, int x, int y) const {
        char et = (t == 0) ? enemyTerr : myTerr;
        if (territory(x, y) != et) return false;
        for (int q = 0; q < 4; ++q) {
            if (!s.alive[q] || teamOf(q) == t) continue;
            if (max(abs(s.x[q] - x), abs(s.y[q] - y)) <= 1) return true;
        }
        return false;
    }

    // Fast greedy policy for the 3 non-searched players.
    int modelMove(const GState& s, int p) const {
        if (!s.alive[p]) return 4;
        int t = teamOf(p);
        bool chase = false;
        int cx = 0, cy = 0;
        const vector<int>* fld = nullptr;
        if (s.carry[p]) {
            fld = (t == 0) ? &dMyTerr : &dEnemyTerr;
        } else if (t == 1 && s.fcar[1] >= 0) {
            // an enemy: our team holds the enemy flag — hunt the carrier.
            chase = true;
            cx = s.x[s.fcar[1]];
            cy = s.y[s.fcar[1]];
        } else {
            // No carrier to chase. A player DEFENDS its own flag when the
            // other team has a real attacker bearing down on it and this
            // player is the closer of its two team-mates — the other one
            // keeps attacking. Symmetric across both teams: it both stops the
            // search walking our attacker into a defender, AND splits our two
            // bots into attacker/defender by position (no sticky roles).
            bool defend = false;
            int oppTeam = 1 - t;
            const vector<int>& flagField = (t == 0) ? dMyFlag : dEnemyFlag;
            int atk = -1, atkD = INF;
            for (int q = 0; q < 4; ++q) {
                if (teamOf(q) != oppTeam || !s.alive[q] || s.carry[q]) continue;
                int dd = flagField[idx(s.x[q], s.y[q])];
                if (dd < atkD) { atkD = dd; atk = q; }
            }
            if (atk >= 0 && atkD <= 24) {
                int mate = (t == 0) ? (p == 0 ? 1 : 0) : (p == 2 ? 3 : 2);
                int myD = max(abs(s.x[p] - s.x[atk]), abs(s.y[p] - s.y[atk]));
                int mateD = s.alive[mate]
                    ? max(abs(s.x[mate] - s.x[atk]), abs(s.y[mate] - s.y[atk]))
                    : INF;
                if (myD < mateD || (myD == mateD && p < mate)) {
                    defend = true;
                    chase = true;
                    cx = s.x[atk];
                    cy = s.y[atk];
                }
            }
            if (!defend) fld = (t == 0) ? &dEnemyFlag : &dMyFlag;
        }
        if (!chase && fld != nullptr && !inOasis(s.x[p], s.y[p])) {
            int doa = dOasis[idx(s.x[p], s.y[p])];
            if (doa < INF && s.h[p] < 2 * doa + 16) fld = &dOasis;
        }
        int best = 4, bestU = 9, bestD = INF;
        for (int d = 0; d < 5; ++d) {
            int nx = s.x[p] + MV[d].dx, ny = s.y[p] + MV[d].dy;
            if (d != 4 && (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE ||
                           !freeCell[idx(nx, ny)]))
                continue;
            int dv = chase ? max(abs(nx - cx), abs(ny - cy))
                           : (*fld)[idx(nx, ny)];
            int u = catchable(s, t, nx, ny) ? 1 : 0;
            if (u < bestU || (u == bestU && dv < bestD)) {
                bestU = u;
                bestD = dv;
                best = d;
            }
        }
        return best;
    }

    // Advance the state by one turn given all four moves. Mirrors the real
    // rules exactly (movement, pickup, hydration, deaths, win, respawn).
    GState step(const GState& s, const array<int, 4>& mv) const {
        GState n = s;
        for (int p = 0; p < 4; ++p) {
            if (!n.alive[p]) continue;
            int nx = s.x[p] + MV[mv[p]].dx, ny = s.y[p] + MV[mv[p]].dy;
            if (0 <= nx && nx < SIZE && 0 <= ny && ny < SIZE &&
                freeCell[idx(nx, ny)]) {
                n.x[p] = nx;
                n.y[p] = ny;
            }
        }
        // flag pickup (lowest index wins)
        for (int t = 0; t < 2; ++t) {
            if (n.fcar[t] >= 0) continue;
            for (int p = 0; p < 4; ++p) {
                if (n.alive[p] && teamOf(p) != t && !n.carry[p] &&
                    n.x[p] == n.flagx[t] && n.y[p] == n.flagy[t]) {
                    n.fcar[t] = p;
                    n.carry[p] = true;
                    break;
                }
            }
        }
        // hydration
        for (int p = 0; p < 4; ++p) {
            if (!n.alive[p]) continue;
            char tr = territory(n.x[p], n.y[p]);
            n.h[p] -= (tr == homeChar(teamOf(p))) ? 2 : 1;
            if (inOasis(n.x[p], n.y[p])) n.h[p] = 140;
        }
        // deaths
        array<bool, 4> dead{{false, false, false, false}};
        for (int p = 0; p < 4; ++p) {
            if (!n.alive[p]) continue;
            if (n.h[p] <= 0) { dead[p] = true; continue; }
            int t = teamOf(p);
            char et = (t == 0) ? enemyTerr : myTerr;
            if (territory(n.x[p], n.y[p]) == et) {
                for (int q = 0; q < 4; ++q) {
                    if (n.alive[q] && teamOf(q) != t &&
                        max(abs(n.x[q] - n.x[p]), abs(n.y[q] - n.y[p])) <= 1) {
                        dead[p] = true;
                        break;
                    }
                }
            }
        }
        for (int p = 0; p < 4; ++p) {
            if (!dead[p]) continue;
            n.alive[p] = false;
            if (n.carry[p]) {
                n.carry[p] = false;
                int ft = 1 - teamOf(p);
                n.flagx[ft] = (ft == 0 ? myCorner : enemyCorner) % SIZE;
                n.flagy[ft] = (ft == 0 ? myCorner : enemyCorner) / SIZE;
                n.fcar[ft] = -1;
            }
            n.x[p] = -1;
            n.y[p] = -1;
            n.h[p] = 0;
            n.respawn[p] = 30;
        }
        // carried flags follow the carrier
        for (int t = 0; t < 2; ++t)
            if (n.fcar[t] >= 0 && n.alive[n.fcar[t]]) {
                n.flagx[t] = n.x[n.fcar[t]];
                n.flagy[t] = n.y[n.fcar[t]];
            }
        // win check
        bool win[2] = {false, false};
        for (int p = 0; p < 4; ++p) {
            if (n.alive[p] && n.carry[p] &&
                territory(n.x[p], n.y[p]) == homeChar(teamOf(p)))
                win[teamOf(p)] = true;
        }
        if (win[0] && win[1]) n.winner = 2;
        else if (win[0]) n.winner = 0;
        else if (win[1]) n.winner = 1;
        // respawns
        for (int p = 0; p < 4; ++p) {
            if (n.alive[p]) continue;
            if (--n.respawn[p] <= 0) {
                n.alive[p] = true;
                int rc = (teamOf(p) == 0) ? myRespawn : enemyRespawn;
                n.x[p] = rc % SIZE;
                n.y[p] = rc / SIZE;
                n.h[p] = 140;
                n.carry[p] = false;
            }
        }
        return n;
    }

    // Static evaluation from our team's perspective.
    double evaluate(const GState& s, int ply) const {
        if (s.winner == 0) return 1e6 - ply * 500.0;
        if (s.winner == 1) return -1e6 + ply * 500.0;
        if (s.winner == 2) return 0.0;
        double v = 0;
        for (int p = 0; p < 4; ++p) {
            int t = teamOf(p);
            double sign = (t == 0) ? 1.0 : -1.0;
            if (!s.alive[p]) {
                v -= sign * (200.0 + s.respawn[p] * 25.0);
                continue;
            }
            int ix = idx(s.x[p], s.y[p]);
            if (s.carry[p]) {
                int d = (t == 0) ? dMyTerr[ix] : dEnemyTerr[ix];
                if (d >= INF) d = 60;
                v += sign * (6000.0 - d * 60.0);
            } else {
                int d = (t == 0) ? dEnemyFlag[ix] : dMyFlag[ix];
                if (d >= INF) d = 60;
                v += sign * (400.0 - d * 7.0);
            }
            // Hydration matters only when scarce. A linear reward across the
            // whole range makes the bot hoard water — loitering at the oasis
            // instead of attacking. Penalise only a genuinely low tank.
            if (s.h[p] < 55)
                v -= sign * (55 - s.h[p]) * 5.0;
            if (catchable(s, t, s.x[p], s.y[p]))
                v -= sign * 700.0;
        }
        // Defend our flag. A shallow search can't see an enemy runner reach
        // our flag ~40 turns out, so price the threat in directly — but only
        // for an enemy that has GENUINELY invaded (stepped onto our soil, or
        // is very close to our flag), never a camper loitering in the oasis.
        if (s.fcar[0] < 0) {
            int atkBest = INF, defBest = 60;
            for (int p = 0; p < 4; ++p) {
                if (!s.alive[p]) continue;
                int dd = dMyFlag[idx(s.x[p], s.y[p])];
                if (dd >= INF) dd = 60;
                if (teamOf(p) == 0) {
                    defBest = min(defBest, dd);
                } else if (!s.carry[p] &&
                           (territory(s.x[p], s.y[p]) == myTerr || dd <= 16)) {
                    atkBest = min(atkBest, dd);
                }
            }
            if (atkBest < INF && atkBest < defBest) {
                double pen = (30 - min(atkBest, 30)) * 16.0 +
                             (defBest - atkBest) * 10.0;
                v -= pen;
            }
        }
        return v;
    }

    // Depth-limited search: we branch our move (player 0), the other three
    // follow the greedy model. The value is a DISCOUNTED SUM of per-ply
    // evaluations, not just the leaf — a pure leaf-max has no urgency (with
    // depth slack every first move can reach the same best leaf, so the bot
    // ties and oscillates). Summing the path rewards progress made NOW.
    static constexpr double GAMMA = 0.92;
    double search(const GState& s, int depth, int ply) const {
        if (++nodeCount > NODE_BUDGET) timeUp = true;
        if (timeUp) return 0.0;
        double e = evaluate(s, ply);
        if (s.winner >= 0 || depth == 0)
            return e;
        array<int, 4> mv{{4, 4, 4, 4}};
        for (int p = 1; p < 4; ++p) mv[p] = modelMove(s, p);
        double best = -1e18;
        for (int d = 0; d < 5; ++d) {
            int nx = s.x[0] + MV[d].dx, ny = s.y[0] + MV[d].dy;
            if (d != 4 && (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE ||
                           !freeCell[idx(nx, ny)]))
                continue;
            mv[0] = d;
            GState ns = step(s, mv);
            double val = search(ns, depth - 1, ply + 1);
            if (val > best) best = val;
        }
        return e + GAMMA * best;
    }

    GState reconstruct(const array<int, 16>& v) const {
        GState s;
        for (int p = 0; p < 4; ++p) {
            s.x[p] = v[p * 4];
            s.y[p] = v[p * 4 + 1];
            s.h[p] = v[p * 4 + 2];
            s.carry[p] = v[p * 4 + 3] == 1;
            s.alive[p] = s.x[p] >= 0;
            s.respawn[p] = s.alive[p] ? 0 : 15;
        }
        // flag of team 0 (ours): carried by an enemy, else at our corner.
        // flag of team 1 (enemy): carried by us, else at their corner.
        for (int t = 0; t < 2; ++t) {
            s.fcar[t] = -1;
            int corner = (t == 0) ? myCorner : enemyCorner;
            s.flagx[t] = corner % SIZE;
            s.flagy[t] = corner / SIZE;
            for (int p = 0; p < 4; ++p) {
                if (s.alive[p] && s.carry[p] && teamOf(p) != t) {
                    s.fcar[t] = p;
                    s.flagx[t] = s.x[p];
                    s.flagy[t] = s.y[p];
                    break;
                }
            }
        }
        s.winner = -1;
        return s;
    }

    char decide(const array<int, 16>& v) {
        if (v[0] < 0) return 's';
        if (!ready) setupTeam(v[1]);
        GState s = reconstruct(v);

        // Fixed node budget — deterministic per-turn cost on any machine.
        nodeCount = 0;
        timeUp = false;
        int bestDir = modelMove(s, 0);   // greedy fallback if nothing completes
        array<int, 4> mv{{4, 4, 4, 4}};
        for (int p = 1; p < 4; ++p) mv[p] = modelMove(s, p);
        int doneDepth = 0;
        array<double, 5> dbg{{-1e18, -1e18, -1e18, -1e18, -1e18}};
        for (int depth = 3; depth <= 12; ++depth) {
            double best = -1e18;
            int localBest = -1;
            array<double, 5> cur{{-1e18, -1e18, -1e18, -1e18, -1e18}};
            for (int d = 0; d < 5; ++d) {
                int nx = s.x[0] + MV[d].dx, ny = s.y[0] + MV[d].dy;
                if (d != 4 && (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE ||
                               !freeCell[idx(nx, ny)]))
                    continue;
                mv[0] = d;
                GState ns = step(s, mv);
                double val = search(ns, depth - 1, 1);
                if (timeUp) break;          // this depth is incomplete
                cur[d] = val;
                val += (rnd() & 31) * 0.01;  // tiny tie-break jitter
                if (val > best) {
                    best = val;
                    localBest = d;
                }
            }
            if (timeUp) break;              // discard partial depth, keep prev
            if (localBest >= 0) {
                bestDir = localBest;
                dbg = cur;
                doneDepth = depth;
            }
        }
#ifdef CTF_DBG
        {
            FILE* f = fopen("ctf_dbg.txt", "a");
            if (f) {
                fprintf(f, "me(%d,%d) depth=%d  u=%.0f d=%.0f l=%.0f r=%.0f "
                        "s=%.0f -> %c\n", s.x[0], s.y[0], doneDepth,
                        dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], MV[bestDir].c);
                fclose(f);
            }
        }
#endif
        (void)doneDepth;
        lastMove = MV[bestDir].c;
        return lastMove;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    Bot bot;
    string line;
    if (!getline(cin, line)) return 0;
    bot.setupBoard(line);
    while (getline(cin, line)) {
        if (line.empty()) continue;
        array<int, 16> vals{};
        istringstream iss(line);
        int n = 0;
        while (n < 16 && (iss >> vals[n])) ++n;
        char move = bot.lastMove;
        if (n >= 16) {
            try {
                move = bot.decide(vals);
            } catch (...) {
                move = bot.lastMove;
            }
        }
        cout << move << '\n' << flush;
    }
    return 0;
}
