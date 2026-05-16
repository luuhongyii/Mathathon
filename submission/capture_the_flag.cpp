// Capture the Flag - Mathathon C++ submission.
//
// 29x29 grid, two teams of two. Read the board on the first line, then each
// round read 16 ints: me, teammate, enemy0, enemy1 as x y hydration carrying.
// Print exactly one of u/d/l/r/s every round.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace std;

static constexpr int SIZE = 29;
static constexpr int N = SIZE * SIZE;
static constexpr int INF = 1 << 28;

struct Dir {
    char c;
    int dx;
    int dy;
};

static constexpr array<Dir, 4> DIRS{{
    {'u', 0, -1}, {'d', 0, 1}, {'l', -1, 0}, {'r', 1, 0}
}};

struct Player {
    int x = -1, y = -1, h = 0, f = 0;
};

static bool inOasis(int x, int y) {
    return 12 <= x && x <= 16 && 12 <= y && y <= 16;
}

static char territory(int x, int y) {
    if (inOasis(x, y)) return 'N';
    if (y <= 13) return 'B';
    if (y >= 15) return 'R';
    return 'N';
}

static int idx(int x, int y) {
    return y * SIZE + x;
}

struct Bot {
    bool boardReady = false;
    bool ready = false;
    array<unsigned char, N> freeCell{};
    array<vector<int>, N> adj{};

    bool blue = true;
    int myFlag = 0;
    int enemyFlag = 0;
    char myTerr = 'B';
    char enemyTerr = 'R';

    vector<int> dEnemyFlag, dMyTerr, dOasis, dMyFlag, dGuard;

    char committedRole = 0;
    bool refilling = false;
    int stall = 0;
    string lastTargetId;
    int lastTargetVal = INF;
    char lastMove = 's';
    uint32_t rng = 2463534242u;

    Bot() {
        auto now = chrono::high_resolution_clock::now().time_since_epoch().count();
        uintptr_t addr = reinterpret_cast<uintptr_t>(this);
        rng ^= (uint32_t)now;
        rng ^= (uint32_t)(now >> 32);
        rng ^= (uint32_t)addr;
#ifdef _WIN32
        rng ^= (uint32_t)_getpid() * 2654435761u;
#else
        rng ^= (uint32_t)getpid() * 2654435761u;
#endif
    }

    uint32_t rnd() {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    }

    void setupBoard(const string& board) {
        for (int i = 0; i < N; ++i) {
            freeCell[i] = (i < (int)board.size() && board[i] == '#') ? 0 : 1;
        }
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int here = idx(x, y);
                adj[here].clear();
                if (!freeCell[here]) continue;
                for (const Dir& d : DIRS) {
                    int nx = x + d.dx, ny = y + d.dy;
                    if (0 <= nx && nx < SIZE && 0 <= ny && ny < SIZE &&
                        freeCell[idx(nx, ny)]) {
                        adj[here].push_back(idx(nx, ny));
                    }
                }
            }
        }
        boardReady = true;
    }

    vector<int> bfs(const vector<int>& sources) const {
        vector<int> dist(N, INF);
        queue<int> q;
        for (int s : sources) {
            if (0 <= s && s < N && freeCell[s] && dist[s] == INF) {
                dist[s] = 0;
                q.push(s);
            }
        }
        while (!q.empty()) {
            int c = q.front();
            q.pop();
            int nd = dist[c] + 1;
            for (int nb : adj[c]) {
                if (dist[nb] == INF) {
                    dist[nb] = nd;
                    q.push(nb);
                }
            }
        }
        return dist;
    }

    void setupTeam(int my) {
        blue = (my <= 14);
        if (blue) {
            myFlag = idx(0, 0);
            enemyFlag = idx(28, 28);
            myTerr = 'B';
            enemyTerr = 'R';
        } else {
            myFlag = idx(28, 28);
            enemyFlag = idx(0, 0);
            myTerr = 'R';
            enemyTerr = 'B';
        }

        dEnemyFlag = bfs({enemyFlag});
        dMyFlag = bfs({myFlag});

        vector<int> oasis, myTerrCells, guard;
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int p = idx(x, y);
                if (!freeCell[p]) continue;
                if (inOasis(x, y)) oasis.push_back(p);
                if (territory(x, y) == myTerr) myTerrCells.push_back(p);

                // Idle defender waits near our front line, still in home
                // territory, so intruders are caught soon after crossing.
                if (myTerr == 'B') {
                    if ((y == 12 || y == 13) && 2 <= x && x <= 26) guard.push_back(p);
                } else {
                    if ((y == 15 || y == 16) && 2 <= x && x <= 26) guard.push_back(p);
                }
            }
        }
        if (guard.empty()) guard.push_back(myFlag);
        dOasis = bfs(oasis);
        dMyTerr = bfs(myTerrCells);
        dGuard = bfs(guard);
        ready = true;
    }

    bool danger(int x, int y, const vector<Player>& enemies) const {
        if (territory(x, y) != enemyTerr) return false;
        for (const Player& e : enemies) {
            if (max(abs(e.x - x), abs(e.y - y)) <= 1) return true;
        }
        return false;
    }

    static int minCheb(int x, int y, const vector<Player>& enemies) {
        int best = 99;
        for (const Player& e : enemies) {
            best = min(best, max(abs(e.x - x), abs(e.y - y)));
        }
        return best;
    }

    char decide(const array<int, 16>& v) {
        array<Player, 4> p;
        for (int i = 0; i < 4; ++i) {
            p[i] = {v[i * 4], v[i * 4 + 1], v[i * 4 + 2], v[i * 4 + 3]};
        }
        Player me = p[0];
        if (me.x < 0) {
            stall = 0;
            refilling = false;
            return 's';
        }
        if (!ready) setupTeam(me.y);

        Player mate = p[1];
        bool mateAlive = mate.x >= 0;
        vector<Player> enemies;
        if (p[2].x >= 0) enemies.push_back(p[2]);
        if (p[3].x >= 0) enemies.push_back(p[3]);

        int myIdx = idx(me.x, me.y);
        bool iCarry = me.f == 1;
        bool mateCarry = mateAlive && mate.f == 1;
        int carrierIndex = -1;
        for (int i = 0; i < (int)enemies.size(); ++i) {
            if (enemies[i].f == 1) {
                carrierIndex = i;
                break;
            }
        }

        vector<int> chaseField;
        bool hasChaseField = false;
        bool provisional = false;
        char role;
        if (iCarry) {
            role = 'A';
        } else if (carrierIndex >= 0) {
            const Player& c = enemies[carrierIndex];
            chaseField = bfs({idx(c.x, c.y)});
            hasChaseField = true;
            int myD = chaseField[myIdx];
            int mateD = mateAlive ? chaseField[idx(mate.x, mate.y)] : INF;
            bool mineCloser = myD < mateD ||
                (myD == mateD && pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y));
            role = (!mateAlive || mineCloser) ? 'D' : 'A';
        } else if (mateCarry) {
            role = 'D';
        } else if (!mateAlive) {
            role = 'A';
        } else if (committedRole) {
            role = committedRole;
        } else if (me.x != mate.x || me.y != mate.y) {
            role = pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y) ? 'A' : 'D';
            committedRole = role;
        } else {
            role = 'A';
            provisional = true;
        }

        int doasis = dOasis[myIdx];
        if (inOasis(me.x, me.y)) {
            refilling = false;
        } else if (doasis != INF && me.h < 2 * doasis + 14) {
            refilling = true;
        }

        const vector<int>* target = nullptr;
        string targetId;
        int goalX = 14, goalY = 14;

        if (refilling && !inOasis(me.x, me.y)) {
            target = &dOasis;
            targetId = "oasis";
            goalX = goalY = 14;
        } else if (role == 'A') {
            if (iCarry) {
                target = &dMyTerr;
                targetId = "home";
                goalX = myFlag % SIZE;
                goalY = myFlag / SIZE;
            } else {
                target = &dEnemyFlag;
                targetId = "eflag";
                goalX = enemyFlag % SIZE;
                goalY = enemyFlag / SIZE;
            }
        } else {
            int intruderIndex = carrierIndex;
            if (intruderIndex < 0) {
                int best = INF;
                for (int i = 0; i < (int)enemies.size(); ++i) {
                    const Player& e = enemies[i];
                    if (territory(e.x, e.y) == myTerr) {
                        int ds = dMyFlag[idx(e.x, e.y)];
                        if (ds < best) {
                            best = ds;
                            intruderIndex = i;
                        }
                    }
                }
            }
            if (intruderIndex >= 0) {
                const Player& intr = enemies[intruderIndex];
                if (hasChaseField && intruderIndex == carrierIndex) {
                    target = &chaseField;
                } else {
                    chaseField = bfs({idx(intr.x, intr.y)});
                    hasChaseField = true;
                    target = &chaseField;
                }
                targetId = "chase";
                goalX = intr.x;
                goalY = intr.y;
            } else {
                target = &dGuard;
                targetId = "guard";
                goalX = blue ? 14 : 14;
                goalY = blue ? 13 : 15;
            }
        }

        int cur = (*target)[myIdx];
        if (targetId != lastTargetId || cur < lastTargetVal || cur == 0) {
            stall = 0;
        } else {
            ++stall;
        }
        lastTargetId = targetId;
        lastTargetVal = cur;
        bool desperate = stall > 25;

        struct Cand { char c; int x; int y; };
        vector<Cand> cands;
        for (const Dir& d : DIRS) {
            int nx = me.x + d.dx, ny = me.y + d.dy;
            if (0 <= nx && nx < SIZE && 0 <= ny && ny < SIZE && freeCell[idx(nx, ny)]) {
                cands.push_back({d.c, nx, ny});
            }
        }
        cands.push_back({'s', me.x, me.y});

        bool useMan = cur == INF;
        if (provisional) {
            vector<char> forward;
            vector<char> legal;
            for (const Cand& c : cands) {
                legal.push_back(c.c);
                int dval = useMan ? abs(c.x - goalX) + abs(c.y - goalY) : (*target)[idx(c.x, c.y)];
                if (dval <= cur) forward.push_back(c.c);
            }
            const vector<char>& opts = forward.empty() ? legal : forward;
            lastMove = opts[rnd() % opts.size()];
            return lastMove;
        }

        bool hereDanger = danger(me.x, me.y, enemies);
        char bestMove = 's';
        array<int, 4> bestKey{{-INF, -INF, -INF, -INF}};
        for (const Cand& c : cands) {
            int ni = idx(c.x, c.y);
            int dval;
            if (useMan) {
                dval = abs(c.x - goalX) + abs(c.y - goalY);
            } else {
                int dv = (*target)[ni];
                dval = (dv != INF) ? dv : 9000 + abs(c.x - goalX) + abs(c.y - goalY);
            }
            bool badDanger = danger(c.x, c.y, enemies) && !desperate;
            bool terrPenalty = role == 'D' && territory(c.x, c.y) == enemyTerr;
            int safe = (badDanger || terrPenalty) ? 0 : 1;
            int jitter = (int)(rnd() & 1023);
            array<int, 4> key;
            if (hereDanger) {
                key = {{safe, minCheb(c.x, c.y, enemies), -dval, jitter}};
            } else {
                key = {{safe, -dval, jitter, 0}};
            }
            if (key > bestKey) {
                bestKey = key;
                bestMove = c.c;
            }
        }
        lastMove = bestMove;
        return bestMove;
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
