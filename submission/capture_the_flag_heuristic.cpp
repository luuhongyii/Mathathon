// Capture the Flag - Mathathon C++ submission (balanced build).
// Submit this source via Submit C++/Binary; Linux g++ compiles it on the platform.
// First line: board. Each round: 16 ints (me, mate, e0, e1) as x y h carrying.
// Print exactly one of u/d/l/r/s per round. Never write to stderr.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
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
    bool ready = false;
    array<unsigned char, N> freeCell{};
    array<vector<int>, N> adj{};

    bool blue = true;
    int myFlag = 0;
    int enemyFlag = 0;
    char myTerr = 'B';
    char enemyTerr = 'R';

    vector<int> dEnemyFlag, dMyTerr, dEnemyTerr, dOasis, dMyFlag, dGuard, dMidGuard;
    vector<int> myTerrCellsList;
    vector<int> chaseScratch;

    bool refilling = false;
    int stall = 0;
    int stageRounds = 0;
    string lastTargetId;
    int lastTargetVal = INF;
    char lastMove = 's';
    uint32_t rng = 2463534242u;

    Bot() {
        uintptr_t addr = reinterpret_cast<uintptr_t>(this);
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
    }

    vector<int> bfs(const vector<int>& sources) {
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

        vector<int> oasis, myTerrCells, enemyTerrCells, guard, midGuard;
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int p = idx(x, y);
                if (!freeCell[p]) continue;
                if (inOasis(x, y)) oasis.push_back(p);
                if (territory(x, y) == myTerr) myTerrCells.push_back(p);
                if (territory(x, y) == enemyTerr) enemyTerrCells.push_back(p);
                if (myTerr == 'B') {
                    if ((y == 12 || y == 13) && 2 <= x && x <= 26) guard.push_back(p);
                    if ((y == 10 || y == 11 || y == 12) && 4 <= x && x <= 24) midGuard.push_back(p);
                } else {
                    if ((y == 15 || y == 16) && 2 <= x && x <= 26) guard.push_back(p);
                    if ((y == 16 || y == 17 || y == 18) && 4 <= x && x <= 24) midGuard.push_back(p);
                }
            }
        }
        if (guard.empty()) guard.push_back(myFlag);
        if (midGuard.empty()) midGuard = guard;
        dOasis = bfs(oasis);
        dMyTerr = bfs(myTerrCells);
        dEnemyTerr = bfs(enemyTerrCells);
        myTerrCellsList = myTerrCells;
        dGuard = bfs(guard);
        dMidGuard = bfs(midGuard);
        ready = true;
    }

    bool danger(int x, int y, const vector<Player>& enemies) const {
        if (territory(x, y) != enemyTerr) return false;
        for (const Player& e : enemies) {
            if (max(abs(e.x - x), abs(e.y - y)) <= 1) return true;
        }
        return false;
    }

    bool dangerNext(int x, int y, const vector<Player>& enemies) const {
        if (territory(x, y) != enemyTerr) return false;
        if (danger(x, y, enemies)) return true;
        for (const Player& e : enemies) {
            if (max(abs(e.x - x), abs(e.y - y)) > 2) continue;
            for (const Dir& d : DIRS) {
                int nx = e.x + d.dx, ny = e.y + d.dy;
                if (0 <= nx && nx < SIZE && 0 <= ny && ny < SIZE &&
                    freeCell[idx(nx, ny)] &&
                    max(abs(nx - x), abs(ny - y)) <= 1) {
                    return true;
                }
            }
        }
        return false;
    }

    // An enemy effectively can't contest our flag if it is about to die:
    // very low hydration, or standing in OUR territory adjacent to our mate
    // (our mate kills it next turn).
    bool enemyEffectivelyDead(const Player& e, const Player& mate,
                              bool mateAlive) const {
        if (e.h > 0 && e.h <= 3) return true;
        if (mateAlive && territory(e.x, e.y) == myTerr) {
            if (max(abs(mate.x - e.x), abs(mate.y - e.y)) <= 1) return true;
        }
        return false;
    }

    bool flagGuarded(int myIdx, const vector<Player>& enemies,
                     const Player& mate, bool mateAlive) const {
        if (enemies.empty()) return false;
        int myDist = dEnemyFlag[myIdx];
        int enemyBest = INF;
        for (const Player& e : enemies) {
            if (enemyEffectivelyDead(e, mate, mateAlive)) continue;
            // Only an enemy actually inside its own flag-side territory can
            // guard the flag. An enemy parked in the neutral oasis, or roaming
            // OUR territory, is not a guard: it cannot catch us on neutral
            // ground, and once we commit it is behind us (same-speed pursuit
            // from behind never catches). Counting such a camper made us stage
            // at the oasis forever and score zero.
            if (territory(e.x, e.y) != enemyTerr) continue;
            // A low-hydration enemy can't sustain a long contest; treat it as
            // farther away by the deficit it cannot afford.
            int ed = dEnemyFlag[idx(e.x, e.y)];
            if (e.h > 0 && e.h < 30) ed += (30 - e.h) / 4;
            enemyBest = min(enemyBest, ed);
        }
        if (enemyBest == INF) return false;
        return enemyBest <= myDist;
    }

    // Multi-source BFS that avoids cells where we would be caught (enemy
    // territory, Chebyshev-1 of a living enemy). Pure shortest-path descent
    // dead-ends head-on against a defender guarding the only direct cells;
    // this field routes AROUND it instead. `keep` is a cell that must stay
    // traversable even if enemy-adjacent (the goal we still must reach); pass
    // -1 for none.
    vector<int> safeField(const vector<int>& sources,
                          const vector<Player>& enemies, int keep) {
        vector<unsigned char> blocked(N, 0);
        for (const Player& e : enemies) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int x = e.x + dx, y = e.y + dy;
                    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) continue;
                    int c = idx(x, y);
                    if (c == keep) continue;
                    if (territory(x, y) != enemyTerr) continue;
                    blocked[c] = 1;
                }
            }
        }
        vector<int> dist(N, INF);
        queue<int> q;
        for (int s : sources) {
            if (0 <= s && s < N && freeCell[s] && !blocked[s] &&
                dist[s] == INF) {
                dist[s] = 0;
                q.push(s);
            }
        }
        while (!q.empty()) {
            int c = q.front();
            q.pop();
            for (int nb : adj[c]) {
                if (dist[nb] == INF && !blocked[nb]) {
                    dist[nb] = dist[c] + 1;
                    q.push(nb);
                }
            }
        }
        return dist;
    }

    // Attacker: route to the enemy flag around any guards.
    vector<int> attackField(const vector<Player>& enemies) {
        return safeField({enemyFlag}, enemies, enemyFlag);
    }

    // Carrier: route home around interceptors. Losing the flag mid-route to a
    // defender camped on the shortest path is catastrophic, so the carry path
    // gets the same danger-avoiding treatment as the attack path.
    vector<int> homeField(const vector<Player>& enemies) {
        return safeField(myTerrCellsList, enemies, -1);
    }

    // Predictive interception: find where a defender can cut an intruder off
    // instead of chasing from behind (equal-speed pursuit from behind never
    // catches). Picks the cell on the intruder's shortest path to its goal
    // that we reach no later than it, as early on that path as possible.
    // Restricted to OUR territory: a kill only happens with the intruder on
    // our soil, and an oasis/enemy-soil intercept just deadlocks (memory note).
    // Returns the intercept cell index, or -1 if none is reachable in time.
    int interceptCell(const Player& intr, const vector<int>& goalField,
                      const vector<int>& myField) {
        vector<int> intrField = bfs({idx(intr.x, intr.y)});
        int goalDist = goalField[idx(intr.x, intr.y)];
        int bestCell = -1;
        int bestRemaining = -1;
        for (int i = 0; i < N; ++i) {
            if (!freeCell[i]) continue;
            if (territory(i % SIZE, i / SIZE) != myTerr) continue;
            int enemyHere = intrField[i];
            int enemyRest = goalField[i];
            if (enemyHere == INF || enemyRest == INF) continue;
            if (goalDist != INF && enemyHere + enemyRest > goalDist + 1) continue;
            int mineHere = myField[i];
            if (mineHere == INF || mineHere > enemyHere) continue;
            if (enemyRest > bestRemaining) {
                bestRemaining = enemyRest;
                bestCell = i;
            }
        }
        return bestCell;
    }

    vector<int> escortField(const Player& mate) {
        int mateI = idx(mate.x, mate.y);
        int dm = dMyFlag[mateI];
        vector<int> sources;
        for (int i = 0; i < N; ++i) {
            if (!freeCell[i]) continue;
            int d = dMyFlag[i];
            if (d == INF || d > dm) continue;
            int x = i % SIZE, y = i / SIZE;
            char t = territory(x, y);
            if (t != myTerr && t != 'N') continue;
            if (dm - d <= 10) sources.push_back(i);
        }
        if (sources.empty()) sources.push_back(mateI);
        return bfs(sources);
    }

    void updateRefill(int mx, int my, int mh, int myIdx, bool iCarry,
                      const Player& mate, bool mateAlive) {
        if (mateAlive && inOasis(mate.x, mate.y) && mate.h >= 105 && mh >= 72) {
            refilling = false;
            return;
        }
        int doasis = dOasis[myIdx];
        if (inOasis(mx, my)) {
            refilling = mh < (iCarry ? 95 : 108);
            return;
        }
        if (doasis == INF) {
            refilling = false;
            return;
        }
        if (iCarry) {
            refilling = mh < dMyTerr[myIdx] + 22;
        } else {
            refilling = mh < 2 * doasis + 12;
        }
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
        vector<int> attField;
        bool provisional = false;
        char role;
        if (iCarry) {
            role = 'A';
        } else if (carrierIndex >= 0) {
            // Our flag is being carried away — chase the carrier. Use raw
            // distance (whoever is closer); break exact ties by position only.
            const Player& c = enemies[carrierIndex];
            chaseField = bfs({idx(c.x, c.y)});
            int myD = chaseField[myIdx];
            int mateD = mateAlive ? chaseField[idx(mate.x, mate.y)] : INF;
            bool mineCloser = myD < mateD ||
                (myD == mateD && pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y));
            role = (!mateAlive || mineCloser) ? 'D' : 'A';
        } else if (mateCarry) {
            role = 'D';
        } else if (!mateAlive) {
            role = 'A';
        } else {
            // No flag carried. The teammate is an INDEPENDENT bot — never
            // assume it will take a role for us. Decide purely from board
            // state: attack by default; defend only when our flag is
            // genuinely threatened AND we are the better-placed defender.
            // If the mate camps the flag it is the closer one, so we attack;
            // if the mate roams off, we cover. Needs no cooperation from it.
            int threat = INF;
            for (const Player& e : enemies) {
                threat = min(threat, dMyFlag[idx(e.x, e.y)]);
            }
            int myFlagDist = dMyFlag[myIdx];
            int mateFlagDist = dMyFlag[idx(mate.x, mate.y)];
            bool iAmCloserToFlag = myFlagDist < mateFlagDist ||
                (myFlagDist == mateFlagDist &&
                 pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y));
            // Re-derived every round (never a sticky role): if the mate is
            // observably the committed attacker — much nearer the enemy flag,
            // or basically on it — then offense is covered, so on any real
            // threat to our flag we drop back to defend earlier than the bare
            // threat<=10 reaction. With no threat, both still attack.
            int myEf = dEnemyFlag[myIdx];
            int mateEf = dEnemyFlag[idx(mate.x, mate.y)];
            bool matePrimaryAttack = mateEf != INF &&
                (mateEf + 8 < myEf || mateEf <= 12);
            bool defend = (threat <= 10 && iAmCloserToFlag) ||
                          (threat <= 16 && matePrimaryAttack);
            role = defend ? 'D' : 'A';
            if (me.x == mate.x && me.y == mate.y) provisional = true;
        }

        updateRefill(me.x, me.y, me.h, myIdx, iCarry, mate, mateAlive);

        bool flagGuardedNow = role == 'A' && !iCarry &&
                              flagGuarded(myIdx, enemies, mate, mateAlive);
        // Staging is a brief wait for an opening, never a permanent state.
        // Against a passive defender no opening ever comes — after a short
        // wait, commit and attack instead of staging until the game ends.
        if (flagGuardedNow) {
            if (++stageRounds > 12) flagGuardedNow = false;
        } else {
            stageRounds = 0;
        }

        const vector<int>* target = nullptr;
        string targetId;
        int goalX = 14, goalY = 14;

        if (refilling && !inOasis(me.x, me.y)) {
            target = &dOasis;
            targetId = "oasis";
            goalX = goalY = 14;
        } else if (flagGuardedNow) {
            target = &dOasis;
            targetId = "stage";
            goalX = goalY = 14;
        } else if (role == 'A') {
            if (iCarry) {
                attField = homeField(enemies);
                target = &attField;
                targetId = "home";
                goalX = myFlag % SIZE;
                goalY = myFlag / SIZE;
            } else {
                attField = attackField(enemies);
                target = &attField;
                targetId = "eflag";
                goalX = enemyFlag % SIZE;
                goalY = enemyFlag / SIZE;
            }
        } else if (mateCarry) {
            chaseScratch = escortField(mate);
            target = &chaseScratch;
            targetId = "escort";
            goalX = myFlag % SIZE;
            goalY = myFlag / SIZE;
            if (!enemies.empty()) {
                int best = INF;
                int blocker = -1;
                for (int i = 0; i < (int)enemies.size(); ++i) {
                    int ds = dMyFlag[idx(enemies[i].x, enemies[i].y)];
                    if (ds < best) {
                        best = ds;
                        blocker = i;
                    }
                }
                if (blocker >= 0) {
                    chaseField = bfs({idx(enemies[blocker].x, enemies[blocker].y)});
                    target = &chaseField;
                    targetId = "block";
                    goalX = enemies[blocker].x;
                    goalY = enemies[blocker].y;
                }
            }
        } else {
            // Defender. Chase the flag-carrier anywhere; otherwise only chase
            // enemies that are actually INSIDE our territory. Chasing an enemy
            // onto neutral/oasis ground cannot kill it and deadlocks both bots
            // on a single cell — for those, hold the guard line instead.
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
                // Intercept ahead of the intruder, not behind it. A loose
                // intruder heads for our flag (goal field dMyFlag); a flag
                // carrier flees toward its own territory (goal field
                // dEnemyTerr) and must still cross our soil, where we can cut
                // it off before it reaches the neutral oasis.
                vector<int> myField = bfs({myIdx});
                const vector<int>& goalField =
                    (intruderIndex == carrierIndex) ? dEnemyTerr : dMyFlag;
                int cell = interceptCell(intr, goalField, myField);
                if (cell >= 0) {
                    chaseField = bfs({cell});
                    target = &chaseField;
                    targetId = "intercept";
                    goalX = cell % SIZE;
                    goalY = cell / SIZE;
                } else {
                    chaseField = bfs({idx(intr.x, intr.y)});
                    target = &chaseField;
                    targetId = "chase";
                    goalX = intr.x;
                    goalY = intr.y;
                }
            } else {
                target = enemies.empty() ? &dGuard : &dMidGuard;
                targetId = enemies.empty() ? "guard" : "midguard";
                goalX = 14;
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
        int stallLim = iCarry ? 16 : 25;
        bool desperate = stall > stallLim && !flagGuardedNow;

        struct Cand {
            char c;
            int x;
            int y;
        };
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
            vector<char> forward, lateral, legal;
            for (const Cand& c : cands) {
                legal.push_back(c.c);
                if (c.c == 's') continue;
                int dval = useMan ? abs(c.x - goalX) + abs(c.y - goalY)
                                  : (*target)[idx(c.x, c.y)];
                if (dval <= cur) {
                    forward.push_back(c.c);
                    if (c.x != me.x) lateral.push_back(c.c);
                }
            }
            const vector<char>* opts = &legal;
            if (!lateral.empty()) opts = &lateral;
            else if (!forward.empty()) opts = &forward;
            lastMove = (*opts)[rnd() % opts->size()];
            return lastMove;
        }

        bool hereDanger = dangerNext(me.x, me.y, enemies);
        char bestMove = 's';
        array<int, 4> bestKey{{-INF, -INF, -INF, -INF}};
        for (const Cand& c : cands) {
            int dval;
            if (useMan) {
                dval = abs(c.x - goalX) + abs(c.y - goalY);
            } else {
                int dv = (*target)[idx(c.x, c.y)];
                dval = (dv != INF) ? dv : 9000 + abs(c.x - goalX) + abs(c.y - goalY);
            }
            // A stalled bot relaxes lookahead danger to break free, but a
            // stalled CARRIER must still never step into a one-step catch
            // (losing the flag mid-route is catastrophic).
            bool badDanger;
            if (desperate) {
                badDanger = iCarry && danger(c.x, c.y, enemies);
            } else {
                badDanger = dangerNext(c.x, c.y, enemies);
            }
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
