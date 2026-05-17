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

    vector<int> dEnemyFlag, dEnemyTerr, dMyTerr, dHomeGate, dEnemyGate;
    vector<int> dOasis, dMyFlag, dGuard, dMidGuard, dFlagRing;
    vector<int> chaseScratch;
    vector<int> interceptScratch;

    char committedRole = 0;
    bool refilling = false;
    int oasisToHome = 0;
    int stall = 0;
    int guardedWait = 0;
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

        int fx = myFlag % SIZE, fy = myFlag / SIZE;
        vector<int> oasis, myTerrCells, enemyTerrCells, homeGate, enemyGate;
        vector<int> guard, midGuard, ring;
        for (int y = 0; y < SIZE; ++y) {
            for (int x = 0; x < SIZE; ++x) {
                int p = idx(x, y);
                if (!freeCell[p]) continue;
                if (inOasis(x, y)) oasis.push_back(p);
                if (territory(x, y) == myTerr) myTerrCells.push_back(p);
                if (territory(x, y) == enemyTerr) enemyTerrCells.push_back(p);
                if (myTerr == 'B') {
                    if (y == 12 || y == 13) homeGate.push_back(p);
                    if (y == 15 || y == 16) enemyGate.push_back(p);
                } else {
                    if (y == 15 || y == 16) homeGate.push_back(p);
                    if (y == 12 || y == 13) enemyGate.push_back(p);
                }
                int cheb = max(abs(x - fx), abs(y - fy));
                if (1 <= cheb && cheb <= 2 && territory(x, y) == myTerr) {
                    ring.push_back(p);
                }
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
        dHomeGate = bfs(homeGate.empty() ? vector<int>{myFlag} : homeGate);
        dEnemyGate = bfs(enemyGate.empty() ? vector<int>{enemyFlag} : enemyGate);
        dGuard = bfs(guard);
        dMidGuard = bfs(midGuard);
        if (ring.empty()) ring.push_back(myFlag);
        dFlagRing = bfs(ring);
        oasisToHome = 0;
        for (int p : oasis) {
            int d = dMyTerr[p];
            if (d != INF && d > oasisToHome) oasisToHome = d;
        }
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
            // A low-hydration enemy can't sustain a long contest; treat it as
            // farther away by the deficit it cannot afford.
            int ed = dEnemyFlag[idx(e.x, e.y)];
            if (e.h > 0 && e.h < 30) ed += (30 - e.h) / 4;
            enemyBest = min(enemyBest, ed);
        }
        if (enemyBest == INF) return false;
        return enemyBest <= myDist;
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

    pair<vector<int>, pair<int, int>> interceptField(const Player& intr,
                                                     const vector<int>& goalField,
                                                     const vector<int>& myField) {
        int intrI = idx(intr.x, intr.y);
        vector<int> intrField = bfs({intrI});
        int goalDist = goalField[intrI];
        int bestCell = -1;
        int bestRemaining = -1;
        for (int i = 0; i < N; ++i) {
            if (!freeCell[i]) continue;
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
        if (bestCell < 0) {
            return {std::move(intrField), {intr.x, intr.y}};
        }
        return {bfs({bestCell}), {bestCell % SIZE, bestCell / SIZE}};
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
            int homeDirect = dMyTerr[myIdx];
            int viaOasis = doasis + oasisToHome;
            bool canGoHome = mh >= homeDirect + 24;
            bool canReachOasis = mh >= doasis + 4;
            bool oasisHelps = viaOasis <= homeDirect + 16;
            refilling = !canGoHome && canReachOasis && oasisHelps;
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
        bool matePrimaryAttack = false;
        if (mateAlive && !iCarry && !mateCarry) {
            int myEf = dEnemyFlag[myIdx];
            int mateEf = dEnemyFlag[idx(mate.x, mate.y)];
            matePrimaryAttack = mateEf != INF && (mateEf + 8 < myEf || mateEf <= 12);
        }
        bool enemyNearHome = false;
        for (const Player& e : enemies) {
            if (dMyFlag[idx(e.x, e.y)] <= 14) {
                enemyNearHome = true;
                break;
            }
        }
        int carrierIndex = -1;
        for (int i = 0; i < (int)enemies.size(); ++i) {
            if (enemies[i].f == 1) {
                carrierIndex = i;
                break;
            }
        }

        vector<int> chaseField;
        bool provisional = false;
        char role;
        if (iCarry) {
            role = 'A';
        } else if (mateCarry) {
            role = 'D';
        } else if (carrierIndex >= 0) {
            const Player& c = enemies[carrierIndex];
            chaseField = bfs({idx(c.x, c.y)});
            int myD = chaseField[myIdx];
            int mateD = mateAlive ? chaseField[idx(mate.x, mate.y)] : INF;
            bool mineCloser = myD < mateD ||
                (myD == mateD && pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y));
            role = (!mateAlive || mineCloser) ? 'D' : 'A';
        } else if (matePrimaryAttack && carrierIndex < 0) {
            role = 'D';
        } else if (!mateAlive) {
            role = enemyNearHome ? 'D' : 'A';
        } else if (committedRole) {
            role = committedRole;
        } else if (me.x != mate.x || me.y != mate.y) {
            role = pair<int, int>(me.x, me.y) < pair<int, int>(mate.x, mate.y) ? 'A' : 'D';
            committedRole = role;
        } else {
            role = 'A';
            provisional = true;
        }

        updateRefill(me.x, me.y, me.h, myIdx, iCarry, mate, mateAlive);
        bool carrierEmergency = carrierIndex >= 0 && !iCarry && !mateCarry;
        bool continueSteal = carrierEmergency && dEnemyFlag[myIdx] <= 6 && me.h >= 44;

        bool flagGuardedNow = role == 'A' && !iCarry &&
                              flagGuarded(myIdx, enemies, mate, mateAlive);
        if (flagGuardedNow) {
            ++guardedWait;
        } else {
            guardedWait = 0;
        }
        bool stageGuarded = flagGuardedNow && guardedWait <= 8;

        const vector<int>* target = nullptr;
        string targetId;
        int goalX = 14, goalY = 14;

        if (refilling && !inOasis(me.x, me.y)) {
            target = &dOasis;
            targetId = "oasis";
            goalX = goalY = 14;
        } else if (stageGuarded) {
            target = &dOasis;
            targetId = "stage";
            goalX = goalY = 14;
        } else if (role == 'A' && carrierEmergency && !continueSteal) {
            if (inOasis(me.x, me.y) || me.h >= 80) {
                target = &dMidGuard;
                targetId = "resetguard";
                goalX = 14;
                goalY = blue ? 13 : 15;
            } else {
                target = &dOasis;
                targetId = "reset";
                goalX = goalY = 14;
            }
        } else if (role == 'A') {
            if (iCarry) {
                bool gateBlocked = dHomeGate[myIdx] <= 8;
                if (gateBlocked) {
                    gateBlocked = false;
                    for (const Player& e : enemies) {
                        if (dHomeGate[idx(e.x, e.y)] <= 3) {
                            gateBlocked = true;
                            break;
                        }
                    }
                }
                bool exitTrap = territory(me.x, me.y) == enemyTerr &&
                                !inOasis(me.x, me.y) &&
                                dHomeGate[myIdx] > 5;
                if (exitTrap) {
                    exitTrap = false;
                    for (const Player& e : enemies) {
                        if (e.f == 0 &&
                            max(abs(e.x - me.x), abs(e.y - me.y)) <= 6) {
                            exitTrap = true;
                            break;
                        }
                    }
                }
                if ((gateBlocked || exitTrap) && !inOasis(me.x, me.y)) {
                    target = &dOasis;
                    targetId = "reroute";
                    goalX = goalY = 14;
                } else {
                    target = &dMyTerr;
                    targetId = "home";
                    goalX = myFlag % SIZE;
                    goalY = myFlag / SIZE;
                }
            } else {
                target = &dEnemyFlag;
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
                vector<int> mateField = bfs({idx(mate.x, mate.y)});
                int mateHomeDist = dMyTerr[idx(mate.x, mate.y)];
                int best = INF;
                int blocker = -1;
                int mateGateDist = dHomeGate[idx(mate.x, mate.y)];
                for (int i = 0; i < (int)enemies.size(); ++i) {
                    int eidx = idx(enemies[i].x, enemies[i].y);
                    int ds = mateField[eidx];
                    if (ds == INF) continue;
                    if (territory(mate.x, mate.y) == enemyTerr &&
                        enemies[i].f == 0 && ds <= 8) {
                        ds -= 14;
                    }
                    if (mateGateDist <= 10) {
                        if (dHomeGate[eidx] <= 4) {
                            ds -= 10;
                        } else {
                            ds += 6;
                        }
                    }
                    if (enemies[i].f == 1) {
                        vector<int> enemyField = bfs({eidx});
                        int catchDist = enemyField[myIdx];
                        if (catchDist <= dEnemyTerr[eidx] + 3) {
                            ds -= 6;
                        } else {
                            ds += 18;
                        }
                    }
                    if (ds < best) {
                        best = ds;
                        blocker = i;
                    }
                }
                if (blocker >= 0 && best <= mateHomeDist + 20) {
                    chaseField = bfs({idx(enemies[blocker].x, enemies[blocker].y)});
                    target = &chaseField;
                    targetId = "block";
                    goalX = enemies[blocker].x;
                    goalY = enemies[blocker].y;
                }
            }
        } else {
            int enemyNearFlag = INF;
            for (const Player& e : enemies) {
                enemyNearFlag = min(enemyNearFlag, dMyFlag[idx(e.x, e.y)]);
            }

            bool anchorDefense = matePrimaryAttack && carrierIndex < 0 &&
                                 enemyNearFlag > 9;
            int intruderIndex = carrierIndex;
            if (anchorDefense) {
                if (inOasis(me.x, me.y) && me.h >= 120) {
                    target = &dMidGuard;
                    targetId = "anchor";
                    goalX = 14;
                    goalY = blue ? 13 : 15;
                } else {
                    target = &dOasis;
                    targetId = "anchorfill";
                    goalX = goalY = 14;
                }
            } else if (intruderIndex < 0) {
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
                if (intruderIndex < 0) {
                    for (int i = 0; i < (int)enemies.size(); ++i) {
                        const Player& e = enemies[i];
                        int pidx = idx(e.x, e.y);
                        bool approaching = myTerr == 'B'
                            ? (e.y >= 11 && e.y <= 16)
                            : (e.y >= 12 && e.y <= 17);
                        if (approaching && dMyFlag[pidx] < best) {
                            best = dMyFlag[pidx];
                            intruderIndex = i;
                        }
                    }
                }
            }

            if (target == nullptr && carrierIndex < 0 && enemyNearFlag <= 9) {
                target = &dFlagRing;
                targetId = "campflag";
                goalX = myFlag % SIZE;
                goalY = myFlag / SIZE;
            } else if (target == nullptr && intruderIndex >= 0) {
                const Player& intr = enemies[intruderIndex];
                const vector<int>& goalField = (intruderIndex == carrierIndex)
                    ? dEnemyTerr : dMyFlag;
                vector<int> myField = bfs({myIdx});
                auto got = interceptField(intr, goalField, myField);
                interceptScratch = std::move(got.first);
                target = &interceptScratch;
                targetId = "intercept";
                goalX = got.second.first;
                goalY = got.second.second;
            } else if (target == nullptr) {
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
            bool immediate = danger(c.x, c.y, enemies);
            bool lookahead = dangerNext(c.x, c.y, enemies);
            bool badDanger = immediate || (lookahead && !desperate);
            bool terrPenalty = role == 'D' && territory(c.x, c.y) == enemyTerr;
            int safe = (badDanger || terrPenalty) ? 0 : 1;
            int enemyClear = enemies.empty() ? 9 : min(9, minCheb(c.x, c.y, enemies));
            int jitter = (int)(rnd() & 1023);
            array<int, 4> key;
            if (hereDanger) {
                key = {{safe, minCheb(c.x, c.y, enemies), -dval, jitter}};
            } else if (iCarry) {
                key = {{safe, -dval, enemyClear, jitter}};
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
