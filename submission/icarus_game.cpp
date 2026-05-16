#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

using namespace std;

static const int TARGET = 999;
static const int TIME_BUDGET_MS = 8;

struct Player {
    int pos;
    int cum;
};

array<vector<int>, 4> hist;
array<Player, 4> last_players;
array<bool, 4> alive = {true, true, true, true};
bool have_last = false;
int round_no = 0;
unsigned int rng_state = 2463534242u;

int rnd(int lo, int hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return lo + (int)(rng_state % (unsigned int)(hi - lo + 1));
}

int clamp_bid(int x) {
    if (x < 1) return 1;
    if (x > 100) return 100;
    return x;
}

double mean_recent(int idx, int fallback) {
    const vector<int>& h = hist[idx];
    if (h.empty()) return fallback;

    int start = max(0, (int)h.size() - 8);
    int sum = 0;
    for (int i = start; i < (int)h.size(); ++i) sum += h[i];
    return (double)sum / (double)((int)h.size() - start);
}

int max_recent(int idx, int fallback) {
    const vector<int>& h = hist[idx];
    if (h.empty()) return fallback;

    int start = max(0, (int)h.size() - 8);
    int best = 0;
    for (int i = start; i < (int)h.size(); ++i) best = max(best, h[i]);
    return best;
}

void update_memory(const array<Player, 4>& p) {
    if (!have_last) return;

    for (int i = 0; i < 4; ++i) {
        int dc = p[i].cum - last_players[i].cum;

        if (alive[i] && (dc < 0 || (p[i].pos == 0 && p[i].cum == 0 && last_players[i].cum > 0))) {
            alive[i] = false;
            hist[i].clear();
            continue;
        }

        if (alive[i] && 1 <= dc && dc <= 100) {
            hist[i].push_back(dc);
            if ((int)hist[i].size() > 24) hist[i].erase(hist[i].begin());
        }
    }
}

int heuristic_bid(const array<Player, 4>& p) {
    int my_pos = p[0].pos;
    int dist = TARGET - my_pos;

    vector<int> estimates;
    vector<int> highs;
    for (int i = 1; i < 4; ++i) {
        if (!alive[i]) continue;

        int fallback = 64;
        int odist = TARGET - p[i].pos;
        if (odist <= 100) fallback = max(45, min(100, odist));

        estimates.push_back((int)round(mean_recent(i, fallback)));
        highs.push_back(max_recent(i, fallback + 8));
    }

    if (estimates.empty()) return clamp_bid(dist <= 100 ? dist : 70);

    sort(estimates.begin(), estimates.end());
    sort(highs.begin(), highs.end());

    int expected_high = estimates.back();
    int expected_second = estimates.size() >= 2 ? estimates[estimates.size() - 2] : 55;
    int recent_high = highs.back();

    int leader = p[0].pos;
    for (int i = 1; i < 4; ++i) leader = max(leader, p[i].pos);
    int gap = leader - my_pos;

    int bid;
    if (dist <= 100) {
        int finish_bid = clamp_bid(dist);
        if (finish_bid <= expected_high - 2 || finish_bid <= recent_high - 5) {
            bid = finish_bid;
        } else {
            bid = min(finish_bid, max(1, expected_high - rnd(3, 8)));
        }
    } else if (gap > 160) {
        bid = max(expected_second + 6, expected_high - rnd(1, 4)) + rnd(-2, 5);
    } else if (gap > 70) {
        bid = max(expected_second + 2, expected_high - rnd(3, 7)) + rnd(-3, 3);
    } else if (my_pos >= leader - 20 && my_pos >= 760) {
        bid = min(expected_second + 1, expected_high - rnd(6, 12)) + rnd(-2, 2);
    } else {
        bid = expected_second + rnd(0, 6);
        bid = min(bid, expected_high - rnd(2, 7));
        bid += rnd(-3, 3);
    }

    if (round_no <= 1 && dist > 100) bid = 63 + rnd(-3, 5);
    if (dist > 100) bid = max(bid, 34);
    return clamp_bid(bid);
}

int sample_opponent_bid(int idx, const array<Player, 4>& p) {
    int dist = TARGET - p[idx].pos;

    if (dist <= 100 && rnd(1, 100) <= 62) {
        return clamp_bid(dist + rnd(-10, 10));
    }

    const vector<int>& h = hist[idx];
    if (!h.empty() && rnd(1, 100) <= 78) {
        int start = max(0, (int)h.size() - 10);
        int base = h[start + rnd(0, (int)h.size() - start - 1)];
        return clamp_bid(base + rnd(-10, 10));
    }

    int r = rnd(1, 100);
    if (r <= 52) return rnd(56, 82);
    if (r <= 84) return rnd(35, 72);
    return rnd(76, 98);
}

array<bool, 4> blocked_players(const array<Player, 4>& p, const array<int, 4>& bids) {
    array<bool, 4> blocked = {false, false, false, false};

    int high = -1;
    for (int i = 0; i < 4; ++i) {
        if (alive[i] || i == 0) high = max(high, bids[i]);
    }

    vector<int> candidates;
    for (int i = 0; i < 4; ++i) {
        if ((alive[i] || i == 0) && bids[i] == high) candidates.push_back(i);
    }

    if ((int)candidates.size() == 1) {
        blocked[candidates[0]] = true;
        return blocked;
    }

    int low_cum = 1 << 30;
    for (int idx : candidates) low_cum = min(low_cum, p[idx].cum + bids[idx]);
    for (int idx : candidates) {
        if (p[idx].cum + bids[idx] == low_cum) blocked[idx] = true;
    }

    return blocked;
}

int pessimistic_point_for_me(const array<int, 4>& pos) {
    int worse_rank = 0;
    for (int i = 0; i < 4; ++i) {
        if (pos[i] >= pos[0]) ++worse_rank;
    }
    if (worse_rank == 1) return 3;
    if (worse_rank == 2) return 2;
    if (worse_rank == 3) return 1;
    return 0;
}

double rollout_score(int my_bid, const array<Player, 4>& p) {
    array<int, 4> bids = {my_bid, 0, 0, 0};
    for (int i = 1; i < 4; ++i) {
        if (alive[i]) bids[i] = sample_opponent_bid(i, p);
    }

    array<bool, 4> blocked = blocked_players(p, bids);
    array<int, 4> pos;
    for (int i = 0; i < 4; ++i) {
        pos[i] = p[i].pos;
        if ((alive[i] || i == 0) && !blocked[i]) pos[i] += bids[i];
    }

    double value = (double)(pos[0] - p[0].pos);
    if (blocked[0]) value -= 15.0;

    int leader_after = *max_element(pos.begin(), pos.end());
    value -= max(0, leader_after - pos[0]) * 0.10;

    if (pos[0] >= TARGET) {
        value += 760.0;
        value += pessimistic_point_for_me(pos) * 260.0;
    }

    for (int i = 1; i < 4; ++i) {
        if (alive[i] && pos[i] >= TARGET && pos[0] < TARGET) value -= 470.0;
    }

    return value;
}

vector<int> build_candidates(const array<Player, 4>& p, int base) {
    array<bool, 101> seen{};
    vector<int> out;

    auto add = [&](int x) {
        x = clamp_bid(x);
        if (!seen[x]) {
            seen[x] = true;
            out.push_back(x);
        }
    };

    int dist = TARGET - p[0].pos;
    add(base);

    for (int d = -10; d <= 10; d += 2) add(base + d);
    for (int b = 38; b <= 90; b += 4) add(b);
    for (int b : {57, 61, 64, 67, 70, 74, 79, 85}) add(b);

    if (dist <= 100) {
        for (int d = -14; d <= 14; d += 2) add(dist + d);
    }

    for (int i = 1; i < 4; ++i) {
        if (!alive[i]) continue;
        int m = (int)round(mean_recent(i, 64));
        int hi = max_recent(i, 72);
        for (int d : {-10, -7, -4, -2, 0, 2}) {
            add(m + d);
            add(hi + d);
        }
    }

    return out;
}

int choose_bid(const array<Player, 4>& p) {
    int fallback = heuristic_bid(p);
    vector<int> candidates = build_candidates(p, fallback);

    vector<double> score(candidates.size(), 0.0);
    vector<int> count(candidates.size(), 0);

    auto deadline = chrono::steady_clock::now() + chrono::milliseconds(TIME_BUDGET_MS);

    for (int pass = 0; pass < 45; ++pass) {
        for (int i = 0; i < (int)candidates.size(); ++i) {
            score[i] += rollout_score(candidates[i], p);
            ++count[i];
        }
    }

    while (chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < (int)candidates.size(); ++i) {
            score[i] += rollout_score(candidates[i], p);
            ++count[i];
        }
    }

    int best = fallback;
    double best_score = -1e100;
    for (int i = 0; i < (int)candidates.size(); ++i) {
        double avg = score[i] / max(1, count[i]);
        avg += rnd(-100, 100) / 250.0;
        if (avg > best_score) {
            best_score = avg;
            best = candidates[i];
        }
    }

    return clamp_bid(best);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    rng_state ^= (unsigned int)time(nullptr);

    while (true) {
        array<Player, 4> p;
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            if (!(cin >> p[i].pos >> p[i].cum)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;

        update_memory(p);

        cout << choose_bid(p) << '\n' << flush;

        last_players = p;
        have_last = true;
        ++round_no;
    }

    return 0;
}
