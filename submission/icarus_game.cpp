#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

static const int TARGET = 999;

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

int choose_bid(const array<Player, 4>& p) {
    int my_pos = p[0].pos;
    int dist = TARGET - my_pos;

    vector<int> opp_est;
    vector<int> opp_max;
    for (int i = 1; i < 4; ++i) {
        if (!alive[i]) continue;

        int fallback = 64;
        int odist = TARGET - p[i].pos;
        if (odist <= 100) fallback = max(45, min(100, odist));

        opp_est.push_back((int)round(mean_recent(i, fallback)));
        opp_max.push_back(max_recent(i, fallback + 8));
    }

    if (opp_est.empty()) {
        if (dist <= 100) return clamp_bid(dist);
        return 70;
    }

    sort(opp_est.begin(), opp_est.end());
    sort(opp_max.begin(), opp_max.end());

    int expected_high = opp_est.back();
    int expected_second = opp_est.size() >= 2 ? opp_est[opp_est.size() - 2] : 55;
    int recent_high = opp_max.back();

    int leader = p[0].pos;
    for (int i = 1; i < 4; ++i) leader = max(leader, p[i].pos);
    int gap = leader - my_pos;

    int bid;

    if (dist <= 100) {
        int finish_bid = clamp_bid(dist);

        // If finishing is likely below the table's high bid, take it.
        if (finish_bid <= expected_high - 2 || finish_bid <= recent_high - 5) {
            bid = finish_bid;
        } else {
            // Otherwise slip under the expected high instead of being blocked.
            bid = min(finish_bid, max(1, expected_high - rnd(3, 8)));
        }
    } else if (gap > 160) {
        bid = max(expected_second + 6, expected_high - rnd(1, 4));
        bid += rnd(-2, 5);
    } else if (gap > 70) {
        bid = max(expected_second + 2, expected_high - rnd(3, 7));
        bid += rnd(-3, 3);
    } else if (my_pos >= leader - 20 && my_pos >= 760) {
        // Late while leading/near leading: avoid being Icarus.
        bid = min(expected_second + 1, expected_high - rnd(6, 12));
        bid += rnd(-2, 2);
    } else {
        // Normal mode: aim near the second-highest expected bid.
        bid = expected_second + rnd(0, 6);
        bid = min(bid, expected_high - rnd(2, 7));
        bid += rnd(-3, 3);
    }

    // Opening prior: against random bots, 62-69 is usually strong.
    if (round_no <= 1 && dist > 100) {
        bid = 63 + rnd(-3, 5);
    }

    // Never be too timid unless finishing exactly requires it.
    if (dist > 100) bid = max(bid, 34);

    return clamp_bid(bid);
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
