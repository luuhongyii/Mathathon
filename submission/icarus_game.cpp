// Icarus Game submission.
//
// Rules: 4 players race to position 999. Each round every player bids 1..100;
// the bid is added to a cumulative total. The highest bidder is "Icarus" and
// is blocked (does not move); on a tie for highest, the tied player(s) with
// the lowest cumulative total are blocked. Everyone not blocked advances by
// their bid. Final ranking is by position (pessimistic on ties): 3/2/1/0 pts.
//
// Strategy:
//  * Learn each opponent's bid distribution from observed cumulative deltas.
//  * P(I am blocked | bid b) is computed EXACTLY: with the tie-break rule I am
//    blocked iff every lower-cumulative opponent bids < b and every other
//    opponent bids <= b -> a simple product over opponent CDFs.
//  * Cannot finish yet: bid b maximising expected gain  b * (1 - P(blocked)).
//  * Can finish (distance <= 100): finishing now dominates finishing later, so
//    pick the finishing bid with the highest success probability. Only when
//    finishing is genuinely risky AND every opponent is far away do we spend a
//    round advancing safely to set up a safer finish.
// Everything is exact and deterministic -- no RNG, no timing, cannot time out.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

static const int TARGET = 999;

array<vector<int>, 4> bidHist;   // observed bids (cumulative deltas)
array<bool, 4> alive = {true, true, true, true};
array<int, 4> zeroStreak = {0, 0, 0, 0};
array<int, 4> lastPos{}, lastCum{};
bool haveLast = false;

// Feedback term: when I get blocked, opponents bid lower than the model thinks
// (e.g. a pack of best-responders spiralling their bids down). This shifts my
// risk assessment downward until I am safely under them, then decays away.
// Down-only: probing upward is unsafe near a hard ceiling (a constant bidder).
double blockBias = 0.0;

int clampBid(int x) { return x < 1 ? 1 : (x > 100 ? 100 : x); }

void updateMemory(const array<int, 4>& pos, const array<int, 4>& cum) {
    if (!haveLast) return;
    for (int i = 0; i < 4; ++i) {
        if (!alive[i]) continue;
        int dc = cum[i] - lastCum[i];

        // A player whose totals were reset to 0 has died.
        if (dc < 0 || (pos[i] == 0 && cum[i] == 0 && lastCum[i] > 0)) {
            alive[i] = false;
            bidHist[i].clear();
            continue;
        }
        // A live player always bids >= 1, so its cumulative total grows every
        // round. No growth means the player was not prompted (dead).
        if (dc == 0) {
            if (++zeroStreak[i] >= 2) {
                alive[i] = false;
                bidHist[i].clear();
            }
            continue;
        }
        zeroStreak[i] = 0;

        if (dc >= 1 && dc <= 100) {
            bidHist[i].push_back(dc);
            if ((int)bidHist[i].size() > 32) bidHist[i].erase(bidHist[i].begin());
        }
    }
}

void normalize(array<double, 101>& a) {
    double s = 0.0;
    for (int b = 1; b <= 100; ++b) s += a[b];
    if (s <= 0.0) {
        for (int b = 1; b <= 100; ++b) a[b] = 1.0 / 100.0;
    } else {
        for (int b = 1; b <= 100; ++b) a[b] /= s;
    }
}

// Recency-weighted Gaussian-kernel estimate of a bid distribution.
array<double, 101> kernelDist(const vector<int>& h, double sigma) {
    array<double, 101> d{};
    int n = (int)h.size();
    for (int k = 0; k < n; ++k) {
        double recw = pow(0.70, n - 1 - k);  // exponential recency decay
        int v = h[k];
        for (int b = 1; b <= 100; ++b) {
            double dd = b - v;
            d[b] += recw * exp(-dd * dd / (2.0 * sigma * sigma));
        }
    }
    normalize(d);
    return d;
}

// Probability distribution of opponent i's next bid over 1..100.
array<double, 101> oppDist(int i, int oppPos) {
    const vector<int>& h = bidHist[i];
    int n = (int)h.size();
    array<double, 101> p{};

    if (n == 0) {
        // No data: uniform (max entropy -- unbiased, cannot be exploited).
        for (int b = 1; b <= 100; ++b) p[b] = 1.0 / 100.0;
    } else {
        // Spread of the opponent's recent bids. Measured over a SHORT recent
        // window: an opponent that ramped up early and then plateaued (e.g. a
        // snipe bot stuck at 95) must be read as a low-variance reliable
        // shield, not penalised forever for stale ramp-up history.
        int start = max(0, n - 6);
        double mean = 0.0;
        int c = 0;
        for (int k = start; k < n; ++k) { mean += h[k]; ++c; }
        mean /= c;
        double var = 0.0;
        for (int k = start; k < n; ++k) { double d = h[k] - mean; var += d * d; }
        double sd = sqrt(var / c);

        // A tight kernel is the point estimate; a wide kernel -- still centred
        // on observed bids -- is the hedge against model error. The hedge is
        // centred on the data, NOT uniform, so a reliable high bidder is still
        // treated as a reliable shield. Trust the tight estimate more with more
        // data and lower spread; with little data, lean on the wide hedge.
        double sigT = clamp(0.85 * sd + 0.35, 0.35, 9.0);
        double sigW = max(15.0, 2.5 * sigT);
        array<double, 101> tight = kernelDist(h, sigT);
        array<double, 101> wide = kernelDist(h, sigW);

        double kTrust = clamp(0.3 + 0.33 * sd + 4.5 / n, 0.3, 9.0);
        double w = n / (n + kTrust);
        for (int b = 1; b <= 100; ++b)
            p[b] = w * tight[b] + (1.0 - w) * wide[b];

        // Trend extrapolation: if the opponent's bids are climbing or falling,
        // project the distribution one step in that direction so the estimate
        // is not permanently lagging behind a moving opponent.
        if (n >= 4) {
            int L = min(n, 8), half = L / 2;
            double m1 = 0.0, m2 = 0.0;
            for (int k = n - L; k < n - L + half; ++k) m1 += h[k];
            for (int k = n - half; k < n; ++k) m2 += h[k];
            // per-round slope, projected ~1.5 rounds forward
            double slope = (m2 / half - m1 / half) / half;
            double shift = clamp(1.5 * slope, -9.0, 9.0);

            // Skip extrapolation when the most recent bids are already flat:
            // an opponent that ramped up early and has since plateaued (a
            // snipe bot locked at 95) must read as flat. Otherwise the stale
            // ramp still in the 8-bid window keeps projecting it upward and
            // smears its true spike -- and the bot stops riding the shield.
            int rl = min(n, 4);
            double rmean = 0.0, rspread = 0.0;
            for (int k = n - rl; k < n; ++k) rmean += h[k];
            rmean /= rl;
            for (int k = n - rl; k < n; ++k) rspread += fabs(h[k] - rmean);
            rspread /= rl;
            if (rspread < 2.0) shift = 0.0;

            if (fabs(shift) > 0.3) {
                array<double, 101> sh{};
                for (int b = 1; b <= 100; ++b) {
                    double src = b - shift;
                    int lo = (int)floor(src);
                    double fr = src - lo;
                    int a = clamp(lo, 1, 100), bb = clamp(lo + 1, 1, 100);
                    sh[b] = p[a] * (1.0 - fr) + p[bb] * fr;
                }
                normalize(sh);
                p = sh;
            }
        }
    }

    // Tiny uniform floor: never assign a bid exactly zero probability.
    double floorW = 0.01;
    for (int b = 1; b <= 100; ++b)
        p[b] = (1.0 - floorW) * p[b] + floorW / 100.0;

    // Finisher adjustment: an opponent within finishing range will likely bid
    // at or above its remaining distance to cross the line.
    int dist = TARGET - oppPos;
    if (dist >= 1 && dist <= 100) {
        array<double, 101> fin{};
        double fs = 0.0;
        for (int b = dist; b <= 100; ++b) {
            double w = exp(-(double)(b - dist) / 22.0);
            fin[b] = w;
            fs += w;
        }
        if (fs > 0.0) {
            for (int b = 1; b <= 100; ++b) fin[b] /= fs;
            double mix = 0.5;
            for (int b = 1; b <= 100; ++b)
                p[b] = (1.0 - mix) * p[b] + mix * fin[b];
        }
    }
#ifdef DBG
    {
        int mode = 1;
        for (int b = 1; b <= 100; ++b) if (p[b] > p[mode]) mode = b;
        fprintf(stderr, "[opp%d n=%d mode=%d p80=%.3f p90=%.3f p95=%.3f] ", i,
                n, mode, p[80], p[90], p[95]);
    }
#endif
    return p;
}

struct OppInfo {
    int cum;
    int dist;
    array<double, 101> cdf;  // cdf[b] = P(bid <= b), cdf[0] = 0
};

int chooseBid(const array<int, 4>& pos, const array<int, 4>& cum) {
    int myCum = cum[0], myDist = TARGET - pos[0];

    vector<OppInfo> opps;
    for (int i = 1; i < 4; ++i) {
        if (!alive[i]) continue;
        OppInfo o;
        o.cum = cum[i];
        o.dist = TARGET - pos[i];
        array<double, 101> d = oppDist(i, pos[i]);
        o.cdf[0] = 0.0;
        for (int b = 1; b <= 100; ++b) o.cdf[b] = o.cdf[b - 1] + d[b];
        opps.push_back(o);
    }

    if (opps.empty()) return 1;  // alone: cannot move, just answer validly

    // Exact probability that my bid b is the blocked one.
    // Blocked iff every lower-cum opponent bids < b and every equal/higher-cum
    // opponent bids <= b (an opponent tied at b with lower cum saves me).
    // The block-penalty shift evaluates risk as if opponents bid that much
    // lower than the model thinks -- the correction learned from being blocked.
    auto cdfAt = [&](const OppInfo& o, int idx) -> double {
        if (idx <= 0) return 0.0;
        if (idx >= 100) return 1.0;
        return o.cdf[idx];
    };
    auto pSafe = [&](int b) -> double {
        int hi = b + (int)lround(blockBias);
        double pb = 1.0;
        for (const OppInfo& o : opps)
            pb *= (o.cum < myCum) ? cdfAt(o, hi - 1) : cdfAt(o, hi);
        return 1.0 - pb;
    };

    if (myDist >= 1 && myDist <= 100) {
        // I can reach the line this round. Pick the finishing bid (>= myDist)
        // most likely to get through unblocked.
        int bFin = myDist;
        double psFin = -1.0;
        for (int b = myDist; b <= 100; ++b) {
            double ps = pSafe(b);
            if (ps > psFin) { psFin = ps; bFin = b; }
        }

        // A safe finish dominates everything else -- take it.
        if (psFin >= 0.80) return bFin;

        // An opponent whose distance is within one bid can cross the line next
        // round: there is no time to set up, take the best finish shot now.
        bool oppCanFinish = false;
        for (const OppInfo& o : opps)
            if (o.dist <= 100) oppCanFinish = true;
        if (oppCanFinish) return bFin;

        // Finishing now is risky -- my finishing bid (>= myDist) sticks out
        // above a spiralled-down pack and would make me Icarus -- and nobody
        // else can finish next round. Spend this round advancing safely under
        // the pack, shrinking the remaining distance so next round's finishing
        // bid is itself low enough to hide in the pack and get through.
        int bSet = 0;
        double bestEV = -1.0;
        for (int b = 1; b < myDist; ++b) {
            double ev = b * pSafe(b);
            if (ev > bestEV) { bestEV = ev; bSet = b; }
        }
        if (bSet > 0 && pSafe(bSet) >= 0.70) return bSet;
        return bFin;
    }

    // Cannot finish yet: maximise expected position gain.
    int best = 1;
    double bestEV = -1.0;
    for (int b = 1; b <= 100; ++b) {
        double ev = b * pSafe(b);
        if (ev > bestEV) { bestEV = ev; best = b; }
    }
#ifdef DBG
    fprintf(stderr, "dist=%d bias=%.0f best=%d EV=%.1f | ", myDist, blockBias,
            best, bestEV);
    for (int b = 60; b <= 90; b += 5)
        fprintf(stderr, "%d:%.0f(s%.2f) ", b, b * pSafe(b), pSafe(b));
    for (const OppInfo& o : opps)
        fprintf(stderr, "| cum%d d%d cdf[85,90,94,95]=%.2f,%.2f,%.2f,%.2f ",
                o.cum, o.dist, o.cdf[85], o.cdf[90], o.cdf[94], o.cdf[95]);
    fprintf(stderr, "\n");
#endif
    return best;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    while (true) {
        array<int, 4> pos{}, cum{};
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            if (!(cin >> pos[i] >> cum[i])) {
                ok = false;
                break;
            }
        }
        if (!ok) break;

        // Did my previous bid get blocked? (position unchanged, total grew.)
        if (haveLast) {
            int myBid = cum[0] - lastCum[0];
            bool blocked = (pos[0] == lastPos[0]) && myBid > 0;
            if (blocked) blockBias = min(48.0, blockBias + 9.0);
            else         blockBias = max(0.0, blockBias - 3.0);
        }

        updateMemory(pos, cum);
        cout << clampBid(chooseBid(pos, cum)) << "\n" << flush;

        lastPos = pos;
        lastCum = cum;
        haveLast = true;
    }
    return 0;
}
