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
// Pack / best-response opponents bid in a tight cluster under each other.
// When detected, prefer riding just below the observed ceiling instead of
// the independent-model EV peak (which over-bids and keeps getting blocked).

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

static const int TARGET = 999;

// Tunable strategy parameters. These are the SINGLE source of truth shared
// (by value) with icarus_game.py and the LearnedBot in icarus_rl_train.py.
// The RL tuner searches this exact set; keep the three files in lockstep.
struct Params {
    // Values below are the CEM-tuned set (eval: 598/768 outright wins).
    double no_info_floor   = 0.0171807;   // uniform floor on every opponent dist
    double recency_decay   = 0.624394;    // kernel recency weight base
    double tight_sd_mul    = 1.29265;     // sigT = mul*sd + add ...
    double tight_sd_add    = 0.280066;
    double trust_base      = 0.05;        // kTrust = base + sd_mul*sd + n_mul/n
    double trust_sd_mul    = 0.462153;
    double trust_n_mul     = 7.00283;
    double trend_mul       = 0.508882;    // trend shift = mul * slope
    double block_up        = 8.88151;     // blockBias increment when blocked
    double block_down      = 1.15079;     // blockBias decay when not blocked
    double block_cap       = 66.0955;     // blockBias hard ceiling
    double finish_safe     = 0.697257;    // pSafe needed to take a finish shot
    double setup_safe      = 0.691082;    // pSafe needed to spend a setup round
    // Tight-kernel (best-responder pack) branch.
    double tight_sd_thresh = 1.5;      // sd below this -> tight reliable shield
    double tight_sigma     = 0.2;      // sigT override inside the tight branch
    double tight_w_boost   = 0.0845868;// extra weight on the tight kernel
    // Shield / pack-riding thresholds.
    double shield_safe1    = 0.661246; // ride ceiling-1 if pSafe >= this ...
    double shield_evfrac1  = 0.981208; // ... and EV >= bestEV * this
    double shield_safe2    = 0.793981; // deeper shield candidate gate
    double shield_safe3    = 0.917121; // final shield accept gate ...
    double shield_evfrac3  = 0.886851; // ... and EV >= bestEV * this
    double match_safe      = 0.545142; // low-cum match gate
    double match_evfrac    = 0.859635; // low-cum match EV fraction
    double block_shift_cap = 19.5093;  // max effective blockBias shift in pSafe
    // No-info opening: Gaussian prior over opponents' round-0 bids. Real
    // lobbies open conservatively (~44-60); a uniform model over-bids R0.
    double open_mean       = 52.0;
    double open_sd         = 16.0;
};

static const Params PARAMS;

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

// Field's top bid per round, for the descending-war guard (see warCap).
vector<int> topHist;

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
    if (sigma < 0.05) sigma = 0.05;
    for (int k = 0; k < n; ++k) {
        double recw = pow(PARAMS.recency_decay, n - 1 - k);  // recency decay
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
        // No data: Gaussian opening prior (real lobbies open ~44-60), not a
        // uniform model -- the latter makes the EV-max round-0 bid an overbid.
        double denom = 2.0 * PARAMS.open_sd * PARAMS.open_sd;
        for (int b = 1; b <= 100; ++b) {
            double dd = b - PARAMS.open_mean;
            p[b] = exp(-(dd * dd) / denom);
        }
        normalize(p);
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
        // Tight recent cluster -> treat as a reliable shield (best-responders).
        double sigT;
        bool tightBranch = (sd < PARAMS.tight_sd_thresh);
        if (tightBranch) sigT = PARAMS.tight_sigma;
        else sigT = clamp(PARAMS.tight_sd_mul * sd + PARAMS.tight_sd_add, 0.35, 9.0);
        double sigW = max(12.0, 2.2 * sigT);
        array<double, 101> tight = kernelDist(h, sigT);
        array<double, 101> wide = kernelDist(h, sigW);

        double kTrust = clamp(PARAMS.trust_base + PARAMS.trust_sd_mul * sd +
                                  PARAMS.trust_n_mul / n,
                              0.25, 9.0);
        double w = n / (n + kTrust);
        if (tightBranch) w = min(0.92, w + PARAMS.tight_w_boost);
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
            // m1, m2 are the two half-window *means*; slope then divides their
            // difference by `half` again (matches the Python reference, which
            // divides by `half` twice in total -- once per mean, once here).
            m1 /= half;
            m2 /= half;
            double slope = (m2 - m1) / half;
            double shift = clamp(PARAMS.trend_mul * slope, -9.0, 9.0);

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
    double floorW = PARAMS.no_info_floor;
    for (int b = 1; b <= 100; ++b)
        p[b] = (1.0 - floorW) * p[b] + floorW / 100.0;

    // Finisher adjustment: an opponent within finishing range will likely bid
    // at or above its remaining distance to cross the line -- BUT only if its
    // recent bidding shows it will actually bid that high. A spiralled-down
    // crawler (recent bid far below its distance) keeps crawling; modelling it
    // as a finisher overestimates its bids and makes us over-bid into the pack
    // and get blocked.
    int dist = TARGET - oppPos;
    if (dist >= 1 && dist <= 100 && (n == 0 || h.back() >= dist - 4)) {
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
    int lastBid = 0;
    array<double, 101> cdf;  // cdf[b] = P(bid <= b), cdf[0] = 0
};

// True when opponents look like a spiralling best-response pack.
static bool detectPack(const vector<OppInfo>& opps) {
    if ((int)opps.size() < 2) return false;
    vector<int> recent;
    for (const OppInfo& o : opps) {
        if (o.lastBid >= 1) recent.push_back(o.lastBid);
    }
    if ((int)recent.size() < 2) return false;
    int lo = recent[0], hi = recent[0];
    for (int b : recent) {
        lo = min(lo, b);
        hi = max(hi, b);
    }
    return (hi - lo) <= 10;
}

static int recentCeil(const vector<OppInfo>& opps) {
    vector<int> bids;
    for (const OppInfo& o : opps)
        if (o.lastBid >= 1) bids.push_back(o.lastBid);
    if (bids.empty()) return 0;
    sort(bids.begin(), bids.end(), greater<int>());
    return bids.size() >= 2 ? bids[1] : bids[0];
}

// Safe bid ceiling during a descending bid war (else a no-op high cap of 1000).
// In a war the pack's top bid drops ~2-3/round; the EV/shield logic still
// targets last round's stale (higher) top, so a bid "safely under" it ties the
// new top and gets blocked. Extrapolate the descent and cap our bid a margin
// under the predicted next top. The monotonic check keeps the guard from
// mis-firing on noisy opponents that are not in a war.
static int warCap(const vector<int>& th, bool pack) {
    int m = (int)th.size();
    if (!pack || m < 3) return 1000;
    int a = th[m - 3], b = th[m - 2], c = th[m - 1];
    if (a < b || b < c) return 1000;  // require a consistent descent
    double drop = (a - c) / 2.0;
    if (drop < 1.0) return 1000;
    if (drop > 12.0) drop = 12.0;
    int cap = (int)(c - drop - 2.0);
    return cap >= 1 ? cap : 1;
}

int chooseBid(const array<int, 4>& pos, const array<int, 4>& cum) {
    int myCum = cum[0], myDist = TARGET - pos[0];

    vector<OppInfo> opps;
    for (int i = 1; i < 4; ++i) {
        if (!alive[i]) continue;
        OppInfo o;
        o.cum = cum[i];
        o.dist = TARGET - pos[i];
        const vector<int>& h = bidHist[i];
        o.lastBid = h.empty() ? 0 : h.back();
        array<double, 101> d = oppDist(i, pos[i]);
        o.cdf[0] = 0.0;
        for (int b = 1; b <= 100; ++b) o.cdf[b] = o.cdf[b - 1] + d[b];
        opps.push_back(o);
    }

    // Track the field's top bid each round for the descending-war guard.
    int curTop = 0;
    for (const OppInfo& o : opps) curTop = max(curTop, o.lastBid);
    if (curTop >= 1) {
        topHist.push_back(curTop);
        if ((int)topHist.size() > 12) topHist.erase(topHist.begin());
    }

    if (opps.empty()) return 1;  // alone: cannot move, just answer validly

    int ceilBid = recentCeil(opps);

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
    // The effective shift is capped so a large accumulated blockBias cannot
    // push `hi` far past 100: that would saturate every opponent CDF to 1.0,
    // collapse pSafe to 0, and lock the bot into bidding 1-2 for many rounds.
    auto pSafe = [&](int b) -> double {
        int shift = (int)lround(blockBias);
        if (shift > (int)PARAMS.block_shift_cap) shift = (int)PARAMS.block_shift_cap;
        if (ceilBid >= 8 && b + 3 <= ceilBid)
            shift = min(shift, max(0, ceilBid - b - 2));
        int hi = b + shift;
        if (hi > 100) hi = 100;  // saturate gently at the bid ceiling
        double pb = 1.0;
        for (const OppInfo& o : opps)
            pb *= (o.cum < myCum) ? cdfAt(o, hi - 1) : cdfAt(o, hi);
        return 1.0 - pb;
    };

    if (myDist >= 1 && myDist <= 100) {
        // I can reach the line this round. Pick the finishing bid (>= myDist)
        // most likely to get through unblocked.
        // A bigger finishing bid (>= myDist) only helps when a rival is ALSO
        // finishing this round: overshooting TARGET wins the final position
        // tie-break. If nobody else can finish, bid the minimal safe finish
        // -- a b*pSafe over-bid would stick out above the pack and get us
        // blocked while leading.
        bool oppFinishing = false;
        for (const auto &o : opps)
            if (o.dist <= 100 && o.lastBid >= o.dist - 4) oppFinishing = true;
        int bFin = myDist;
        double psFin = -1.0, finEV = -1.0;
        for (int b = myDist; b <= 100; ++b) {
            double ps = pSafe(b);
            double ev = oppFinishing ? (double)b * ps : ps;
            if (ev > finEV || (ev == finEV && ps > psFin)) {
                finEV = ev;
                psFin = ps;
                bFin = b;
            }
        }

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

        // A safe finish dominates everything else -- take it.
        if (psFin >= PARAMS.finish_safe) return bFin;

        // If a rival is already at least as far as we are and can finish, there
        // is no time to set up. But it counts as urgent only if its recent bid
        // is actually large enough to finish: a rival that could reach the line
        // yet is bidding far below its distance (a spiralled-down pack) will not
        // finish next round, and panicking with a doomed finishing bid above
        // the pack just gets us blocked.
        bool urgentFinish = false;
        for (const OppInfo& o : opps)
            if (o.dist <= 100 && TARGET - o.dist >= pos[0] && o.lastBid >= o.dist - 4)
                urgentFinish = true;
        if (urgentFinish) return bFin;

        if (bSet > 0 && pSafe(bSet) >= PARAMS.setup_safe) return bSet;
        // EV-best setup is too risky -- but DON'T fall back to bFin: a doomed
        // full-distance bid sticks out alone above a low pack and is a certain
        // block that freezes us. Step down to the safest setup bid available.
        int safeSet = 0;
        for (int b = 1; b < myDist; ++b)
            if (pSafe(b) >= PARAMS.setup_safe) safeSet = b;
        if (safeSet > 0) return safeSet;
        return bSet > 0 ? bSet : bFin;
    }

    // Cannot finish yet: maximise expected position gain.
    int best = 1;
    double bestEV = -1.0;
    for (int b = 1; b <= 100; ++b) {
        double ev = b * pSafe(b);
        if (ev > bestEV) { bestEV = ev; best = b; }
    }

    bool pack = detectPack(opps) || blockBias >= 8.0;
    // Descending-war guard: cap every non-finishing bid so we never tie the
    // pack's (descending) top -- see warCap.
    int cap = warCap(topHist, pack);
    if (pack && ceilBid >= 8) {
        int shield = clamp(ceilBid - 1, 1, 100);
        double shieldEV = shield * pSafe(shield);
        // Ride just under the pack when it beats the naive EV peak.
        if (pSafe(shield) >= PARAMS.shield_safe1 &&
            shieldEV >= bestEV * PARAMS.shield_evfrac1)
            return min(shield, cap);
        for (int d = 2; d <= 4; ++d) {
            int sb = clamp(ceilBid - d, 1, 100);
            double ev = sb * pSafe(sb);
            if (pSafe(sb) >= PARAMS.shield_safe2 && ev > shieldEV) {
                shieldEV = ev;
                shield = sb;
            }
        }
        if (pSafe(shield) >= PARAMS.shield_safe3 &&
            shieldEV >= bestEV * PARAMS.shield_evfrac3)
            return min(shield, cap);
        if (best > ceilBid) best = clamp(ceilBid - 1, 1, 100);
    }

    // Lowest cumulative: can match the pack without losing tie-breaks as often.
    int minOppCum = opps[0].cum;
    for (const OppInfo& o : opps) minOppCum = min(minOppCum, o.cum);
    if (myCum <= minOppCum && pack && ceilBid >= 5) {
        int match = clamp(ceilBid, 1, 100);
        if (pSafe(match) >= PARAMS.match_safe &&
            match * pSafe(match) > bestEV * PARAMS.match_evfrac)
            return min(match, cap);
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
    return min(best, cap);
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
            if (blocked) blockBias = min(PARAMS.block_cap, blockBias + PARAMS.block_up);
            else         blockBias = max(0.0, blockBias - PARAMS.block_down);
        }

        updateMemory(pos, cum);
        cout << clampBid(chooseBid(pos, cum)) << "\n" << flush;

        lastPos = pos;
        lastCum = cum;
        haveLast = true;
    }
    return 0;
}
