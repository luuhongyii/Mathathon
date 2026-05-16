// mathathon.hpp — single-header C++ skeleton mirroring the Python kit.
//
// Drop-in pieces:
//   - TimeBudget  (wall-clock budget per move)
//   - GameState   (CRTP base for your adapter)
//   - RandomBot, GreedyBot, MinimaxBot, MCTSBot templates
//   - run_per_move_loop(...)  — stdin-line -> action-line submission loop
//
// Build: any C++17 compiler, no third-party deps.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace mathathon {

// ---------------------------------------------------------------------------
// TimeBudget
// ---------------------------------------------------------------------------

class TimeBudget {
 public:
  explicit TimeBudget(double seconds, double safety_margin = 0.01)
      : seconds_(seconds),
        safety_margin_(safety_margin),
        start_(std::chrono::steady_clock::now()) {}

  double elapsed() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_).count();
  }

  double remaining() const { return std::max(0.0, seconds_ - elapsed()); }
  bool expired() const { return remaining() <= safety_margin_; }

  TimeBudget fraction(double part) const {
    return TimeBudget(std::max(0.001, remaining() * part), safety_margin_);
  }

 private:
  double seconds_;
  double safety_margin_;
  std::chrono::steady_clock::time_point start_;
};

// ---------------------------------------------------------------------------
// CRTP GameState contract
// ---------------------------------------------------------------------------
//
// Your adapter:
//
//   struct MyState : mathathon::GameStateBase<MyState, int /*Action*/> {
//     int currentPlayer() const;
//     std::vector<int> legalActions() const;
//     MyState apply(int action) const;
//     bool isTerminal() const;
//     double score(int player) const;
//     static constexpr int numPlayers = 2;
//   };

template <typename Derived, typename ActionT>
struct GameStateBase {
  using Action = ActionT;
};

// ---------------------------------------------------------------------------
// Random helpers
// ---------------------------------------------------------------------------

inline std::mt19937 make_rng(uint64_t seed) { return std::mt19937(seed); }

template <typename State>
typename State::Action random_legal_action(const State& s, std::mt19937& rng) {
  auto legal = s.legalActions();
  std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
  return legal[d(rng)];
}

// ---------------------------------------------------------------------------
// RandomBot
// ---------------------------------------------------------------------------

template <typename State>
struct RandomBot {
  typename State::Action choose(const State& s, int player,
                                TimeBudget& /*budget*/, std::mt19937& rng) const {
    (void)player;
    return random_legal_action(s, rng);
  }
};

// ---------------------------------------------------------------------------
// GreedyBot — pick the action whose successor maximises evaluator(state, player).
// ---------------------------------------------------------------------------

template <typename State>
struct GreedyBot {
  using Action = typename State::Action;
  std::function<double(const State&, int)> evaluator;

  GreedyBot()
      : evaluator([](const State& s, int p) { return s.score(p); }) {}
  explicit GreedyBot(std::function<double(const State&, int)> e)
      : evaluator(std::move(e)) {}

  Action choose(const State& s, int player, TimeBudget& budget,
                std::mt19937& rng) const {
    auto legal = s.legalActions();
    Action best = legal[0];
    double best_v = -std::numeric_limits<double>::infinity();
    for (const auto& a : legal) {
      if (budget.expired()) break;
      double v = evaluator(s.apply(a), player);
      if (v > best_v) {
        best_v = v;
        best = a;
      }
    }
    return best;
  }
};

// ---------------------------------------------------------------------------
// MinimaxBot — alpha-beta, 2-player only.
// ---------------------------------------------------------------------------

template <typename State>
struct MinimaxBot {
  using Action = typename State::Action;
  int depth;
  std::function<double(const State&, int)> evaluator;

  MinimaxBot(int d = 4)
      : depth(d),
        evaluator([](const State& s, int p) { return s.score(p); }) {}
  MinimaxBot(int d, std::function<double(const State&, int)> e)
      : depth(d), evaluator(std::move(e)) {}

  Action choose(const State& s, int player, TimeBudget& budget,
                std::mt19937& rng) const {
    auto legal = s.legalActions();
    std::shuffle(legal.begin(), legal.end(), rng);
    Action best = legal[0];
    double best_v = -std::numeric_limits<double>::infinity();
    double alpha = -std::numeric_limits<double>::infinity();
    double beta = std::numeric_limits<double>::infinity();
    for (const auto& a : legal) {
      if (budget.expired()) break;
      double v = value_(s.apply(a), player, depth - 1, alpha, beta, budget);
      if (v > best_v) {
        best_v = v;
        best = a;
      }
      alpha = std::max(alpha, best_v);
    }
    return best;
  }

 private:
  double value_(const State& s, int root, int d, double alpha, double beta,
                TimeBudget& budget) const {
    if (d <= 0 || s.isTerminal() || budget.expired()) {
      return evaluator(s, root);
    }
    auto legal = s.legalActions();
    if (legal.empty()) return evaluator(s, root);
    if (s.currentPlayer() == root) {
      double v = -std::numeric_limits<double>::infinity();
      for (const auto& a : legal) {
        v = std::max(v,
                     value_(s.apply(a), root, d - 1, alpha, beta, budget));
        alpha = std::max(alpha, v);
        if (alpha >= beta || budget.expired()) break;
      }
      return v;
    }
    double v = std::numeric_limits<double>::infinity();
    for (const auto& a : legal) {
      v = std::min(v, value_(s.apply(a), root, d - 1, alpha, beta, budget));
      beta = std::min(beta, v);
      if (alpha >= beta || budget.expired()) break;
    }
    return v;
  }
};

// ---------------------------------------------------------------------------
// MCTSBot — opponent-aware UCB with full parent-pointer backprop.
// ---------------------------------------------------------------------------

template <typename State>
struct MCTSBot {
  using Action = typename State::Action;

  int simulations = 1000;
  int rollout_depth = 80;
  double exploration = 1.4;
  std::function<double(const State&, int)> evaluator;

  MCTSBot()
      : evaluator([](const State& s, int p) { return s.score(p); }) {}

  Action choose(const State& s, int /*player*/, TimeBudget& budget,
                std::mt19937& rng) const {
    NodeStore store;
    int root_id = store.create(/*parent*/ -1, /*mover*/ -1, Action{});
    store.nodes[root_id].untried = s.legalActions();
    if (store.nodes[root_id].untried.empty()) return random_legal_action(s, rng);

    for (int sim = 0; sim < simulations; ++sim) {
      if (budget.expired()) break;
      State current = s;
      int leaf = select_(store, root_id, current, rng);
      auto scores = rollout_(current, rng, budget);
      backprop_(store, leaf, scores);
    }

    const auto& root = store.nodes[root_id];
    if (root.children.empty()) return random_legal_action(s, rng);
    int best_child = root.children.front();
    int best_visits = -1;
    for (int cid : root.children) {
      if (store.nodes[cid].visits > best_visits) {
        best_visits = store.nodes[cid].visits;
        best_child = cid;
      }
    }
    return store.nodes[best_child].action;
  }

 private:
  struct Node {
    int parent = -1;
    int mover = -1;
    Action action{};
    std::vector<Action> untried;
    std::vector<int> children;  // indices into NodeStore::nodes
    int visits = 0;
    double value = 0.0;
  };

  struct NodeStore {
    std::vector<Node> nodes;
    int create(int parent, int mover, const Action& action) {
      Node n;
      n.parent = parent;
      n.mover = mover;
      n.action = action;
      nodes.push_back(std::move(n));
      return static_cast<int>(nodes.size()) - 1;
    }
  };

  int select_(NodeStore& store, int node_id, State& s,
              std::mt19937& rng) const {
    while (!s.isTerminal()) {
      Node& node = store.nodes[node_id];
      if (!node.untried.empty()) {
        std::uniform_int_distribution<size_t> d(0, node.untried.size() - 1);
        size_t idx = d(rng);
        Action a = node.untried[idx];
        node.untried.erase(node.untried.begin() + idx);
        int mover = s.currentPlayer();
        s = s.apply(a);
        int child_id = store.create(node_id, mover, a);
        store.nodes[child_id].untried = s.isTerminal() ? std::vector<Action>{}
                                                       : s.legalActions();
        // ``node`` reference may be invalidated by the push_back above.
        store.nodes[node_id].children.push_back(child_id);
        return child_id;
      }
      if (node.children.empty()) return node_id;
      int parent_visits = std::max(1, node.visits);
      int best_child = node.children.front();
      double best_u = -std::numeric_limits<double>::infinity();
      for (int cid : node.children) {
        const Node& ch = store.nodes[cid];
        double u;
        if (ch.visits == 0) {
          u = std::numeric_limits<double>::infinity();
        } else {
          u = (ch.value / ch.visits) +
              exploration * std::sqrt(std::log(parent_visits) / ch.visits);
        }
        if (u > best_u) {
          best_u = u;
          best_child = cid;
        }
      }
      s = s.apply(store.nodes[best_child].action);
      node_id = best_child;
    }
    return node_id;
  }

  std::vector<double> rollout_(State& s, std::mt19937& rng,
                                TimeBudget& budget) const {
    int depth = 0;
    while (!s.isTerminal() && depth < rollout_depth && !budget.expired()) {
      auto legal = s.legalActions();
      if (legal.empty()) break;
      std::uniform_int_distribution<size_t> d(0, legal.size() - 1);
      s = s.apply(legal[d(rng)]);
      ++depth;
    }
    std::vector<double> scores(State::numPlayers, 0.0);
    for (int p = 0; p < State::numPlayers; ++p) scores[p] = evaluator(s, p);
    return scores;
  }

  void backprop_(NodeStore& store, int node_id,
                 const std::vector<double>& scores) const {
    while (node_id >= 0) {
      Node& n = store.nodes[node_id];
      n.visits += 1;
      if (n.mover >= 0 && n.mover < static_cast<int>(scores.size())) {
        n.value += scores[n.mover];
      }
      node_id = n.parent;
    }
  }
};

// ---------------------------------------------------------------------------
// stdin/stdout submission loop
// ---------------------------------------------------------------------------
//
// Usage:
//   int main() {
//     mathathon::run_per_move_loop<MyState>(
//       [](const std::string& line) { return parseState(line); },
//       [](const MyState& s, int player, mathathon::TimeBudget& b,
//          std::mt19937& rng) {
//         return MyBot{}.choose(s, player, b, rng);
//       },
//       [](const auto& action) -> std::string { return formatAction(action); },
//       /*time_per_move=*/0.2,
//       /*seed=*/42);
//   }

template <typename State, typename ParseFn, typename ChooseFn,
          typename FormatFn>
void run_per_move_loop(ParseFn parse, ChooseFn choose, FormatFn format,
                       double time_per_move = 0.2, uint64_t seed = 0) {
  std::mt19937 rng(seed);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    State s = parse(line);
    TimeBudget budget(time_per_move);
    int player = s.currentPlayer();
    auto action = choose(s, player, budget, rng);
    std::cout << format(action) << "\n";
    std::cout.flush();
  }
}

}  // namespace mathathon
