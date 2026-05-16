// nim_example.cpp — minimal C++ Nim adapter using mathathon.hpp.
//
// Build:  g++ -std=c++17 -O2 nim_example.cpp -o nim
// Run local self-test (no stdin needed):
//   ./nim --selftest
// Run as platform bot (per-move stdio):
//   ./nim
//   stdin lines: "<pile>"
//   stdout lines: "<take>"

#include <iostream>
#include <sstream>
#include <string>

#include "mathathon.hpp"

struct NimState : mathathon::GameStateBase<NimState, int> {
  int pile = 21;
  int current_index = 0;
  int max_take = 3;
  static constexpr int numPlayers = 2;

  int currentPlayer() const { return current_index; }

  std::vector<int> legalActions() const {
    std::vector<int> out;
    if (isTerminal()) return out;
    for (int t = 1; t <= std::min(max_take, pile); ++t) out.push_back(t);
    return out;
  }

  NimState apply(int action) const {
    NimState n = *this;
    n.pile -= action;
    n.current_index = 1 - current_index;
    return n;
  }

  bool isTerminal() const { return pile <= 0; }

  double score(int player) const {
    if (!isTerminal()) {
      // Heuristic: positions where opponent faces multiple of 4 are great.
      return (pile % (max_take + 1) == 0 && current_index != player) ? 1.0
                                                                     : 0.0;
    }
    int winner = 1 - current_index;
    return player == winner ? 1.0 : 0.0;
  }
};

template <typename Bot>
static int play_one(Bot& bot_p0, NimState s, std::mt19937& rng) {
  using namespace mathathon;
  RandomBot<NimState> opp;
  while (!s.isTerminal()) {
    TimeBudget budget(0.2);
    int action = (s.currentPlayer() == 0)
                     ? bot_p0.choose(s, 0, budget, rng)
                     : opp.choose(s, 1, budget, rng);
    s = s.apply(action);
  }
  return 1 - s.current_index;  // winner
}

static int self_test() {
  using namespace mathathon;
  std::mt19937 rng(42);

  int mm_wins = 0, mcts_wins = 0;
  for (int i = 0; i < 10; ++i) {
    NimState s;
    s.pile = 20;  // L-position for player to move; player 0 (bot) wants this
    MinimaxBot<NimState> mm(12);
    if (play_one(mm, s, rng) == 0) ++mm_wins;
  }
  for (int i = 0; i < 10; ++i) {
    NimState s;
    s.pile = 20;
    MCTSBot<NimState> mcts;
    mcts.simulations = 2000;
    if (play_one(mcts, s, rng) == 0) ++mcts_wins;
  }
  std::cout << "minimax wins (out of 10): " << mm_wins << "\n";
  std::cout << "mcts wins    (out of 10): " << mcts_wins << "\n";
  // Both should win consistently with pile=20 (random opponent in L-position).
  return (mm_wins >= 8 && mcts_wins >= 8) ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") return self_test();

  using namespace mathathon;
  run_per_move_loop<NimState>(
      [](const std::string& line) {
        NimState s;
        std::istringstream iss(line);
        iss >> s.pile;
        return s;
      },
      [](const NimState& s, int player, TimeBudget& b, std::mt19937& rng) {
        MinimaxBot<NimState> bot(8);
        return bot.choose(s, player, b, rng);
      },
      [](int action) { return std::to_string(action); }, 0.2, 1);
  return 0;
}
