# C++ Skeleton

Mirror of the Python kit for problems that won't run fast enough in Python.
Single-header library: `mathathon.hpp` (TimeBudget, Random/Greedy/Minimax/MCTS bots, stdio loop).

## Build

```bash
# Linux / macOS / WSL
g++ -std=c++17 -O2 nim_example.cpp -o nim

# Windows (MSVC)
cl /std:c++17 /O2 nim_example.cpp /Fe:nim.exe

# Windows (MinGW)
g++ -std=c++17 -O2 nim_example.cpp -o nim.exe
```

**For submissions, always link statically** so the binary has no MinGW
runtime-DLL dependency (the judge machine won't have `libstdc++-6.dll` etc.):

```bash
g++ -std=c++17 -O2 -static nim_example.cpp -o nim.exe
```

Local MinGW: WinLibs g++ 16.1.0 at
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*\mingw64\bin`
(on PATH after a shell restart). A non-static build only runs from a shell
that has that `bin\` on PATH — `-static` avoids the issue entirely.

## Run

```bash
# Local sanity check (no stdin):
./nim --selftest

# Platform-style: read state per line, write action per line:
echo 21 | ./nim
```

## Adapter contract

```cpp
struct MyState : mathathon::GameStateBase<MyState, ActionType> {
  int currentPlayer() const;
  std::vector<ActionType> legalActions() const;
  MyState apply(ActionType action) const;
  bool isTerminal() const;
  double score(int player) const;
  static constexpr int numPlayers = 2;
};
```

## Notes / Trade-offs vs Python kit

- C++ MCTS uses a flat `NodeStore` with parent indices — full backprop, no
  pointer-chasing across heap allocations. Verified against `--selftest`.
- For maximum strength on heavy turn-based games, port the Python
  `MinimaxBotTT` (transposition table + iterative deepening + PV move
  ordering). The single-header `MinimaxBot` here is plain α-β; the structure
  is the same so adding TT is a 50-line change.
- No I/O parser presets in C++ — write a hand-rolled parser per problem.
  JSON: drop in [nlohmann/json](https://github.com/nlohmann/json) header.
