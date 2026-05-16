# Mathathon Starter Kit

Reusable bot-development kit for the ETH Mathathon (May 16–17 2026) and similar
algorithmic-game competitions. Goal: turn unfamiliar rules into a stable,
testable bot — and a working stdio submission — as fast as possible.

## What's Included

### Core engine — `mathathon_kit/`

| Module | Purpose |
|---|---|
| `core.py` | `GameState`, `SimultaneousState`, `Simulator`, `SimultaneousSimulator`, `TimeBudget` |
| `bots.py` | Random / Greedy / Minimax (α-β) / **`MinimaxBotTT` (TT + iterative deepening + PV ordering)** / BeamSearch / MCTS (opponent-aware) / MetaBot |
| `strategy.py` | RegretMatching · FictitiousPlay · ISMCTS · TitForTat · Grim · Pavlov · ε-Greedy bandit |
| `nash.py` | **`solve_zero_sum`** (LP via scipy, fictitious-play fallback) · `NashMatrixBot` |
| `tournament.py` | **N-player** round-robin — win-rate, avg score, multi-Elo · `significance()` (auto baseline = 1/seats) · `error_report()` (error messages without `keep_results`) |
| `parallel.py` | `ParallelRoundRobin` — drop-in process-pool round-robin (~3× speedup) |
| `tuning.py` | `GridTuner` for hyperparameter sweeps |
| `validate.py` | **`validate_adapter`** (mutation / hashability / legality / termination checks) · **`benchmark_bot`** (per-move latency: mean / p95 / worst / overruns) |
| `utils.py` | Iterative deepening, evaluator normalization / composition, legality guard |
| `io.py` | **stdio submission loops** — `run_per_move_loop` (one-line) · `run_protocol_loop` (handshake / multi-line / sentinel) · crash-safe fallback · transcript record + replay |
| `replay.py` | JSONL move-by-move logs |

### Examples — `examples/`

| File | Game type | Players | Mode |
|---|---|---|---|
| `nim_game.py` | Combinatorial subtraction game | 2 | turn-based |
| `iterated_pd.py` | Iterated Prisoner's Dilemma | 2 | simultaneous |
| `colonel_blotto.py` | Allocate forces across battlefields | 2 | simultaneous |
| `beauty_contest.py` | Guess 2/3 of average | 4 | simultaneous |
| `auction.py` | First-price sealed-bid | 3 | simultaneous |
| `grid_pursuit.py` | Pursuit-evasion on a grid | 2 | turn-based |
| `kuhn_poker.py` | Kuhn poker (hidden info, ISMCTS smoke test) | 2 | turn-based + hidden info |
| `platform_submission_nim.py` | stdio submission template | 1 | live judge |
| `template_adapter.py` | **copy-me scaffold** — blank adapter + opening checklist | 2 | turn-based |
| `figgie.py` | Simplified Jane Street Figgie — full-pipeline rehearsal | 4 | simultaneous + hidden info |
| `figgie_submission.py` | Figgie stdio submission (bundles to one file) | 1 | live judge |

### Tooling — `tools/`

| File | Purpose |
|---|---|
| `bundle.py` | **Single-file amalgamator** — inlines `mathathon_kit/` + `examples/` into one self-contained `.py` for judges that accept only one file. |

### C++ skeleton — `cpp/`

Single-header `mathathon.hpp` with the same TimeBudget / RandomBot / GreedyBot /
MinimaxBot / MCTSBot interfaces and a `run_per_move_loop` helper, plus a
working Nim example. Use this when Python is too slow.

## Quick Start

```bash
# Run any example
python examples/nim_game.py
python examples/iterated_pd.py
python examples/beauty_contest.py

# Submit a Python bot to the platform (stdio):
echo 21 | python examples/platform_submission_nim.py

# Bundle it into ONE self-contained file (for single-file judges):
python tools/bundle.py examples/platform_submission_nim.py -o submission.py
echo 21 | python submission.py        # verify from outside the repo

# Run the test suite
python -m pytest tests/ -q

# Build & run the C++ skeleton
cd cpp && g++ -std=c++17 -O2 nim_example.cpp -o nim && ./nim --selftest
```

## Adapter Contract

For a turn-based game implement `GameState`:

```python
class MyState:
    players = (0, 1)

    @property
    def current_player(self): ...
    def legal_actions(self, player=None): ...
    def apply(self, action) -> "MyState": ...
    def is_terminal(self) -> bool: ...
    def score(self, player) -> float: ...
```

For a simultaneous-move game implement `SimultaneousState` (see
`examples/iterated_pd.py`):

```python
class MyState:
    players = (0, 1)

    def active_players(self): ...
    def legal_actions(self, player): ...
    def apply_joint(self, actions: dict) -> "MyState": ...
    def is_terminal(self) -> bool: ...
    def score(self, player) -> float: ...
```

Both interfaces work with `RoundRobin` (set `simultaneous=True` for the
joint-move version).

## Competition-Day Playbook

1. **First 30 min.** Read the rules. Copy `examples/template_adapter.py` and
   fill in its TODO markers (it ships with an opening checklist), then run
   `validate_adapter(make_state(0))` — it catches the silent adapter bugs
   (mutation, unhashable state, illegal moves, non-termination) before they
   cost you games. Smoke-test with `RandomBot` self-play. Submit immediately
   if the platform allows early submissions — the legality-guarded random bot
   already won't crash.
2. **Next 60 min.** Write a domain heuristic. Run `GreedyBot` against
   `RandomBot` round-robin to confirm the heuristic is signed correctly.
3. **Choose the main engine:**
   - **Turn-based, 2-player, deterministic, repeating positions** → `MinimaxBotTT` (5–20× faster than plain minimax).
   - **Turn-based, 2-player, no repetition / unbounded depth** → `IterativeDeepeningMinimax` or `MinimaxBot`.
   - **Long horizon / large branching / stochastic** → `MCTSBot`.
   - **Repeated matrix-style** → `RegretMatchingBot` or `FictitiousPlayBot`.
   - **Pure 2-player zero-sum matrix** → `solve_zero_sum(payoff_matrix)` → `NashMatrixBot` (unexploitable mixed strategy).
   - **Hidden information** → `ISMCTSBot` (give it a `determinize()` sampler).
   - **Simultaneous one-shot** → enumerate-then-best-respond + `RegretMatchingBot` over rounds.
4. **Tune.** Use `GridTuner` over your engine's parameters. Switch to
   `ParallelRoundRobin` when sweeps get slow.
5. **Stress test.** Run `keep_results=True` round-robins against the random
   bot and your earlier versions; inspect any games with errors > 0. Use
   `report.significance()` to confirm a tweak's win-rate gain is real and not
   round-robin noise before you keep it. Run `benchmark_bot` to confirm your
   p95/worst move time stays under the judge's limit — one overrun forfeits a
   game.
6. **Submit.** Wrap your bot with `run_per_move_loop` (one line in/out) or
   `run_protocol_loop` (handshake / multi-line states / sentinel lines — build
   the reader from `read_line` / `read_until` / `count_prefixed_reader`).
   Always pass `fallback=random_legal_fallback(parse_state)` so a crashing
   engine emits a legal move instead of forfeiting. If the wire format
   surprises you, wrap stdin with `record_stdin(...)` and debug offline with
   `replay_transcript`. If the judge accepts only one file, run
   `python tools/bundle.py your_bot.py -o submission.py` and verify the bundle
   from a directory *outside* the repo before uploading.

## High-Win-Rate Rules

- **Always return a legal action.** `with_legality_guard(bot)` and the
  simulator both enforce this defensively, but bake it into your bot too.
- **The evaluator is your bot.** Greedy, Minimax, BeamSearch, and MCTS
  rollouts all bottleneck on it. Spend most of your time here.
- **Optimize for the round-robin average**, not just one opponent.
- **Keep a `MetaBot`** stack — fall back from a clever strategy to a robust
  one if the clever one is exploitable or unstable.
- **Normalize the evaluator** to roughly `[-1, 1]` so MCTS' UCB constant
  (`exploration=1.4`) stays calibrated. Use `normalize_evaluator(...)`.
- **Log losing games and inspect the first bad move**, not only the score.

## Layout

```
mathathon_kit/   # core engine
examples/        # one adapter per archetype
tools/           # bundle.py — single-file submission amalgamator
cpp/             # C++17 single-header mirror
tests/           # pytest suite (58 tests, all green)
pyproject.toml
```

## Test status

```
$ python -m pytest tests/ -q
..........................................................                  [100%]
58 passed
```
