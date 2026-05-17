"""Replay a fixed Snaky Greed gamelog: bot vs scripted opponent moves.

Usage:
  python tools/greed_replay.py <bot> [--slot 0|1] [--grid path] [--verbose]

Default grid: tools/gamelogs/snaky_stRW_r0_grid.txt (from stRWategy vs tungtung R0).
Default opponent: stRWategy (P0) scripted moves; bot plays P1 unless --slot 0.

Example (play as tungtung slot, chase stRWategy line):
  python tools/greed_replay.py agent/snaky_greed.exe --slot 1 -v
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = 32
DIRS = {"u": (0, -1), "d": (0, 1), "l": (-1, 0), "r": (1, 0)}
STARTS = [(8, 8), (23, 23)]

# stRWategy (P0) vs tungtungsahuur (P1) — official gamelog round 0 board
DEFAULT_GRID = Path(__file__).resolve().parent / "gamelogs" / "snaky_stRW_r0_grid.txt"

# Recorded OUT moves (may be shorter than full game if log truncated)
SCRIPT_P0 = "r d r r u r d l l l d r d l u u u l u r r".split()
SCRIPT_P1 = "u r d d l l u l u r u l d d l l u r d".split()


def load_grid(path: Path) -> list[list[int]]:
    text = path.read_text(encoding="utf-8").split()
    vals = [int(x) for x in text[: GRID * GRID]]
    if len(vals) < GRID * GRID:
        raise SystemExit(f"grid file needs {GRID*GRID} digits, got {len(vals)}")
    return [[vals[y * GRID + x] for x in range(GRID)] for y in range(GRID)]


def bot_cmd(path: str) -> list[str]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.suffix.lower() == ".py":
        return [sys.executable, str(p)]
    return [str(p)]


def run_replay(bot: list[str], bot_slot: int, grid: list[list[int]],
               script: list[list[str]], verbose: bool) -> None:
    gl = " ".join(str(grid[y][x]) for y in range(GRID) for x in range(GRID))
    opp_slot = 1 - bot_slot
    opp_moves = script[opp_slot]

    proc = subprocess.Popen(
        bot, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    pos = [list(STARTS[0]), list(STARTS[1])]
    claimed = {tuple(STARTS[0]), tuple(STARTS[1])}
    score = [1, 1]
    alive = [True, True]
    ply = 0

    def send_bot():
        o = 1 - bot_slot
        if ply == 0:
            proc.stdin.write(gl + "\n")
        proc.stdin.write(
            f"{pos[bot_slot][0]} {pos[bot_slot][1]} "
            f"{pos[o][0]} {pos[o][1]}\n"
        )
        proc.stdin.flush()

    def recv_bot() -> str:
        line = proc.stdout.readline()
        return line.strip() if line else ""

    try:
        while ply < 4000 and alive[bot_slot]:
            send_bot()
            bot_mv = recv_bot()
            opp_mv = opp_moves[ply] if ply < len(opp_moves) else None

            if verbose:
                ref = opp_mv if opp_slot else bot_mv
                print(
                    f"ply {ply:2d}: bot={bot_mv:1s} opp={opp_mv or '-':1s} "
                    f"pos_bot={pos[bot_slot]} pos_opp={pos[opp_slot]}"
                )

            paths = [[], []]
            for p, mv in enumerate([None, None]):
                if p == bot_slot:
                    mv = bot_mv
                elif p == opp_slot:
                    mv = opp_mv
                if not alive[p] or mv not in DIRS:
                    continue
                dx, dy = DIRS[mv]
                ax, ay = pos[p][0] + dx, pos[p][1] + dy
                dist = grid[ay][ax] if 0 <= ax < GRID and 0 <= ay < GRID else 1
                for k in range(1, dist + 1):
                    paths[p].append((pos[p][0] + dx * k, pos[p][1] + dy * k))

            maxlen = max(len(paths[0]), len(paths[1]))
            done = [not (alive[p] and (p == bot_slot or opp_moves[ply:])) for p in (0, 1)]
            for p in (0, 1):
                if p == bot_slot and bot_mv not in DIRS:
                    alive[p] = False
                if p == opp_slot and (ply >= len(opp_moves) or opp_mv not in DIRS):
                    alive[p] = False

            for k in range(maxlen):
                cell = {}
                for p in (0, 1):
                    if not alive[p]:
                        continue
                    if k >= len(paths[p]):
                        continue
                    cell[p] = paths[p][k]
                deaths = set()
                for p, c in cell.items():
                    if not (0 <= c[0] < GRID and 0 <= c[1] < GRID) or c in claimed:
                        deaths.add(p)
                if len(cell) == 2 and cell.get(0) == cell.get(1):
                    deaths.update((0, 1))
                for p, c in cell.items():
                    if p in deaths:
                        alive[p] = False
                    else:
                        claimed.add(c)
                        pos[p] = list(c)
                        score[p] += 1

            if bot_slot == 0 and ply < len(SCRIPT_P0) and bot_mv != SCRIPT_P0[ply]:
                print(f"  ** opening diff R{ply}: bot={bot_mv} log={SCRIPT_P0[ply]}")
            if bot_slot == 1 and ply < len(SCRIPT_P1) and bot_mv != SCRIPT_P1[ply]:
                print(f"  ** opening diff R{ply}: bot={bot_mv} log={SCRIPT_P1[ply]}")

            ply += 1
            if not alive[bot_slot]:
                break
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=3)

    print(f"\nfinal score bot={score[bot_slot]} opp={score[opp_slot]} "
          f"alive={alive[bot_slot]} rounds={ply}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bot", help="bot .py or .exe path")
    ap.add_argument("--slot", type=int, default=1, choices=(0, 1),
                    help="0 = NW start (8,8), 1 = SE start (23,23)")
    ap.add_argument("--grid", type=str, default=str(DEFAULT_GRID))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    grid_path = Path(args.grid)
    if not grid_path.is_absolute():
        grid_path = ROOT / grid_path
    if not grid_path.exists():
        raise SystemExit(
            f"missing grid file {grid_path}\n"
            "Paste round-0 board digits (1024 ints) into that file."
        )

    grid = load_grid(grid_path)
    script = [SCRIPT_P0, SCRIPT_P1]
    print(f"bot={bot_cmd(args.bot)} slot={args.slot} grid={grid_path.name}")
    run_replay(bot_cmd(args.bot), args.slot, grid, script, args.verbose)


if __name__ == "__main__":
    main()
