"""The single-file bundler must produce a submission that runs with NO access
to the repo -- that is the whole point of it."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_runs_self_contained(tmp_path):
    out = tmp_path / "submission.py"

    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "bundle.py"),
            str(ROOT / "examples" / "platform_submission_nim.py"),
            "-o",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    assert out.exists()

    # Run from tmp_path (no mathathon_kit on disk there) and with PYTHONPATH
    # stripped, so success proves the bundle carries everything it needs.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(out)],
        cwd=tmp_path,
        input="21\n8\n",
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    moves = proc.stdout.split()
    # One legal Nim move (take 1..3) per input line -- proves the inlined
    # mathathon_kit + examples imports all resolved inside the single file.
    assert len(moves) == 2
    assert all(m in {"1", "2", "3"} for m in moves)
    # 21 stones is a winning position: optimal play leaves a multiple of 4.
    assert moves[0] == "1"
