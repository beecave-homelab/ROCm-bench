from __future__ import annotations
import json
import subprocess
from pathlib import Path

def test_cli_run_exec(tmp_path: Path) -> None:
    outdir = tmp_path / "bm"
    cmd = [
        "python",
        "-m",
        "rocm_bench",
        "run",
        "exec",
        "--label",
        "echo-test",
        "--output-dir",
        str(outdir),
        "--",
        "python",
        "-c",
        "print('hello')",
    ]
    res = subprocess.run(cmd, check=False)
    assert res.returncode == 0
    # exactly one JSON written
    files = list(outdir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["label"] == "echo-test"
    assert data["cmd"][-1] == "print('hello')"
    assert "total_time_seconds" in data
