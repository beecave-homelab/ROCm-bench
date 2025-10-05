from __future__ import annotations
from pathlib import Path
from rocm_bench.utils.helpers import ensure_dir

def test_ensure_dir(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b"
    ensure_dir(p)
    assert p.exists()
    assert p.is_dir()
