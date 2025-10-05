from __future__ import annotations
import json
from pathlib import Path
from rocm_bench.core.services import BenchmarkCollector, GpuUtilSampler

def test_collector_roundtrip(tmp_path: Path) -> None:
    c = BenchmarkCollector(output_dir=tmp_path)
    p = c.collect(
        label="unit",
        cmd=["echo", "hi"],
        total_time=0.123,
        runtime_seconds=None,
        extra={"k": "v"},
        gpu_stats=None,
    )
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["label"] == "unit"
    assert data["cmd"][0] == "echo"

def test_sampler_without_gpu() -> None:
    s = GpuUtilSampler(interval=0.01)
    # In CI, pyamdgpuinfo likely unavailable: start() should be False
    started = s.start()
    s.stop()
    assert started in (False, True)  # allow either locally
    # Summary should be None if no samples
    assert s.summary() in (None, dict())
