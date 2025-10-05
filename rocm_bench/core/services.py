from __future__ import annotations
import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rocm_bench.core.config import APP_TIMEZONE

logger = logging.getLogger(__name__)

# Optional dependency for AMD GPU telemetry
try:  # pragma: no cover
    import pyamdgpuinfo  # type: ignore
except Exception:  # pragma: no cover
    pyamdgpuinfo = None  # type: ignore

_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    s = _SAFE_PATTERN.sub("-", value).strip("-._")
    return s or "benchmark"


@dataclass
class RunResult:
    exit_code: int
    total_time: float
    gpu_stats: Optional[dict[str, Any]]
    json_path: Path


class BenchmarkCollector:
    """Persist benchmark records to JSON files."""

    def __init__(self, output_dir: Path | str = "benchmarks") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        *,
        label: str,
        cmd: list[str],
        total_time: float,
        runtime_seconds: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
        gpu_stats: Optional[dict[str, Any]] = None,
    ) -> Path:
        try:
            tz = ZoneInfo(APP_TIMEZONE)
        except ZoneInfoNotFoundError:
            tz = ZoneInfo("UTC")

        record = {
            "label": label,
            "cmd": cmd,
            "total_time_seconds": total_time,
            "runtime_seconds": runtime_seconds,
            "gpu_stats": gpu_stats or {},
            "extra": extra or {},
            "recorded_at": datetime.now(tz).isoformat(),
        }

        slug = _slugify(label or (cmd[0] if cmd else "benchmark"))
        ts = datetime.now(tz).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{slug}_{ts}.json"
        out = self.output_dir / filename
        out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        logger.info("Benchmark written: %s", out)
        return out


class GpuUtilSampler:
    """Sample AMD GPU utilisation (load %, VRAM bytes) at a fixed interval in the background."""

    def __init__(self, interval: float = 0.5) -> None:
        self._interval = interval
        self._samples: list[tuple[float, float]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._gpu = None

    def start(self) -> bool:
        if pyamdgpuinfo is None:  # pragma: no cover (depends on env)
            logger.warning("pyamdgpuinfo not available; GPU sampling disabled.")
            return False
        try:  # pragma: no cover (hardware specific)
            self._gpu = pyamdgpuinfo.get_gpu(0)
        except Exception:
            self._gpu = None
            logger.warning("AMD GPU not detected; sampling disabled.")
            return False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
        # reset handles
        self._stop_event = None
        self._thread = None

    def summary(self) -> Optional[dict[str, Any]]:
        if not self._samples:
            return None
        loads = [s[0] for s in self._samples]
        vrams = [s[1] for s in self._samples]
        count = len(self._samples)
        return {
            "provider": "pyamdgpuinfo",
            "sample_interval_seconds": self._interval,
            "sample_count": count,
            "avg_gpu_load_percent": round((sum(loads) / count) * 100, 2),
            "max_gpu_load_percent": round(max(loads) * 100, 2),
            "avg_vram_mb": round((sum(vrams) / count) / (1024 ** 2), 2),
            "max_vram_mb": round(max(vrams) / (1024 ** 2), 2),
        }

    def _run_loop(self) -> None:
        if self._gpu is None or self._stop_event is None:
            return
        while not self._stop_event.is_set():  # pragma: no cover (hardware specific)
            try:
                load = float(self._gpu.query_load())
                vram = float(self._gpu.query_vram_usage())
            except Exception:
                break
            self._samples.append((load, vram))
            self._stop_event.wait(self._interval)


def run_command_and_collect(
    *,
    cmd: list[str],
    label: str,
    interval: float,
    output_dir: Path,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[int, Path]:
    """Run an external command while sampling GPU, then write a benchmark JSON.

    Returns (exit_code, json_path).
    """
    sampler = GpuUtilSampler(interval=interval)
    started = sampler.start()
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, check=False)
        code = proc.returncode
    finally:
        sampler.stop()
    t1 = time.time()

    total_time = t1 - t0
    gpu_stats = sampler.summary() if started else None

    collector = BenchmarkCollector(output_dir=output_dir)
    out = collector.collect(
        label=label,
        cmd=cmd,
        total_time=total_time,
        runtime_seconds=None,
        extra=extra,
        gpu_stats=gpu_stats,
    )
    return code, out
