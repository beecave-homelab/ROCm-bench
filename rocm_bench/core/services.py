"""Core services for benchmark collection and GPU utilization sampling."""

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
from typing import Any
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
    """Result of running a benchmark command.

    Attributes:
        exit_code: Exit code of the executed command.
        total_time: Total execution time in seconds.
        gpu_stats: GPU utilization statistics, if available.
        json_path: Path to the generated benchmark JSON file.

    """

    exit_code: int
    total_time: float
    gpu_stats: dict[str, Any] | None
    json_path: Path


class BenchmarkCollector:
    """Persist benchmark records to JSON files."""

    def __init__(self, output_dir: Path | str = "benchmarks") -> None:
        """Initialize benchmark collector.

        Args:
            output_dir: Directory to store benchmark JSON files.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        *,
        label: str,
        cmd: list[str],
        total_time: float,
        runtime_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
        gpu_stats: dict[str, Any] | None = None,
    ) -> Path:
        """Collect benchmark results and write to JSON file.

        Args:
            label: Label for this benchmark.
            cmd: Command that was executed.
            total_time: Total execution time in seconds.
            runtime_seconds: Runtime seconds if different from total_time.
            extra: Additional metadata key-value pairs.
            gpu_stats: GPU utilization statistics.

        Returns:
            Path to the generated JSON file.

        """
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
    """Sample AMD GPU utilisation in the background at a fixed interval."""

    def __init__(self, interval: float = 0.5) -> None:
        """Initialize GPU utilization sampler.

        Args:
            interval: Sampling interval in seconds.

        """
        self._interval = interval
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._gpu = None

    def start(self) -> bool:
        """Start GPU utilization sampling in background.

        Returns:
            True if sampling started successfully, False otherwise.

        """
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
        """Stop GPU utilization sampling."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
        # reset handles
        self._stop_event = None
        self._thread = None

    def summary(self) -> dict[str, Any] | None:
        """Get summary of GPU utilization statistics.

        Returns:
            Dictionary with GPU statistics if samples available, None otherwise.

        """
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
            "avg_vram_mb": round((sum(vrams) / count) / (1024**2), 2),
            "max_vram_mb": round(max(vrams) / (1024**2), 2),
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
    extra: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> tuple[int, Path]:
    """Run an external command while sampling GPU, then write a benchmark JSON.

    Args:
        cmd: Command and arguments to execute.
        label: Label for this benchmark.
        interval: Sampling interval in seconds.
        output_dir: Directory to write benchmark results.
        extra: Additional metadata key-value pairs.
        dry_run: Skip command execution and GPU sampling while still writing a
            benchmark record.

    Returns:
        Tuple of (exit_code, json_path).

    """
    collector = BenchmarkCollector(output_dir=output_dir)

    if dry_run:
        metadata = dict(extra or {})
        metadata["dry_run"] = True
        out = collector.collect(
            label=label,
            cmd=cmd,
            total_time=0.0,
            runtime_seconds=0.0,
            extra=metadata,
            gpu_stats={},
        )
        logger.info("Dry-run benchmark written: %s", out)
        return 0, out

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

    out = collector.collect(
        label=label,
        cmd=cmd,
        total_time=total_time,
        runtime_seconds=None,
        extra=extra,
        gpu_stats=gpu_stats,
    )
    return code, out
