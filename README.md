# ROCm-bench

A **Typer**-based CLI wrapper that samples AMD GPU utilisation with **pyamdgpuinfo** while any external command runs
(ASR pipelines, local LLMs, encoders — anything). It writes timestamped JSON benchmark artifacts with wall-clock runtime
and aggregated GPU stats (avg/max load %, avg/max VRAM MB).

> PoC focus: leverage `pyamdgpuinfo` to sample usage during *other* processes.

## Quickstart

```bash
# 1) Install (editable)
pip install -e ".[gpu]"  # adds pyamdgpuinfo optional dependency

# 2) Run any command under GPU sampling
rocm-bench run --label "asr_infer" --output-dir benchmarks -- echo "hello"
# or, a real workload using your GPU (example):
# rocm-bench run --label "llm" -- python your_llm_script.py --model my.gguf --prompt "hi"
```

Artifacts are stored in `benchmarks/` by default and look like:

```json
{
  "label": "asr_infer",
  "cmd": ["python", "my_asr.py", "--input", "x.wav"],
  "total_time_seconds": 12.34,
  "runtime_seconds": null,
  "gpu_stats": {
    "provider": "pyamdgpuinfo",
    "sample_interval_seconds": 0.5,
    "sample_count": 24,
    "avg_gpu_load_percent": 64.21,
    "max_gpu_load_percent": 97.88,
    "avg_vram_mb": 1023.12,
    "max_vram_mb": 2048.00
  },
  "extra": {"dataset": "dev"},
  "recorded_at": "2025-10-05T12:00:00+00:00"
}
```

## CLI

```bash
# Help
rocm-bench --help
rocm-bench run --help
rocm-bench status --help
```

Common usage:

```bash
# Sample GPU during a process and persist a JSON record
rocm-bench run --label "my-job" --output-dir bm --interval 0.25 --   python my_gpu_script.py --model small --input foo.wav

# See a short summary of latest records
rocm-bench status --dir bm --limit 5
```

## Notes

- If `pyamdgpuinfo` or compatible AMD GPU is not available, sampling no-ops and `gpu_stats` is omitted.
- Timestamps are timezone-aware based on `APP_TIMEZONE` (default: `"UTC"`).

## References

- pyamdgpuinfo (GitHub & PyPI) — GPU load / VRAM queries.  
- Typer — CLI framework.  
- `zoneinfo` — Python time zone support (PEP 615).

See project `pyproject.toml` for extras and scripts.

---

**Author:** elvee  
**Date:** 2025-10-05
