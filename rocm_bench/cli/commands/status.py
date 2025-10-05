from __future__ import annotations
import json
from pathlib import Path
import typer

app = typer.Typer(help="Summarize recent benchmark JSON files.")

@app.command("list")
def list_records(
    dir: Path = typer.Option("benchmarks", "--dir", help="Directory containing benchmark JSON files"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max files to show (newest first)"),
) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    files = sorted(dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    if not files:
        typer.echo("[status] No records found.")
        raise typer.Exit(0)

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            typer.echo(f"- {f.name} (failed to parse: {e})")
            continue
        label = data.get("label", "?")
        total = data.get("total_time_seconds")
        gpu = data.get("gpu_stats") or {}
        avg = gpu.get("avg_gpu_load_percent")
        mx = gpu.get("max_gpu_load_percent")
        typer.echo(f"- {f.name} | label={label} total={total:.2f}s avg={avg}% max={mx}%")
