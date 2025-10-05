from __future__ import annotations
from pathlib import Path
import typer

output_dir_option = typer.Option(
    Path("benchmarks"),
    "--output-dir",
    "-o",
    help="Directory where benchmark JSON files are written.",
)

interval_option = typer.Option(
    0.5,
    "--interval",
    "-i",
    help="GPU sampling interval in seconds.",
)
