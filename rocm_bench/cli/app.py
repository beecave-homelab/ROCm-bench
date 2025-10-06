"""Main CLI application for ROCm benchmark tool."""

import typer

from rocm_bench.cli.commands.run import app as run_app
from rocm_bench.cli.commands.status import app as status_app

app = typer.Typer(
    help="ROCm-bench: sample AMD GPU usage while running any "
    "command and persist JSON benchmarks."
)
app.add_typer(
    run_app,
    name="run",
    help="Run a command under GPU sampling and write a benchmark JSON.",
)
app.add_typer(
    status_app, name="status", help="Inspect latest benchmark JSON files."
)
