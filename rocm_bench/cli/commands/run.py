"""Commands for running external commands under AMD GPU sampling."""

from __future__ import annotations

from pathlib import Path

import typer

from rocm_bench.cli.common_options import interval_option, output_dir_option
from rocm_bench.core.services import run_command_and_collect

app = typer.Typer(help="Run an external command under AMD GPU sampling")


@app.command("exec")
def exec_(
    cmd: list[str] = typer.Argument(
        ..., help="Command and args to execute (use -- to separate)"
    ),
    label: str = typer.Option(..., "--label", "-l", help="Label for this benchmark"),
    output_dir: Path = output_dir_option,
    interval: float = interval_option,
    extra: list[str] | None = typer.Option(
        None, "--extra", "-e", help="Extra metadata key=val (repeatable)"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "Skip command execution and GPU sampling while still writing a "
            "benchmark record."
        ),
    ),
) -> None:
    """Examples of running external commands under AMD GPU sampling.

    Args:
        cmd: Command and arguments to execute (use -- to separate).
        label: Label for this benchmark.
        output_dir: Directory to write benchmark results.
        interval: Sampling interval in seconds.
        extra: Extra metadata key=val pairs (repeatable).
        dry_run: Skip command execution and GPU sampling while still writing
            a benchmark record.

    Raises:
        typer.Exit: Exits with the command's exit code.

    """
    extra_dict = {}
    if extra:
        for kv in extra:
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra_dict[k] = v
            else:
                typer.echo(f"[warn] ignoring extra '{kv}', expected key=val", err=True)

    exit_code, json_path = run_command_and_collect(
        cmd=cmd,
        label=label,
        interval=interval,
        output_dir=output_dir,
        extra=extra_dict,
        dry_run=dry_run,
    )
    typer.echo(f"[rocm-bench] wrote: {json_path}")
    raise typer.Exit(code=exit_code)
