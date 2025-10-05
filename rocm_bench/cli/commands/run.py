from __future__ import annotations
import shlex
from pathlib import Path
from typing import List, Optional

import typer

from rocm_bench.cli.common_options import output_dir_option, interval_option
from rocm_bench.core.services import run_command_and_collect

app = typer.Typer(help="Run an external command under AMD GPU sampling")

@app.command("exec")
def exec_(
    cmd: List[str] = typer.Argument(..., help="Command and args to execute (use -- to separate)"),
    label: str = typer.Option(..., "--label", "-l", help="Label for this benchmark"),
    output_dir: Path = output_dir_option,
    interval: float = interval_option,
    extra: Optional[List[str]] = typer.Option(None, "--extra", "-e", help="Extra metadata key=val (repeatable)"),
) -> None:
    """Example:
        rocm-bench run exec -l asr -- echo hello
        rocm-bench run exec -l llm -- python your_llm.py --model foo --prompt hi
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
    )
    typer.echo(f"[rocm-bench] wrote: {json_path}")
    raise typer.Exit(code=exit_code)
