"""MolNex command-line interface (the ``molnex`` console script).

Exposes a Typer :data:`app` whose entry point is declared in
``pyproject.toml`` as ``molnex = "molix.cli:app"``. The first command is
``molnex check`` — an environment self-check (see :mod:`molix.cli.doctor`).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from molix.cli.doctor import CheckResult, Doctor, Status

app = typer.Typer(
    name="molnex",
    help="MolNex command-line interface.",
    add_completion=False,
    no_args_is_help=True,
)

_console = Console()


@app.callback()
def _root() -> None:
    """MolNex command-line interface."""
    # Presence of a callback keeps ``check`` a named subcommand even while it
    # is the only command (otherwise Typer collapses to a single-command CLI).


def _render_table(results: tuple[CheckResult, ...]) -> None:
    """Print a rich table of check results to stdout."""
    table = Table(title="MolNex environment check", title_style="bold", show_lines=False)
    table.add_column("", justify="center", no_wrap=True)
    table.add_column("Check", style="bold")
    table.add_column("Detail")
    for result in results:
        symbol = f"[{result.status.color}]{result.status.symbol}[/]"
        detail = result.detail
        if result.hint and result.status is not Status.OK:
            detail += f"\n[dim]→ {result.hint}[/]"
        table.add_row(symbol, result.name, detail)
    _console.print(table)


@app.command()
def check(
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero on warnings as well as failures."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit results as JSON instead of a table."),
) -> None:
    """Check that PyTorch, CUDA, native ops, and dependencies are in place."""
    results = Doctor().run()

    if json_out:
        import json

        payload = [
            {
                "name": r.name,
                "status": r.status.value,
                "detail": r.detail,
                "hint": r.hint,
            }
            for r in results
        ]
        # Plain stdout, no rich styling: ``--json`` must stay machine-readable
        # (``_console.print_json`` injects ANSI colour codes that break parsers).
        typer.echo(json.dumps(payload, indent=2))
    else:
        _render_table(results)

    worst = Doctor.worst_status(results)
    if worst is Status.FAIL or (strict and worst is Status.WARN):
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (also usable as ``python -m molix.cli``)."""
    app()


__all__ = ["app", "main"]
