"""Thin CLI around independent operator profiling primitives."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from molix.profiler.nsight import NsightCompute, NsightSystems
from molix.profiler.ptxas import demangle, format_resources, parse_ptxas, resources_as_dicts

app = typer.Typer(help="Profile an explicitly prepared operator workload.", no_args_is_help=True)


@app.command()
def doctor():
    """Print installed NVIDIA profiler paths and versions."""
    typer.echo(
        json.dumps(
            {"nsys": NsightSystems().inspect(), "ncu": NsightCompute(kernel=".").inspect()},
            indent=2,
        )
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def nsys(
    ctx: typer.Context,
    output: Path = typer.Option(..., "--output"),
    executable: str = typer.Option("nsys", "--executable"),
    timeout: float = typer.Option(600, "--timeout"),
):
    """Capture a timeline: nsys --output DIR -- python workload.py --capture."""
    tool = NsightSystems(executable)
    record = tool.run(list(ctx.args), output, timeout=timeout)
    statistics = tool.stats(Path(record["report"]), output / "stats.csv")
    (output / "stats.json").write_text(json.dumps(statistics, indent=2) + "\n")
    typer.echo(json.dumps(record, indent=2))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ncu(
    ctx: typer.Context,
    output: Path = typer.Option(..., "--output"),
    kernel: str = typer.Option(..., "--kernel"),
    executable: str = typer.Option("ncu", "--executable"),
    launches: int = typer.Option(4, "--launches"),
    timeout: float = typer.Option(600, "--timeout"),
):
    """Capture selected kernels: ncu --kernel REGEX --output DIR -- COMMAND."""
    tool = NsightCompute(executable, kernel=kernel, launches=launches)
    record = tool.run(list(ctx.args), output, timeout=timeout)
    metrics = tool.metrics(Path(record["report"]), output / "metrics.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    typer.echo(json.dumps(record, indent=2))


@app.command()
def summarize(
    stats: Path = typer.Argument(..., help="stats.json or stats.csv of a `profile nsys` capture"),
    iterations: int = typer.Option(1, "--iterations", help="workload iterations in the window"),
    top: int = typer.Option(20, "--top"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Kernels of a capture by time, per workload iteration, template arguments stripped."""
    text = stats.read_text()
    tables = json.loads(text) if stats.suffix == ".json" else NsightSystems.parse_stats(text)
    costs = NsightSystems.kernel_summary(tables, iterations=iterations, top=top)
    if as_json:
        typer.echo(json.dumps([cost.__dict__ for cost in costs], indent=2))
    else:
        typer.echo(NsightSystems.format_summary(costs))


@app.command()
def ptxas(
    log: Path = typer.Argument(..., help="build log of a -Xptxas=-v compilation"),
    pattern: str = typer.Option(None, "--pattern", help="regex on the (mangled) entry name"),
    readable: bool = typer.Option(False, "--demangle", help="through c++filt when installed"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Registers, spills and shared memory per kernel entry from a build log."""
    kernels = parse_ptxas(log.read_text())
    if readable:
        names = demangle([k.function for k in kernels])
        kernels = [k.__class__(**{**k.__dict__, "function": n}) for k, n in zip(kernels, names)]
    if as_json:
        typer.echo(json.dumps(resources_as_dicts(kernels), indent=2))
    else:
        typer.echo(format_resources(kernels, pattern=pattern))
