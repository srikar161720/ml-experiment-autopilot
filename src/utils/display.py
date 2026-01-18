"""Rich console display utilities for the autopilot."""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.markdown import Markdown

from src import __version__

console = Console()


def print_header():
    """Print the application header."""
    header = Text()
    header.append("ML Experiment Autopilot", style="bold blue")
    header.append(f" v{__version__}", style="dim")

    console.print(Panel(
        header,
        subtitle="Autonomous ML experimentation powered by Gemini",
        border_style="blue",
    ))
    console.print()


def print_config(
    data_path: Path,
    target: str,
    task: str,
    max_iterations: int,
    time_budget: int,
    constraints: Optional[Path] = None,
    verbose: bool = False,
):
    """Print the configuration summary."""
    table = Table(title="Configuration", show_header=False, border_style="dim")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Dataset", str(data_path))
    table.add_row("Target Column", target)
    table.add_row("Task Type", task)
    table.add_row("Max Iterations", str(max_iterations))
    table.add_row("Time Budget", f"{time_budget}s ({time_budget // 60}m)")
    table.add_row("Constraints", str(constraints) if constraints else "None")
    table.add_row("Verbose", "Yes" if verbose else "No")

    console.print(table)
    console.print()


def print_phase(phase: str, description: str = ""):
    """Print a phase header."""
    console.print(Panel(
        f"[bold]{phase}[/bold]\n{description}" if description else f"[bold]{phase}[/bold]",
        border_style="green",
        padding=(0, 2),
    ))


def print_data_profile(profile: dict):
    """Print the data profile summary."""
    console.print("\n[bold cyan]Dataset Profile[/bold cyan]")

    # Basic info table
    info_table = Table(show_header=False, border_style="dim", box=None)
    info_table.add_column("", style="dim")
    info_table.add_column("")

    info_table.add_row("Rows", str(profile.get("n_rows", "?")))
    info_table.add_row("Columns", str(profile.get("n_columns", "?")))
    info_table.add_row("Numeric Features", str(len(profile.get("numeric_columns", []))))
    info_table.add_row("Categorical Features", str(len(profile.get("categorical_columns", []))))
    info_table.add_row("Target", profile.get("target_column", "?"))
    info_table.add_row("Target Type", profile.get("target_type", "?"))

    console.print(info_table)

    # Missing values
    missing = profile.get("missing_values", {})
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    if missing_cols:
        console.print(f"\n[yellow]Missing Values:[/yellow] {len(missing_cols)} columns")
        for col, count in list(missing_cols.items())[:5]:
            pct = profile.get("missing_percentages", {}).get(col, 0)
            console.print(f"  • {col}: {count} ({pct}%)")
        if len(missing_cols) > 5:
            console.print(f"  • ... and {len(missing_cols) - 5} more")
    else:
        console.print("\n[green]No missing values[/green]")

    console.print()


def print_iteration(iteration: int, total: int, experiment_name: str):
    """Print an iteration header."""
    console.print(Panel(
        f"[bold]Iteration {iteration}/{total}[/bold]\nExperiment: {experiment_name}",
        border_style="blue",
        padding=(0, 2),
    ))


def print_results(metrics: dict, success: bool, execution_time: float):
    """Print experiment results."""
    if not success:
        console.print("[red]Experiment failed[/red]")
        return

    console.print("\n[bold green]Results[/bold green]")

    table = Table(show_header=True, border_style="dim")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for metric, value in metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.6f}")
        else:
            table.add_row(metric, str(value))

    table.add_row("Execution Time", f"{execution_time:.2f}s", style="dim")

    console.print(table)


def print_reasoning(
    iteration: int,
    reasoning: str,
    hypothesis: str,
    turn_count: int = 0,
):
    """Print Gemini's reasoning with thought signature indicator."""
    content = []

    if turn_count > 0:
        content.append(f"[dim]Thought Signature Active | Context: {turn_count} turns[/dim]\n")

    if hypothesis:
        content.append(f"[bold]Hypothesis:[/bold] {hypothesis}\n")

    if reasoning:
        content.append(f"[bold]Reasoning:[/bold]\n{reasoning}")

    console.print(Panel(
        "\n".join(content),
        title=f"ITERATION {iteration} - GEMINI'S REASONING",
        border_style="magenta",
        padding=(1, 2),
    ))


def print_best_result(experiment_name: str, metric_name: str, metric_value: float):
    """Print the best result found so far."""
    console.print(Panel(
        f"[bold green]Best: {experiment_name}[/bold green]\n"
        f"{metric_name}: {metric_value:.6f}",
        border_style="green",
        padding=(0, 2),
    ))


def print_termination(reason: str):
    """Print termination message."""
    console.print(Panel(
        f"[bold]Experiment Loop Terminated[/bold]\n{reason}",
        border_style="yellow",
        padding=(0, 2),
    ))


def print_summary(state_summary: dict):
    """Print final summary."""
    console.print("\n[bold cyan]Experiment Summary[/bold cyan]")

    table = Table(show_header=False, border_style="dim", box=None)
    table.add_column("", style="dim")
    table.add_column("")

    table.add_row("Session ID", state_summary.get("session_id", "?"))
    table.add_row("Total Iterations", str(state_summary.get("current_iteration", 0)))
    table.add_row("Successful Experiments", str(state_summary.get("successful_experiments", 0)))
    table.add_row("Total Time", f"{state_summary.get('elapsed_time', 0):.1f}s")

    if state_summary.get("best_metric") is not None:
        table.add_row("Best Metric", f"{state_summary['best_metric']:.6f}")
        table.add_row("Best Experiment", state_summary.get("best_experiment", "?"))

    console.print(table)
    console.print()


def print_error(message: str, details: Optional[str] = None):
    """Print an error message."""
    console.print(f"\n[bold red]Error:[/bold red] {message}")
    if details:
        console.print(f"[dim]{details}[/dim]")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]Warning:[/yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[blue]Info:[/blue] {message}")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]Success:[/green] {message}")


def create_progress() -> Progress:
    """Create a progress display for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )
