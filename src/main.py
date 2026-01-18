"""CLI entry point for ML Experiment Autopilot."""

from pathlib import Path
from typing import Optional
from enum import Enum

import typer
from rich.console import Console

from src import __version__
from src.config import get_config, ensure_directories

app = typer.Typer(
    name="autopilot",
    help="ML Experiment Autopilot - Autonomous ML experimentation powered by Gemini",
    add_completion=False,
)

console = Console()


class TaskType(str, Enum):
    """Type of ML task."""

    classification = "classification"
    regression = "regression"


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"ML Experiment Autopilot v{__version__}")
        raise typer.Exit()


@app.command()
def run(
    data: Path = typer.Option(
        ...,
        "--data",
        "-d",
        help="Path to dataset (CSV or Parquet file)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Target column name for prediction",
    ),
    task: TaskType = typer.Option(
        ...,
        "--task",
        help="Type of ML task: classification or regression",
    ),
    constraints: Optional[Path] = typer.Option(
        None,
        "--constraints",
        "-c",
        help="Path to constraints file (Markdown)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    max_iterations: int = typer.Option(
        20,
        "--max-iterations",
        "-n",
        help="Maximum number of experiment iterations",
        min=1,
        max=100,
    ),
    time_budget: int = typer.Option(
        3600,
        "--time-budget",
        help="Time budget in seconds",
        min=60,
        max=86400,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for results (auto-generated if not specified)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed Gemini reasoning",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Resume from saved state file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Run the ML Experiment Autopilot on a dataset.

    The autopilot will:
    1. Profile your dataset
    2. Design and run ML experiments
    3. Analyze results and generate hypotheses
    4. Iterate to improve performance
    5. Generate a final report

    Example:
        autopilot run --data train.csv --target SalePrice --task regression
    """
    from src.utils.display import print_header, print_config
    from src.orchestration.controller import ExperimentController

    # Ensure directories exist
    ensure_directories()

    # Get configuration
    config = get_config(verbose=verbose)

    # Print header
    print_header()

    # Print configuration summary
    print_config(
        data_path=data,
        target=target,
        task=task.value,
        max_iterations=max_iterations,
        time_budget=time_budget,
        constraints=constraints,
        verbose=verbose,
    )

    # Read constraints if provided
    constraints_text = None
    if constraints:
        constraints_text = constraints.read_text()

    # Create and run the experiment controller
    controller = ExperimentController(
        data_path=data,
        target_column=target,
        task_type=task.value,
        constraints=constraints_text,
        max_iterations=max_iterations,
        time_budget=time_budget,
        output_dir=output_dir,
        verbose=verbose,
        resume_path=resume,
    )

    try:
        controller.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. Saving state...[/yellow]")
        controller.save_state()
        console.print("[green]State saved. You can resume with --resume[/green]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """ML Experiment Autopilot - Autonomous ML experimentation powered by Gemini."""
    pass


if __name__ == "__main__":
    app()
