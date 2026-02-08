"""Visualization generator - Creates plots from experiment results.

This component generates matplotlib visualizations at the end of an
autopilot run. It does not use Gemini -- it is pure data processing.
All plot methods are individually wrapped in try/except to ensure
plot generation never crashes the main run.
"""

import matplotlib
matplotlib.use('Agg')  # Headless rendering -- must be before pyplot import

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.orchestration.state import ExperimentState
from src.utils.display import print_warning


class VisualizationGenerator:
    """Generates matplotlib visualizations from experiment state.

    Produces:
    - Metric progression line chart (primary metric vs iteration)
    - Model comparison bar chart (best metric per model type)
    - Improvement over baseline bar chart (baseline vs best)

    Usage:
        viz = VisualizationGenerator()
        plot_paths = viz.generate(state=state, output_dir=output_dir)
    """

    def __init__(self):
        """Initialize the visualization generator with clean defaults."""
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'figure.dpi': 100,
            'savefig.bbox': 'tight',
            'savefig.dpi': 150,
        })

    def generate(
        self,
        state: ExperimentState,
        output_dir: Path,
    ) -> list[Path]:
        """Generate all visualizations.

        Args:
            state: Complete ExperimentState after all iterations.
            output_dir: Base output directory (plots saved to output_dir/plots/).

        Returns:
            List of paths to generated plot PNG files.
        """
        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = []

        # 1. Metric progression line chart
        path = self._plot_metric_progression(state, plots_dir)
        if path:
            plot_paths.append(path)

        # 2. Model comparison bar chart
        path = self._plot_model_comparison(state, plots_dir)
        if path:
            plot_paths.append(path)

        # 3. Improvement over baseline
        path = self._plot_improvement_over_baseline(state, plots_dir)
        if path:
            plot_paths.append(path)

        return plot_paths

    def _plot_metric_progression(
        self,
        state: ExperimentState,
        plots_dir: Path,
    ) -> Optional[Path]:
        """Line plot of primary metric value across iterations.

        Shows all successful iterations on x-axis, metric on y-axis.
        Best result highlighted with a star marker.

        Args:
            state: Experiment state with results history.
            plots_dir: Directory to save the plot.

        Returns:
            Path to saved PNG, or None if generation failed.
        """
        try:
            primary_metric = state.config.primary_metric
            if not primary_metric:
                return None

            successful = [
                e for e in state.experiments
                if e.success and primary_metric in e.metrics
            ]
            if not successful:
                return None

            iterations = [e.iteration for e in successful]
            values = [e.metrics[primary_metric] for e in successful]

            fig, ax = plt.subplots()
            ax.plot(
                iterations, values,
                'o-', color='#2196F3', linewidth=2, markersize=8,
                label=primary_metric.upper(),
            )

            # Highlight best result
            if state.best_experiment is not None:
                for i, e in enumerate(successful):
                    if e.experiment_name == state.best_experiment:
                        best_val = values[i]
                        ax.plot(
                            iterations[i], best_val,
                            '*', color='#4CAF50', markersize=20,
                            label=f'Best: {best_val:.4f}',
                        )
                        break

            ax.set_xlabel('Iteration')
            ax.set_ylabel(primary_metric.upper())
            ax.set_title(f'{primary_metric.upper()} Across Iterations')
            ax.legend()
            ax.grid(True, alpha=0.3)

            path = plots_dir / "metric_progression.png"
            fig.savefig(path)
            plt.close(fig)
            return path

        except Exception as e:
            print_warning(f"Failed to generate metric progression plot: {e}")
            plt.close('all')
            return None

    def _plot_model_comparison(
        self,
        state: ExperimentState,
        plots_dir: Path,
    ) -> Optional[Path]:
        """Bar chart comparing all model types on primary metric.

        Groups by model type, shows best metric per model type.
        Best overall model highlighted in green.

        Args:
            state: Experiment state with results history.
            plots_dir: Directory to save the plot.

        Returns:
            Path to saved PNG, or None if generation failed.
        """
        try:
            primary_metric = state.config.primary_metric
            if not primary_metric:
                return None

            successful = [
                e for e in state.experiments
                if e.success and primary_metric in e.metrics
            ]
            if not successful:
                return None

            # Get best metric per model type
            lower_is_better = self._is_lower_better(primary_metric)
            model_best: dict[str, float] = {}
            for e in successful:
                val = e.metrics[primary_metric]
                if e.model_type not in model_best:
                    model_best[e.model_type] = val
                else:
                    if lower_is_better:
                        model_best[e.model_type] = min(model_best[e.model_type], val)
                    else:
                        model_best[e.model_type] = max(model_best[e.model_type], val)

            models = list(model_best.keys())
            values = [model_best[m] for m in models]

            # Find the best model type for highlighting
            best_model_type = None
            if state.best_experiment:
                for e in state.experiments:
                    if e.experiment_name == state.best_experiment:
                        best_model_type = e.model_type
                        break

            colors = [
                '#4CAF50' if m == best_model_type else '#2196F3'
                for m in models
            ]

            fig, ax = plt.subplots()
            bars = ax.bar(range(len(models)), values, color=colors)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(primary_metric.upper())
            ax.set_title(f'Best {primary_metric.upper()} by Model Type')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9,
                )

            path = plots_dir / "model_comparison.png"
            fig.savefig(path)
            plt.close(fig)
            return path

        except Exception as e:
            print_warning(f"Failed to generate model comparison plot: {e}")
            plt.close('all')
            return None

    def _plot_improvement_over_baseline(
        self,
        state: ExperimentState,
        plots_dir: Path,
    ) -> Optional[Path]:
        """Bar chart showing baseline vs best model with improvement percentage.

        Shows two bars (baseline, best) and annotates percentage improvement.

        Args:
            state: Experiment state with results history.
            plots_dir: Directory to save the plot.

        Returns:
            Path to saved PNG, or None if generation failed.
        """
        try:
            primary_metric = state.config.primary_metric
            if not primary_metric:
                return None

            # Get baseline metric (first experiment)
            if not state.experiments or not state.experiments[0].success:
                return None
            baseline_val = state.experiments[0].metrics.get(primary_metric)

            if baseline_val is None or state.best_metric is None:
                return None
            if abs(baseline_val) < 1e-12:
                return None

            lower_is_better = self._is_lower_better(primary_metric)
            if lower_is_better:
                pct = ((baseline_val - state.best_metric) / abs(baseline_val)) * 100
            else:
                pct = ((state.best_metric - baseline_val) / abs(baseline_val)) * 100

            fig, ax = plt.subplots()

            labels = ['Baseline', 'Best Model']
            values = [baseline_val, state.best_metric]
            colors = ['#FF9800', '#4CAF50']

            bars = ax.bar(labels, values, color=colors, width=0.5)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                )

            improvement_label = f'{pct:+.1f}% {"improvement" if pct > 0 else "change"}'
            ax.set_title(
                f'{primary_metric.upper()}: Baseline vs Best ({improvement_label})'
            )
            ax.set_ylabel(primary_metric.upper())
            ax.grid(True, alpha=0.3, axis='y')

            path = plots_dir / "improvement_over_baseline.png"
            fig.savefig(path)
            plt.close(fig)
            return path

        except Exception as e:
            print_warning(f"Failed to generate improvement plot: {e}")
            plt.close('all')
            return None

    @staticmethod
    def _is_lower_better(metric_name: str) -> bool:
        """Check if a metric is one where lower values are better.

        Args:
            metric_name: Name of the metric.

        Returns:
            True if lower values are better (e.g., RMSE, MAE).
        """
        lower_better = ["rmse", "mse", "mae", "log_loss", "error"]
        return any(m in metric_name.lower() for m in lower_better)
