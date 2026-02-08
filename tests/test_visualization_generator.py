"""Tests for the VisualizationGenerator component."""

import pytest
from unittest.mock import patch
from pathlib import Path

from src.execution.visualization_generator import VisualizationGenerator
from src.orchestration.state import (
    ExperimentResult,
    ExperimentState,
    ExperimentConfig,
    DataProfile,
    TaskType,
)


# ── Shared helpers ──────────────────────────────────────────────────────


def _make_sample_data_profile() -> DataProfile:
    """Create a realistic DataProfile fixture."""
    return DataProfile(
        n_rows=1460,
        n_columns=81,
        columns=["SalePrice", "OverallQual", "GrLivArea", "Neighborhood"],
        column_types={
            "SalePrice": "float64",
            "OverallQual": "int64",
            "GrLivArea": "int64",
            "Neighborhood": "object",
        },
        numeric_columns=["SalePrice", "OverallQual", "GrLivArea"],
        categorical_columns=["Neighborhood"],
        target_column="SalePrice",
        target_type="continuous",
        missing_values={"OverallQual": 0, "GrLivArea": 0, "Neighborhood": 5},
        missing_percentages={"OverallQual": 0.0, "GrLivArea": 0.0, "Neighborhood": 0.34},
        numeric_stats={
            "SalePrice": {"mean": 180921.0, "std": 79442.0, "min": 34900.0, "max": 755000.0},
        },
        categorical_stats={
            "Neighborhood": {"nunique": 25, "top": "NAmes"},
        },
        target_stats={"mean": 180921.0, "std": 79442.0, "min": 34900.0, "max": 755000.0},
    )


def _make_sample_experiment_config() -> ExperimentConfig:
    """Create a regression ExperimentConfig."""
    return ExperimentConfig(
        data_path="/data/house_prices_train.csv",
        target_column="SalePrice",
        task_type=TaskType.REGRESSION,
        max_iterations=10,
        primary_metric="rmse",
        improvement_threshold=0.005,
    )


def _make_sample_state() -> ExperimentState:
    """Create a populated ExperimentState with 3 experiments."""
    config = _make_sample_experiment_config()

    baseline = ExperimentResult(
        experiment_name="baseline",
        iteration=0,
        model_type="LinearRegression",
        metrics={"rmse": 0.50, "r2": 0.70},
        hypothesis="Establish baseline with simple model",
        success=True,
        execution_time=2.5,
    )
    rf_basic = ExperimentResult(
        experiment_name="rf_basic",
        iteration=1,
        model_type="RandomForestRegressor",
        model_params={"n_estimators": 100, "max_depth": 10},
        metrics={"rmse": 0.35, "r2": 0.82},
        hypothesis="RandomForest should capture non-linear patterns",
        reasoning="Tree-based models may outperform linear on this dataset",
        success=True,
        execution_time=5.1,
    )
    rf_tuned = ExperimentResult(
        experiment_name="rf_tuned",
        iteration=2,
        model_type="RandomForestRegressor",
        model_params={"n_estimators": 200, "max_depth": 15},
        metrics={"rmse": 0.30, "r2": 0.87},
        hypothesis="Increasing n_estimators and max_depth improves RF",
        reasoning="Previous RF showed promise, tuning hyperparameters",
        success=True,
        execution_time=8.3,
    )

    state = ExperimentState(config=config)
    state.experiments = [baseline, rf_basic, rf_tuned]
    state.current_iteration = 3
    state.best_metric = 0.30
    state.best_experiment = "rf_tuned"
    state.termination_reason = "Performance plateau detected"
    state.data_profile = _make_sample_data_profile()

    return state


def _make_state_with_failed_experiments() -> ExperimentState:
    """Create an ExperimentState where some experiments failed."""
    config = _make_sample_experiment_config()

    baseline = ExperimentResult(
        experiment_name="baseline",
        iteration=0,
        model_type="LinearRegression",
        metrics={"rmse": 0.50, "r2": 0.70},
        hypothesis="Establish baseline",
        success=True,
        execution_time=2.5,
    )
    failed = ExperimentResult(
        experiment_name="failed_exp",
        iteration=1,
        model_type="SVR",
        metrics={},
        hypothesis="SVR might work",
        success=False,
        error_message="Memory error",
        execution_time=1.0,
    )
    rf = ExperimentResult(
        experiment_name="rf_basic",
        iteration=2,
        model_type="RandomForestRegressor",
        model_params={"n_estimators": 100},
        metrics={"rmse": 0.35, "r2": 0.82},
        hypothesis="RandomForest should capture patterns",
        success=True,
        execution_time=5.1,
    )

    state = ExperimentState(config=config)
    state.experiments = [baseline, failed, rf]
    state.current_iteration = 3
    state.best_metric = 0.35
    state.best_experiment = "rf_basic"

    return state


def _make_all_failed_state() -> ExperimentState:
    """Create an ExperimentState where all experiments failed."""
    config = _make_sample_experiment_config()

    failed1 = ExperimentResult(
        experiment_name="failed1",
        iteration=0,
        model_type="LinearRegression",
        metrics={},
        hypothesis="Baseline",
        success=False,
        error_message="Crash",
        execution_time=1.0,
    )
    failed2 = ExperimentResult(
        experiment_name="failed2",
        iteration=1,
        model_type="RandomForestRegressor",
        metrics={},
        hypothesis="Try RF",
        success=False,
        error_message="Timeout",
        execution_time=1.0,
    )

    state = ExperimentState(config=config)
    state.experiments = [failed1, failed2]
    state.current_iteration = 2

    return state


def _make_single_experiment_state() -> ExperimentState:
    """Create an ExperimentState with just one successful experiment."""
    config = _make_sample_experiment_config()

    baseline = ExperimentResult(
        experiment_name="baseline",
        iteration=0,
        model_type="LinearRegression",
        metrics={"rmse": 0.50, "r2": 0.70},
        hypothesis="Establish baseline",
        success=True,
        execution_time=2.5,
    )

    state = ExperimentState(config=config)
    state.experiments = [baseline]
    state.current_iteration = 1
    state.best_metric = 0.50
    state.best_experiment = "baseline"

    return state


# ── TestVisualizationGenerator ─────────────────────────────────────────


class TestVisualizationGenerator:
    """Main integration tests for VisualizationGenerator."""

    def test_init(self):
        viz = VisualizationGenerator()
        assert viz is not None

    def test_generate_success(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        plot_paths = viz.generate(state, tmp_path)

        assert len(plot_paths) == 3
        assert all(p.exists() for p in plot_paths)
        assert all(p.suffix == ".png" for p in plot_paths)
        assert all(p.stat().st_size > 0 for p in plot_paths)

        # Verify expected filenames
        names = {p.name for p in plot_paths}
        assert "metric_progression.png" in names
        assert "model_comparison.png" in names
        assert "improvement_over_baseline.png" in names

    def test_generate_creates_plots_dir(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        viz.generate(state, tmp_path)

        plots_dir = tmp_path / "plots"
        assert plots_dir.is_dir()

    def test_generate_no_primary_metric(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        state.config.primary_metric = None

        plot_paths = viz.generate(state, tmp_path)
        assert plot_paths == []

    def test_generate_no_successful_experiments(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_all_failed_state()

        plot_paths = viz.generate(state, tmp_path)
        assert plot_paths == []

    def test_generate_single_experiment(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_single_experiment_state()

        plot_paths = viz.generate(state, tmp_path)
        # Should produce at least metric_progression and model_comparison
        assert len(plot_paths) >= 2
        assert all(p.exists() for p in plot_paths)


# ── TestMetricProgression ──────────────────────────────────────────────


class TestMetricProgression:
    """Tests for the metric progression line chart."""

    def test_plot_created(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_metric_progression(state, plots_dir)
        assert path is not None
        assert path.exists()
        assert path.name == "metric_progression.png"

    def test_plot_with_failed_experiments(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_state_with_failed_experiments()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_metric_progression(state, plots_dir)
        assert path is not None
        assert path.exists()

    def test_returns_none_no_metric(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        state.config.primary_metric = None
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_metric_progression(state, plots_dir)
        assert path is None

    def test_returns_none_on_error(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        with patch(
            "src.execution.visualization_generator.plt.subplots",
            side_effect=RuntimeError("plot error"),
        ):
            path = viz._plot_metric_progression(state, plots_dir)
            assert path is None


# ── TestModelComparison ────────────────────────────────────────────────


class TestModelComparison:
    """Tests for the model comparison bar chart."""

    def test_plot_created(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_model_comparison(state, plots_dir)
        assert path is not None
        assert path.exists()
        assert path.name == "model_comparison.png"

    def test_multiple_model_types(self, tmp_path):
        """Verify chart works when multiple distinct model types are present."""
        viz = VisualizationGenerator()
        state = _make_state_with_failed_experiments()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_model_comparison(state, plots_dir)
        assert path is not None
        assert path.exists()

    def test_returns_none_no_experiments(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_all_failed_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_model_comparison(state, plots_dir)
        assert path is None


# ── TestImprovementOverBaseline ────────────────────────────────────────


class TestImprovementOverBaseline:
    """Tests for the improvement over baseline bar chart."""

    def test_plot_created(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_improvement_over_baseline(state, plots_dir)
        assert path is not None
        assert path.exists()
        assert path.name == "improvement_over_baseline.png"

    def test_no_baseline_returns_none(self, tmp_path):
        """If baseline experiment failed, returns None."""
        viz = VisualizationGenerator()
        state = _make_all_failed_state()
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_improvement_over_baseline(state, plots_dir)
        assert path is None

    def test_zero_baseline_returns_none(self, tmp_path):
        """If baseline metric is 0, returns None (can't compute %)."""
        viz = VisualizationGenerator()
        state = _make_sample_state()
        state.experiments[0].metrics["rmse"] = 0.0
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_improvement_over_baseline(state, plots_dir)
        assert path is None

    def test_no_best_metric_returns_none(self, tmp_path):
        viz = VisualizationGenerator()
        state = _make_sample_state()
        state.best_metric = None
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()

        path = viz._plot_improvement_over_baseline(state, plots_dir)
        assert path is None


# ── TestIsLowerBetter ──────────────────────────────────────────────────


class TestIsLowerBetter:
    """Tests for the metric direction helper."""

    def test_rmse_lower_is_better(self):
        assert VisualizationGenerator._is_lower_better("rmse") is True

    def test_mse_lower_is_better(self):
        assert VisualizationGenerator._is_lower_better("mse") is True

    def test_mae_lower_is_better(self):
        assert VisualizationGenerator._is_lower_better("mae") is True

    def test_accuracy_higher_is_better(self):
        assert VisualizationGenerator._is_lower_better("accuracy") is False

    def test_f1_higher_is_better(self):
        assert VisualizationGenerator._is_lower_better("f1") is False

    def test_r2_higher_is_better(self):
        assert VisualizationGenerator._is_lower_better("r2") is False

    def test_log_loss_lower_is_better(self):
        assert VisualizationGenerator._is_lower_better("log_loss") is True
