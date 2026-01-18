"""Pydantic state models for experiment tracking."""

import json
import time
import uuid
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, Field, ConfigDict


class TaskType(str, Enum):
    """Type of ML task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ExperimentPhase(str, Enum):
    """Current phase of the experiment loop."""

    INITIALIZING = "initializing"
    DATA_PROFILING = "data_profiling"
    BASELINE_MODELING = "baseline_modeling"
    EXPERIMENT_DESIGN = "experiment_design"
    CODE_GENERATION = "code_generation"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULTS_ANALYSIS = "results_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ITERATION_DECISION = "iteration_decision"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""

    missing_values: str = "median"  # drop, mean, median, mode, constant
    scaling: str = "standard"  # standard, minmax, none
    encoding: str = "onehot"  # onehot, ordinal
    target_transform: Optional[str] = None  # log, none


class ExperimentSpec(BaseModel):
    """Specification for a single experiment designed by Gemini."""

    model_config = ConfigDict(protected_namespaces=())

    experiment_name: str
    hypothesis: str
    model_type: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    reasoning: str = ""


class ExperimentResult(BaseModel):
    """Results from a single experiment run."""

    model_config = ConfigDict(protected_namespaces=())

    experiment_name: str
    iteration: int
    model_type: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    metrics: dict[str, float] = Field(default_factory=dict)
    hypothesis: str = ""
    reasoning: str = ""
    execution_time: float = 0.0  # seconds
    success: bool = True
    error_message: Optional[str] = None
    code_path: Optional[str] = None
    model_path: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_primary_metric(self, metric_name: str) -> Optional[float]:
        """Get the primary metric value."""
        return self.metrics.get(metric_name)


class ExperimentConfig(BaseModel):
    """Configuration for the experiment session."""

    data_path: str
    target_column: str
    task_type: TaskType
    constraints: Optional[str] = None
    max_iterations: int = 20
    time_budget: int = 3600  # seconds
    plateau_threshold: int = 3
    improvement_threshold: float = 0.005
    target_metric_value: Optional[float] = None
    primary_metric: Optional[str] = None  # e.g., "rmse", "f1", "accuracy"
    output_dir: Optional[str] = None


class DataProfile(BaseModel):
    """Profile of the dataset."""

    n_rows: int
    n_columns: int
    columns: list[str]
    column_types: dict[str, str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    target_column: str
    target_type: str
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    numeric_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    categorical_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    target_stats: dict[str, Any] = Field(default_factory=dict)


class ConversationEntry(BaseModel):
    """Entry in the Gemini conversation history."""

    role: str  # "user" or "model"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ExperimentState(BaseModel):
    """Complete state of an experiment session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: ExperimentConfig
    data_profile: Optional[DataProfile] = None
    experiments: list[ExperimentResult] = Field(default_factory=list)
    current_iteration: int = 0
    phase: ExperimentPhase = ExperimentPhase.INITIALIZING
    best_metric: Optional[float] = None
    best_experiment: Optional[str] = None
    iterations_without_improvement: int = 0
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    gemini_conversation_history: list[ConversationEntry] = Field(default_factory=list)
    agent_recommends_stop: bool = False
    termination_reason: Optional[str] = None

    def add_experiment(self, result: ExperimentResult):
        """Add an experiment result and update tracking."""
        self.experiments.append(result)
        self.current_iteration += 1

        # Update best metric tracking
        if self.config.primary_metric and result.success:
            current_value = result.get_primary_metric(self.config.primary_metric)
            if current_value is not None:
                # Determine if higher or lower is better
                is_better = self._is_better_metric(current_value)
                if is_better:
                    self.best_metric = current_value
                    self.best_experiment = result.experiment_name
                    self.iterations_without_improvement = 0
                else:
                    self.iterations_without_improvement += 1

    def _is_better_metric(self, new_value: float) -> bool:
        """Check if new metric value is better than current best."""
        if self.best_metric is None:
            return True

        # Metrics where lower is better
        lower_is_better = ["rmse", "mse", "mae", "log_loss", "error"]

        metric_name = self.config.primary_metric.lower() if self.config.primary_metric else ""
        if any(m in metric_name for m in lower_is_better):
            improvement = (self.best_metric - new_value) / abs(self.best_metric)
            return improvement > self.config.improvement_threshold
        else:
            # Higher is better (accuracy, f1, r2, etc.)
            improvement = (new_value - self.best_metric) / abs(self.best_metric)
            return improvement > self.config.improvement_threshold

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def should_terminate(self) -> tuple[bool, str]:
        """Check if experiment loop should terminate.

        Returns:
            Tuple of (should_stop, reason).
        """
        # Max iterations
        if self.current_iteration >= self.config.max_iterations:
            return True, "Maximum iterations reached"

        # Time budget
        if self.get_elapsed_time() > self.config.time_budget:
            return True, "Time budget exhausted"

        # Plateau detection
        if self.iterations_without_improvement >= self.config.plateau_threshold:
            return True, "Performance plateau detected"

        # Target achieved
        if self.config.target_metric_value and self.best_metric:
            metric_name = self.config.primary_metric.lower() if self.config.primary_metric else ""
            lower_is_better = ["rmse", "mse", "mae", "log_loss", "error"]
            if any(m in metric_name for m in lower_is_better):
                if self.best_metric <= self.config.target_metric_value:
                    return True, "Target metric achieved"
            else:
                if self.best_metric >= self.config.target_metric_value:
                    return True, "Target metric achieved"

        # Agent decision
        if self.agent_recommends_stop:
            return True, "Agent determined further improvement unlikely"

        return False, ""

    def get_summary(self) -> dict:
        """Get a summary of the current state."""
        return {
            "session_id": self.session_id,
            "phase": self.phase.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.config.max_iterations,
            "elapsed_time": self.get_elapsed_time(),
            "best_metric": self.best_metric,
            "best_experiment": self.best_experiment,
            "total_experiments": len(self.experiments),
            "successful_experiments": sum(1 for e in self.experiments if e.success),
        }

    def save(self, path: Path):
        """Save state to JSON file."""
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from JSON file."""
        return cls.model_validate_json(path.read_text())


def create_initial_state(
    data_path: str,
    target_column: str,
    task_type: str,
    constraints: Optional[str] = None,
    max_iterations: int = 20,
    time_budget: int = 3600,
    output_dir: Optional[str] = None,
) -> ExperimentState:
    """Create initial experiment state.

    Args:
        data_path: Path to the dataset.
        target_column: Name of the target column.
        task_type: 'classification' or 'regression'.
        constraints: Optional user constraints text.
        max_iterations: Maximum experiment iterations.
        time_budget: Time budget in seconds.
        output_dir: Output directory path.

    Returns:
        Initialized ExperimentState.
    """
    config = ExperimentConfig(
        data_path=data_path,
        target_column=target_column,
        task_type=TaskType(task_type),
        constraints=constraints,
        max_iterations=max_iterations,
        time_budget=time_budget,
        output_dir=output_dir,
    )
    return ExperimentState(config=config)
