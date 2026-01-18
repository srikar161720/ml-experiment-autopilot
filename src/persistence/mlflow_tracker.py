"""MLflow integration for experiment tracking."""

from pathlib import Path
from typing import Optional, Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import MLRUNS_DIR
from src.orchestration.state import ExperimentResult, ExperimentState, DataProfile


class MLflowTracker:
    """Track experiments with MLflow.

    Features:
    - Local MLflow tracking
    - Automatic experiment creation
    - Metric and parameter logging
    - Artifact storage
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
    ):
        """Initialize MLflow tracking.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking URI. Defaults to local storage.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or str(MLRUNS_DIR)

        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient(self.tracking_uri)

    def log_data_profile(self, profile: DataProfile):
        """Log the data profile as experiment metadata.

        Args:
            profile: DataProfile from the data profiler.
        """
        with mlflow.start_run(run_name="data_profile"):
            # Log dataset info as parameters
            mlflow.log_params({
                "n_rows": profile.n_rows,
                "n_columns": profile.n_columns,
                "n_numeric_features": len(profile.numeric_columns),
                "n_categorical_features": len(profile.categorical_columns),
                "target_column": profile.target_column,
                "target_type": profile.target_type,
            })

            # Log missing value summary
            total_missing = sum(profile.missing_values.values())
            mlflow.log_metric("total_missing_values", total_missing)

            # Log profile as artifact
            profile_path = Path("data_profile.json")
            profile_path.write_text(profile.model_dump_json(indent=2))
            mlflow.log_artifact(str(profile_path))
            profile_path.unlink()

    def log_experiment(self, result: ExperimentResult):
        """Log a single experiment run.

        Args:
            result: ExperimentResult from the experiment runner.
        """
        with mlflow.start_run(run_name=result.experiment_name):
            # Log parameters
            params = {
                "model_type": result.model_type,
                "iteration": result.iteration,
                **{f"model_{k}": v for k, v in result.model_params.items()},
                "preprocessing_missing": result.preprocessing.missing_values,
                "preprocessing_scaling": result.preprocessing.scaling,
                "preprocessing_encoding": result.preprocessing.encoding,
            }
            mlflow.log_params(params)

            # Log metrics
            if result.success and result.metrics:
                mlflow.log_metrics(result.metrics)

            # Log execution info
            mlflow.log_metric("execution_time", result.execution_time)
            mlflow.log_metric("success", 1 if result.success else 0)

            # Log tags
            mlflow.set_tags({
                "hypothesis": result.hypothesis[:250] if result.hypothesis else "",
                "success": str(result.success),
            })

            # Log reasoning as artifact
            if result.reasoning:
                reasoning_path = Path("reasoning.txt")
                reasoning_path.write_text(result.reasoning)
                mlflow.log_artifact(str(reasoning_path))
                reasoning_path.unlink()

            # Log code as artifact if available
            if result.code_path:
                code_path = Path(result.code_path)
                if code_path.exists():
                    mlflow.log_artifact(str(code_path))

            # Log error if failed
            if not result.success and result.error_message:
                error_path = Path("error.txt")
                error_path.write_text(result.error_message)
                mlflow.log_artifact(str(error_path))
                error_path.unlink()

    def log_final_summary(self, state: ExperimentState):
        """Log final experiment summary.

        Args:
            state: Final ExperimentState after all iterations.
        """
        with mlflow.start_run(run_name="final_summary"):
            # Log summary metrics
            mlflow.log_metrics({
                "total_iterations": state.current_iteration,
                "successful_experiments": sum(1 for e in state.experiments if e.success),
                "total_time_seconds": state.get_elapsed_time(),
            })

            if state.best_metric is not None:
                mlflow.log_metric("best_metric", state.best_metric)

            # Log tags
            mlflow.set_tags({
                "best_experiment": state.best_experiment or "none",
                "termination_reason": state.termination_reason or "completed",
                "phase": state.phase.value,
            })

            # Log state as artifact
            state_path = Path("final_state.json")
            state.save(state_path)
            mlflow.log_artifact(str(state_path))
            state_path.unlink()

    def get_best_run(self, metric_name: str, ascending: bool = True) -> Optional[dict]:
        """Get the best run based on a metric.

        Args:
            metric_name: Name of the metric to optimize.
            ascending: Whether lower values are better.

        Returns:
            Dictionary with run info, or None if no runs found.
        """
        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"metrics.{metric_name} IS NOT NULL",
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1,
        )

        if not runs:
            return None

        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

    def get_all_runs(self) -> list[dict]:
        """Get all runs in the experiment.

        Returns:
            List of run dictionaries.
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"],
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "status": run.info.status,
            }
            for run in runs
        ]


def create_tracker(session_id: str, dataset_name: str) -> MLflowTracker:
    """Create an MLflow tracker with a descriptive experiment name.

    Args:
        session_id: Unique session identifier.
        dataset_name: Name of the dataset being used.

    Returns:
        Configured MLflowTracker.
    """
    experiment_name = f"autopilot_{dataset_name}_{session_id}"
    return MLflowTracker(experiment_name)
