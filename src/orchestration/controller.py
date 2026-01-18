"""Experiment controller - main orchestration loop."""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import get_config, EXPERIMENTS_DIR, OUTPUTS_DIR
from src.orchestration.state import (
    ExperimentState,
    ExperimentPhase,
    ExperimentResult,
    ExperimentSpec,
    create_initial_state,
)
from src.cognitive.gemini_client import (
    GeminiClient,
    create_experiment_designer_prompt,
    GeminiError,
)
from src.execution.data_profiler import DataProfiler
from src.execution.code_generator import CodeGenerator, create_experiment_from_gemini_response
from src.execution.experiment_runner import ExperimentRunner
from src.persistence.mlflow_tracker import create_tracker
from src.utils.display import (
    console,
    print_phase,
    print_data_profile,
    print_iteration,
    print_results,
    print_reasoning,
    print_best_result,
    print_termination,
    print_summary,
    print_error,
    print_warning,
    print_success,
)


class ExperimentController:
    """Main controller for the ML experiment loop.

    Orchestrates:
    1. Data profiling
    2. Baseline model
    3. Iterative experiment design with Gemini
    4. Code generation and execution
    5. Results analysis
    6. Termination decision
    """

    def __init__(
        self,
        data_path: Path,
        target_column: str,
        task_type: str,
        constraints: Optional[str] = None,
        max_iterations: int = 20,
        time_budget: int = 3600,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
        resume_path: Optional[Path] = None,
    ):
        """Initialize the experiment controller.

        Args:
            data_path: Path to the dataset.
            target_column: Name of target column.
            task_type: 'classification' or 'regression'.
            constraints: Optional user constraints text.
            max_iterations: Maximum experiment iterations.
            time_budget: Time budget in seconds.
            output_dir: Output directory.
            verbose: Whether to show detailed reasoning.
            resume_path: Path to state file for resuming.
        """
        self.data_path = Path(data_path)
        self.target_column = target_column
        self.task_type = task_type
        self.constraints = constraints
        self.max_iterations = max_iterations
        self.time_budget = time_budget
        self.output_dir = output_dir or OUTPUTS_DIR
        self.verbose = verbose

        # Initialize or resume state
        if resume_path:
            self.state = ExperimentState.load(resume_path)
            print_success(f"Resumed from {resume_path}")
        else:
            self.state = create_initial_state(
                data_path=str(data_path),
                target_column=target_column,
                task_type=task_type,
                constraints=constraints,
                max_iterations=max_iterations,
                time_budget=time_budget,
                output_dir=str(self.output_dir),
            )

        # Initialize components
        self.config = get_config(verbose=verbose)
        self.gemini = GeminiClient()
        self.profiler = DataProfiler(data_path, target_column, task_type)
        self.code_generator = CodeGenerator()
        self.runner = ExperimentRunner()

        # MLflow tracker (initialized after profiling)
        self.tracker = None

        # Experiment output directory
        self.experiments_dir = EXPERIMENTS_DIR / self.state.session_id
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the complete experiment loop."""
        try:
            # Phase 1: Data Profiling
            if self.state.phase == ExperimentPhase.INITIALIZING:
                self._profile_data()

            # Phase 2: Baseline Model
            if self.state.phase == ExperimentPhase.DATA_PROFILING:
                self._run_baseline()

            # Phase 3: Iteration Loop
            while self.state.phase not in [ExperimentPhase.COMPLETED, ExperimentPhase.FAILED]:
                should_stop, reason = self.state.should_terminate()
                if should_stop:
                    self.state.termination_reason = reason
                    self.state.phase = ExperimentPhase.COMPLETED
                    break

                self._run_iteration()

            # Phase 4: Finalize
            self._finalize()

        except Exception as e:
            self.state.phase = ExperimentPhase.FAILED
            self.state.termination_reason = str(e)
            self.save_state()
            raise

    def _profile_data(self):
        """Profile the dataset."""
        print_phase("Data Profiling", "Analyzing dataset characteristics...")
        self.state.phase = ExperimentPhase.DATA_PROFILING

        try:
            profile = self.profiler.profile()
            self.state.data_profile = profile

            # Determine primary metric
            if self.task_type == "regression":
                self.state.config.primary_metric = "rmse"
            else:
                self.state.config.primary_metric = "f1"

            # Print profile summary
            print_data_profile(profile.model_dump())

            # Initialize MLflow tracker
            dataset_name = self.data_path.stem
            self.tracker = create_tracker(self.state.session_id, dataset_name)
            self.tracker.log_data_profile(profile)

            self.save_state()
            print_success("Data profiling complete")

        except Exception as e:
            print_error("Data profiling failed", str(e))
            raise

    def _run_baseline(self):
        """Run the baseline experiment."""
        print_phase("Baseline Model", "Establishing performance baseline...")
        self.state.phase = ExperimentPhase.BASELINE_MODELING

        try:
            # Generate baseline code
            script_path = self.code_generator.generate_baseline(
                data_path=self.data_path,
                target_column=self.target_column,
                task_type=self.task_type,
                output_dir=self.experiments_dir,
            )

            # Create spec for result tracking
            spec = ExperimentSpec(
                experiment_name="baseline",
                hypothesis="Establish baseline with simple model",
                model_type="LinearRegression" if self.task_type == "regression" else "LogisticRegression",
                reasoning="Starting with a simple model to establish baseline performance",
            )

            # Run the baseline
            result = self.runner.run(script_path, spec, iteration=0)

            # Update state
            self.state.add_experiment(result)
            self.state.phase = ExperimentPhase.EXPERIMENT_DESIGN

            # Log to MLflow
            if self.tracker:
                self.tracker.log_experiment(result)

            # Print results
            print_results(result.metrics, result.success, result.execution_time)

            if result.success and self.state.config.primary_metric:
                metric_value = result.metrics.get(self.state.config.primary_metric)
                if metric_value is not None:
                    print_best_result(
                        "baseline",
                        self.state.config.primary_metric,
                        metric_value,
                    )

            self.save_state()

        except Exception as e:
            print_error("Baseline experiment failed", str(e))
            raise

    def _run_iteration(self):
        """Run a single experiment iteration."""
        iteration = self.state.current_iteration + 1

        # Design experiment
        self.state.phase = ExperimentPhase.EXPERIMENT_DESIGN
        spec = self._design_experiment()

        if spec is None:
            print_warning("Failed to design experiment, stopping")
            self.state.phase = ExperimentPhase.COMPLETED
            self.state.termination_reason = "Failed to design experiment"
            return

        print_iteration(iteration, self.max_iterations, spec.experiment_name)

        if self.verbose:
            print_reasoning(
                iteration,
                spec.reasoning,
                spec.hypothesis,
                self.gemini.get_history_length(),
            )

        # Generate code
        self.state.phase = ExperimentPhase.CODE_GENERATION
        try:
            script_path = self.code_generator.generate(
                spec=spec,
                data_path=self.data_path,
                target_column=self.target_column,
                task_type=self.task_type,
                output_dir=self.experiments_dir,
            )
        except Exception as e:
            print_error("Code generation failed", str(e))
            # Create a failed result
            result = ExperimentResult(
                experiment_name=spec.experiment_name,
                iteration=iteration,
                model_type=spec.model_type,
                success=False,
                error_message=f"Code generation failed: {e}",
            )
            self.state.add_experiment(result)
            return

        # Execute experiment
        self.state.phase = ExperimentPhase.EXPERIMENT_EXECUTION
        result = self.runner.run(script_path, spec, iteration)

        # Update state
        self.state.add_experiment(result)

        # Log to MLflow
        if self.tracker:
            self.tracker.log_experiment(result)

        # Print results
        print_results(result.metrics, result.success, result.execution_time)

        if result.success and self.state.best_experiment:
            print_best_result(
                self.state.best_experiment,
                self.state.config.primary_metric or "metric",
                self.state.best_metric or 0,
            )

        self.state.phase = ExperimentPhase.EXPERIMENT_DESIGN
        self.save_state()

    def _design_experiment(self) -> Optional[ExperimentSpec]:
        """Use Gemini to design the next experiment."""
        # Prepare previous results summary
        previous_results = [
            {
                "name": exp.experiment_name,
                "model": exp.model_type,
                "metrics": exp.metrics,
                "hypothesis": exp.hypothesis,
                "success": exp.success,
            }
            for exp in self.state.experiments[-5:]  # Last 5 experiments
        ]

        # Create prompt
        prompt = create_experiment_designer_prompt(
            data_profile=self.state.data_profile.model_dump() if self.state.data_profile else {},
            previous_results=previous_results,
            constraints=self.constraints,
            task_type=self.task_type,
        )

        try:
            response = self.gemini.generate_json(
                prompt=prompt,
                system_instruction="You are an expert ML engineer designing experiments. Always respond with valid JSON.",
                thinking_level="high",
            )

            return create_experiment_from_gemini_response(response)

        except GeminiError as e:
            print_error("Gemini API error", str(e))
            return self._get_fallback_experiment()

        except ValueError as e:
            print_warning(f"Invalid Gemini response: {e}")
            return self._get_fallback_experiment()

    def _get_fallback_experiment(self) -> ExperimentSpec:
        """Get a fallback experiment when Gemini fails."""
        iteration = self.state.current_iteration + 1

        if self.task_type == "regression":
            models = ["RandomForestRegressor", "GradientBoostingRegressor", "Ridge"]
        else:
            models = ["RandomForestClassifier", "GradientBoostingClassifier", "LogisticRegression"]

        model_idx = iteration % len(models)
        model_type = models[model_idx]

        return ExperimentSpec(
            experiment_name=f"fallback_{model_type.lower()}_{iteration}",
            hypothesis=f"Fallback experiment with {model_type}",
            model_type=model_type,
            model_params={"n_estimators": 100} if "Forest" in model_type or "Boosting" in model_type else {},
            reasoning="Fallback experiment due to Gemini API issues",
        )

    def _finalize(self):
        """Finalize the experiment session."""
        print_termination(self.state.termination_reason or "Completed")

        # Log final summary to MLflow
        if self.tracker:
            self.tracker.log_final_summary(self.state)

        # Print summary
        print_summary(self.state.get_summary())

        # Save final state
        self.state.phase = ExperimentPhase.COMPLETED
        self.save_state()

        print_success(f"Results saved to {self.output_dir}")

    def save_state(self):
        """Save current state to file."""
        state_path = self.output_dir / f"state_{self.state.session_id}.json"
        self.state.save(state_path)
