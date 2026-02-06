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
    AnalysisResult,
    HypothesisSet,
    create_initial_state,
)
from src.cognitive.gemini_client import GeminiClient, GeminiError
from src.cognitive.experiment_designer import ExperimentDesigner
from src.cognitive.results_analyzer import ResultsAnalyzer
from src.cognitive.hypothesis_generator import HypothesisGenerator
from src.execution.data_profiler import DataProfiler
from src.execution.code_generator import CodeGenerator
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
    print_analysis,
    print_hypotheses,
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
        self.experiment_designer = ExperimentDesigner(self.gemini)
        self.results_analyzer = ResultsAnalyzer(self.gemini)
        self.hypothesis_generator = HypothesisGenerator(self.gemini)
        self.profiler = DataProfiler(data_path, target_column, task_type)
        self.code_generator = CodeGenerator()
        self.runner = ExperimentRunner()

        # MLflow tracker (initialized after profiling)
        self.tracker = None

        # Track latest analysis and hypotheses for cross-iteration context
        self._latest_analysis: Optional[AnalysisResult] = None
        self._latest_hypotheses: Optional[HypothesisSet] = None

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

            # Parse constraints and determine primary metric using ExperimentDesigner
            if self.constraints:
                parsed_constraints = self.experiment_designer.parse_constraints(self.constraints)
            else:
                parsed_constraints = None

            self.state.config.primary_metric = self.experiment_designer.select_primary_metric(
                self.task_type, parsed_constraints
            )

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

        # Analyze results
        self.state.phase = ExperimentPhase.RESULTS_ANALYSIS
        self._latest_analysis = self._analyze_results(result)

        if self._latest_analysis:
            print_analysis(self._latest_analysis.model_dump(), verbose=self.verbose)

        # Generate hypotheses for next iteration
        self.state.phase = ExperimentPhase.HYPOTHESIS_GENERATION
        self._latest_hypotheses = self._generate_hypotheses(self._latest_analysis)

        if self._latest_hypotheses:
            print_hypotheses(self._latest_hypotheses.model_dump(), verbose=self.verbose)

        self.state.phase = ExperimentPhase.EXPERIMENT_DESIGN
        self.save_state()

    def _design_experiment(self) -> Optional[ExperimentSpec]:
        """Use Gemini via ExperimentDesigner to design the next experiment."""
        iteration = self.state.current_iteration + 1

        # Build constraints with hypothesis context from previous iteration
        constraints_with_hypotheses = self.constraints or ""
        if self._latest_hypotheses:
            top = self._latest_hypotheses.get_top_hypothesis()
            if top:
                hypothesis_context = (
                    f"\n\n## Current Top Hypothesis\n"
                    f"- Statement: {top.statement}\n"
                    f"- Rationale: {top.rationale}\n"
                    f"- Confidence: {top.confidence_score:.0%}\n"
                )
                if top.suggested_model:
                    hypothesis_context += f"- Suggested model: {top.suggested_model}\n"
                if top.suggested_params:
                    hypothesis_context += f"- Suggested params: {top.suggested_params}\n"
                constraints_with_hypotheses += hypothesis_context

        try:
            return self.experiment_designer.design_experiment(
                data_profile=self.state.data_profile,
                previous_results=self.state.experiments,
                task_type=self.task_type,
                constraints=constraints_with_hypotheses if constraints_with_hypotheses else None,
                iteration=iteration,
            )

        except GeminiError as e:
            print_error("Gemini API error", str(e))
            return None

        except Exception as e:
            print_warning(f"Error designing experiment: {e}")
            return None

    def _analyze_results(self, result: ExperimentResult) -> Optional[AnalysisResult]:
        """Use ResultsAnalyzer to analyze the experiment results.

        Args:
            result: The experiment result to analyze.

        Returns:
            AnalysisResult or None if analysis fails.
        """
        try:
            return self.results_analyzer.analyze(
                current_result=result,
                state=self.state,
            )
        except Exception as e:
            print_warning(f"Results analysis failed: {e}")
            return None

    def _generate_hypotheses(self, analysis: Optional[AnalysisResult]) -> Optional[HypothesisSet]:
        """Use HypothesisGenerator to generate hypotheses for next iteration.

        Args:
            analysis: The analysis result to base hypotheses on.

        Returns:
            HypothesisSet or None if generation fails.
        """
        if analysis is None:
            return None
        try:
            return self.hypothesis_generator.generate(
                analysis=analysis,
                state=self.state,
            )
        except Exception as e:
            print_warning(f"Hypothesis generation failed: {e}")
            return None

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
