"""Experiment execution via subprocess."""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from src.config import ExperimentDefaults
from src.orchestration.state import ExperimentResult, ExperimentSpec


class ExperimentExecutionError(Exception):
    """Error during experiment execution."""

    pass


class ExperimentRunner:
    """Execute generated experiment scripts in isolated subprocesses.

    Features:
    - Subprocess isolation
    - Timeout handling
    - Stdout/stderr capture
    - JSON metrics parsing
    """

    def __init__(
        self,
        timeout: Optional[int] = None,
        python_executable: str = "python",
    ):
        """Initialize the experiment runner.

        Args:
            timeout: Timeout in seconds for each experiment.
            python_executable: Path to Python executable.
        """
        self.timeout = timeout or ExperimentDefaults().experiment_timeout
        self.python_executable = python_executable

    def run(
        self,
        script_path: Path,
        spec: ExperimentSpec,
        iteration: int,
    ) -> ExperimentResult:
        """Execute an experiment script and capture results.

        Args:
            script_path: Path to the generated Python script.
            spec: The experiment specification.
            iteration: Current iteration number.

        Returns:
            ExperimentResult with metrics or error information.
        """
        start_time = time.time()

        try:
            # Execute the script
            result = subprocess.run(
                [self.python_executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=script_path.parent,
            )

            execution_time = time.time() - start_time

            # Check for execution errors
            if result.returncode != 0:
                return self._create_error_result(
                    spec=spec,
                    iteration=iteration,
                    execution_time=execution_time,
                    error_message=f"Script exited with code {result.returncode}: {result.stderr}",
                    code_path=str(script_path),
                )

            # Parse JSON output
            try:
                output = json.loads(result.stdout.strip())
            except json.JSONDecodeError as e:
                return self._create_error_result(
                    spec=spec,
                    iteration=iteration,
                    execution_time=execution_time,
                    error_message=f"Failed to parse JSON output: {e}\nStdout: {result.stdout[:500]}",
                    code_path=str(script_path),
                )

            # Check if experiment reported success
            if not output.get("success", False):
                return self._create_error_result(
                    spec=spec,
                    iteration=iteration,
                    execution_time=execution_time,
                    error_message=output.get("error", "Unknown error"),
                    code_path=str(script_path),
                )

            # Create successful result
            return ExperimentResult(
                experiment_name=spec.experiment_name,
                iteration=iteration,
                model_type=spec.model_type,
                model_params=spec.model_params,
                preprocessing=spec.preprocessing,
                metrics=output.get("metrics", {}),
                hypothesis=spec.hypothesis,
                reasoning=spec.reasoning,
                execution_time=execution_time,
                success=True,
                code_path=str(script_path),
                timestamp=datetime.now(),
            )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return self._create_error_result(
                spec=spec,
                iteration=iteration,
                execution_time=execution_time,
                error_message=f"Experiment timed out after {self.timeout} seconds",
                code_path=str(script_path),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_error_result(
                spec=spec,
                iteration=iteration,
                execution_time=execution_time,
                error_message=str(e),
                code_path=str(script_path),
            )

    def _create_error_result(
        self,
        spec: ExperimentSpec,
        iteration: int,
        execution_time: float,
        error_message: str,
        code_path: str,
    ) -> ExperimentResult:
        """Create an error result."""
        return ExperimentResult(
            experiment_name=spec.experiment_name,
            iteration=iteration,
            model_type=spec.model_type,
            model_params=spec.model_params,
            preprocessing=spec.preprocessing,
            metrics={},
            hypothesis=spec.hypothesis,
            reasoning=spec.reasoning,
            execution_time=execution_time,
            success=False,
            error_message=error_message,
            code_path=code_path,
            timestamp=datetime.now(),
        )

    def run_script_directly(self, script_path: Path) -> dict:
        """Run a script and return raw output (for testing).

        Args:
            script_path: Path to the Python script.

        Returns:
            Dictionary with 'stdout', 'stderr', 'returncode'.
        """
        result = subprocess.run(
            [self.python_executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=script_path.parent,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
