"""Tests for the ExperimentRunner."""

import json
import pytest
from pathlib import Path

from src.execution.experiment_runner import ExperimentRunner
from src.execution.code_generator import CodeGenerator
from src.orchestration.state import ExperimentSpec, PreprocessingConfig


class TestExperimentRunner:
    """Test cases for ExperimentRunner."""

    def test_init(self):
        """Test ExperimentRunner initialization."""
        runner = ExperimentRunner()
        assert runner.timeout > 0
        assert runner.python_executable == "python"

    def test_run_successful_experiment(self, temp_data_file, temp_output_dir):
        """Test running a successful experiment."""
        # Generate a script first
        generator = CodeGenerator()
        spec = ExperimentSpec(
            experiment_name="test_success",
            hypothesis="Test",
            model_type="RandomForestRegressor",
            model_params={"n_estimators": 10},
            preprocessing=PreprocessingConfig(),
            reasoning="Test",
        )

        script_path = generator.generate(
            spec=spec,
            data_path=temp_data_file,
            target_column="target",
            task_type="regression",
            output_dir=temp_output_dir,
        )

        # Run the experiment
        runner = ExperimentRunner(timeout=60)
        result = runner.run(script_path, spec, iteration=1)

        assert result.success
        assert result.experiment_name == "test_success"
        assert "rmse" in result.metrics
        assert "r2" in result.metrics
        assert result.execution_time > 0

    def test_run_classification_experiment(self, temp_classification_file, temp_output_dir):
        """Test running a classification experiment."""
        generator = CodeGenerator()
        spec = ExperimentSpec(
            experiment_name="test_classification",
            hypothesis="Test classification",
            model_type="RandomForestClassifier",
            model_params={"n_estimators": 10},
            reasoning="Test",
        )

        script_path = generator.generate(
            spec=spec,
            data_path=temp_classification_file,
            target_column="target",
            task_type="classification",
            output_dir=temp_output_dir,
        )

        runner = ExperimentRunner(timeout=60)
        result = runner.run(script_path, spec, iteration=1)

        assert result.success
        assert "accuracy" in result.metrics or "f1" in result.metrics

    def test_run_timeout(self, temp_output_dir):
        """Test handling of experiment timeout."""
        # Create a script that sleeps forever
        script_path = temp_output_dir / "slow_script.py"
        script_path.write_text("""
import time
import json
time.sleep(100)
print(json.dumps({"success": True, "metrics": {}}))
""")

        spec = ExperimentSpec(
            experiment_name="test_timeout",
            hypothesis="Test timeout",
            model_type="Test",
            reasoning="Test",
        )

        runner = ExperimentRunner(timeout=1)  # 1 second timeout
        result = runner.run(script_path, spec, iteration=1)

        assert not result.success
        assert "timed out" in result.error_message.lower()

    def test_run_script_error(self, temp_output_dir):
        """Test handling of script errors."""
        # Create a script that raises an error
        script_path = temp_output_dir / "error_script.py"
        script_path.write_text("""
import json
raise ValueError("Test error")
""")

        spec = ExperimentSpec(
            experiment_name="test_error",
            hypothesis="Test error",
            model_type="Test",
            reasoning="Test",
        )

        runner = ExperimentRunner()
        result = runner.run(script_path, spec, iteration=1)

        assert not result.success
        assert result.error_message is not None

    def test_run_invalid_json_output(self, temp_output_dir):
        """Test handling of invalid JSON output."""
        script_path = temp_output_dir / "invalid_json.py"
        script_path.write_text("""
print("This is not JSON")
""")

        spec = ExperimentSpec(
            experiment_name="test_invalid",
            hypothesis="Test",
            model_type="Test",
            reasoning="Test",
        )

        runner = ExperimentRunner()
        result = runner.run(script_path, spec, iteration=1)

        assert not result.success
        assert "JSON" in result.error_message

    def test_run_script_directly(self, temp_output_dir):
        """Test running a script and getting raw output."""
        script_path = temp_output_dir / "simple_script.py"
        script_path.write_text("""
import json
print(json.dumps({"success": True, "metrics": {"test": 1.0}}))
""")

        runner = ExperimentRunner()
        output = runner.run_script_directly(script_path)

        assert output["returncode"] == 0
        assert "success" in output["stdout"]
