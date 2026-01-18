"""Tests for the CodeGenerator."""

import ast
import pytest
from pathlib import Path

from src.execution.code_generator import (
    CodeGenerator,
    CodeGenerationError,
    create_experiment_from_gemini_response,
)
from src.orchestration.state import ExperimentSpec, PreprocessingConfig


class TestCodeGenerator:
    """Test cases for CodeGenerator."""

    def test_init(self):
        """Test CodeGenerator initialization."""
        generator = CodeGenerator()
        assert generator.templates_dir.exists()
        assert generator.env is not None

    def test_generate_regressor(self, temp_data_file, temp_output_dir):
        """Test generating a regression experiment."""
        generator = CodeGenerator()

        spec = ExperimentSpec(
            experiment_name="test_rf_regressor",
            hypothesis="Test hypothesis",
            model_type="RandomForestRegressor",
            model_params={"n_estimators": 100, "max_depth": 5},
            preprocessing=PreprocessingConfig(
                missing_values="median",
                scaling="standard",
                encoding="onehot",
            ),
            reasoning="Test reasoning",
        )

        script_path = generator.generate(
            spec=spec,
            data_path=temp_data_file,
            target_column="target",
            task_type="regression",
            output_dir=temp_output_dir,
        )

        assert script_path.exists()
        assert script_path.suffix == ".py"

        # Validate generated code is valid Python
        code = script_path.read_text()
        ast.parse(code)  # Should not raise

    def test_generate_classifier(self, temp_classification_file, temp_output_dir):
        """Test generating a classification experiment."""
        generator = CodeGenerator()

        spec = ExperimentSpec(
            experiment_name="test_rf_classifier",
            hypothesis="Test classification",
            model_type="RandomForestClassifier",
            model_params={"n_estimators": 50},
            reasoning="Test",
        )

        script_path = generator.generate(
            spec=spec,
            data_path=temp_classification_file,
            target_column="target",
            task_type="classification",
            output_dir=temp_output_dir,
        )

        assert script_path.exists()
        code = script_path.read_text()
        ast.parse(code)

    def test_generate_baseline_regression(self, temp_data_file, temp_output_dir):
        """Test generating baseline for regression."""
        generator = CodeGenerator()

        script_path = generator.generate_baseline(
            data_path=temp_data_file,
            target_column="target",
            task_type="regression",
            output_dir=temp_output_dir,
        )

        assert script_path.exists()
        assert "baseline" in script_path.name
        code = script_path.read_text()
        assert "LinearRegression" in code

    def test_generate_baseline_classification(self, temp_classification_file, temp_output_dir):
        """Test generating baseline for classification."""
        generator = CodeGenerator()

        script_path = generator.generate_baseline(
            data_path=temp_classification_file,
            target_column="target",
            task_type="classification",
            output_dir=temp_output_dir,
        )

        assert script_path.exists()
        code = script_path.read_text()
        assert "LogisticRegression" in code

    def test_different_preprocessing(self, temp_data_file, temp_output_dir):
        """Test different preprocessing configurations."""
        generator = CodeGenerator()

        configs = [
            PreprocessingConfig(missing_values="mean", scaling="minmax", encoding="ordinal"),
            PreprocessingConfig(missing_values="drop", scaling="none", encoding="onehot"),
            PreprocessingConfig(missing_values="mode", scaling="standard", encoding="onehot"),
        ]

        for i, config in enumerate(configs):
            spec = ExperimentSpec(
                experiment_name=f"test_preprocessing_{i}",
                hypothesis="Test",
                model_type="RandomForestRegressor",
                preprocessing=config,
                reasoning="Test",
            )

            script_path = generator.generate(
                spec=spec,
                data_path=temp_data_file,
                target_column="target",
                task_type="regression",
                output_dir=temp_output_dir,
            )

            code = script_path.read_text()
            ast.parse(code)  # Should be valid Python

    def test_model_params_formatting(self, temp_data_file, temp_output_dir):
        """Test that model parameters are correctly formatted."""
        generator = CodeGenerator()

        spec = ExperimentSpec(
            experiment_name="test_params",
            hypothesis="Test params",
            model_type="GradientBoostingRegressor",
            model_params={
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 3,
            },
            reasoning="Test",
        )

        script_path = generator.generate(
            spec=spec,
            data_path=temp_data_file,
            target_column="target",
            task_type="regression",
            output_dir=temp_output_dir,
        )

        code = script_path.read_text()
        assert "n_estimators=200" in code
        assert "learning_rate=0.1" in code
        assert "max_depth=3" in code


class TestCreateExperimentFromResponse:
    """Test cases for parsing Gemini responses."""

    def test_parse_valid_response(self, mock_gemini_response):
        """Test parsing a valid response."""
        spec = create_experiment_from_gemini_response(mock_gemini_response)

        assert spec.experiment_name == "test_random_forest"
        assert spec.hypothesis == "Testing if RandomForest improves over baseline"
        assert spec.model_type == "RandomForestRegressor"
        assert spec.model_params["n_estimators"] == 100
        assert spec.preprocessing.missing_values == "median"

    def test_parse_minimal_response(self):
        """Test parsing a minimal response."""
        response = {
            "experiment_name": "minimal",
            "hypothesis": "Test",
            "model_type": "LinearRegression",
        }

        spec = create_experiment_from_gemini_response(response)
        assert spec.experiment_name == "minimal"
        assert spec.preprocessing.missing_values == "median"  # default

    def test_parse_missing_required_field(self):
        """Test error on missing required field."""
        response = {
            "experiment_name": "test",
            "hypothesis": "Test",
            # missing model_type
        }

        with pytest.raises(ValueError, match="model_type"):
            create_experiment_from_gemini_response(response)
