"""Tests for the ExperimentDesigner component."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.cognitive.experiment_designer import (
    ExperimentDesigner,
    ParsedConstraints,
    EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
)
from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiResponse,
    GeminiInvalidResponseError,
    GeminiError,
)
from src.orchestration.state import (
    DataProfile,
    ExperimentResult,
    ExperimentSpec,
    PreprocessingConfig,
)
from src.config import GeminiConfig


class TestExperimentDesigner:
    """Test cases for ExperimentDesigner class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Gemini config."""
        return GeminiConfig(
            api_key="test_key",
            model="test-model",
            temperature=1.0,
            max_retries=3,
            retry_delay=0.1,
        )

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.generativeai module."""
        with patch("src.cognitive.gemini_client.genai") as mock:
            yield mock

    @pytest.fixture
    def mock_gemini_client(self, mock_config, mock_genai):
        """Create a mocked GeminiClient."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        return GeminiClient(mock_config)

    @pytest.fixture
    def sample_data_profile(self):
        """Create a sample DataProfile for testing."""
        return DataProfile(
            n_rows=1000,
            n_columns=10,
            columns=["feature1", "feature2", "feature3", "cat1", "target"],
            column_types={
                "feature1": "float64",
                "feature2": "float64",
                "feature3": "float64",
                "cat1": "object",
                "target": "float64",
            },
            numeric_columns=["feature1", "feature2", "feature3", "target"],
            categorical_columns=["cat1"],
            target_column="target",
            target_type="float64",
            missing_values={"feature1": 10, "feature2": 0, "cat1": 5},
            missing_percentages={"feature1": 1.0, "feature2": 0.0, "cat1": 0.5},
            numeric_stats={
                "feature1": {"mean": 0.5, "std": 1.0, "min": -3.0, "max": 3.0},
            },
            categorical_stats={
                "cat1": {"unique": 3, "top": "A", "freq": 400},
            },
            target_stats={"mean": 100.0, "std": 50.0, "min": 0.0, "max": 300.0},
        )

    @pytest.fixture
    def sample_experiment_result(self):
        """Create a sample ExperimentResult for testing."""
        return ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="LinearRegression",
            metrics={"rmse": 0.45, "r2": 0.75},
            hypothesis="Establish baseline with simple model",
            reasoning="Starting with linear regression as baseline",
            execution_time=5.0,
            success=True,
        )

    def test_init(self, mock_gemini_client):
        """Test ExperimentDesigner initialization."""
        designer = ExperimentDesigner(mock_gemini_client)

        assert designer.client == mock_gemini_client
        assert designer._parsed_constraints is None
        assert designer._design_count == 0

    def test_design_experiment_success(self, mock_gemini_client, sample_data_profile, mock_genai):
        """Test successful experiment design."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = '''{
            "experiment_name": "rf_experiment_1",
            "hypothesis": "RandomForest should capture non-linear patterns",
            "model_type": "RandomForestRegressor",
            "model_params": {"n_estimators": 100, "max_depth": 10},
            "preprocessing": {
                "missing_values": "median",
                "scaling": "standard",
                "encoding": "onehot"
            },
            "reasoning": "Tree-based models often work well with mixed data types"
        }'''
        mock_gemini_client.model.generate_content.return_value = mock_response

        designer = ExperimentDesigner(mock_gemini_client)
        spec = designer.design_experiment(
            data_profile=sample_data_profile,
            previous_results=[],
            task_type="regression",
            iteration=1,
        )

        assert isinstance(spec, ExperimentSpec)
        assert spec.experiment_name == "rf_experiment_1"
        assert spec.model_type == "RandomForestRegressor"
        assert spec.hypothesis == "RandomForest should capture non-linear patterns"
        assert designer._design_count == 1

    def test_design_experiment_with_previous_results(
        self, mock_gemini_client, sample_data_profile, sample_experiment_result, mock_genai
    ):
        """Test experiment design with previous results."""
        mock_response = Mock()
        mock_response.text = '''{
            "experiment_name": "gb_experiment_2",
            "hypothesis": "GradientBoosting may improve on RandomForest",
            "model_type": "GradientBoostingRegressor",
            "model_params": {"n_estimators": 100, "learning_rate": 0.1},
            "preprocessing": {
                "missing_values": "median",
                "scaling": "none",
                "encoding": "onehot"
            },
            "reasoning": "Trying boosting after seeing tree model performance"
        }'''
        mock_gemini_client.model.generate_content.return_value = mock_response

        designer = ExperimentDesigner(mock_gemini_client)
        spec = designer.design_experiment(
            data_profile=sample_data_profile,
            previous_results=[sample_experiment_result],
            task_type="regression",
            iteration=2,
        )

        assert spec.model_type == "GradientBoostingRegressor"
        # Verify previous results were included in prompt
        call_args = mock_gemini_client.model.generate_content.call_args
        prompt_text = str(call_args)
        assert "baseline" in prompt_text or "Previous" in prompt_text

    def test_design_experiment_fallback_on_invalid_json(
        self, mock_gemini_client, sample_data_profile, mock_genai
    ):
        """Test fallback when Gemini returns invalid JSON."""
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"
        mock_gemini_client.model.generate_content.return_value = mock_response

        designer = ExperimentDesigner(mock_gemini_client)
        spec = designer.design_experiment(
            data_profile=sample_data_profile,
            previous_results=[],
            task_type="regression",
            iteration=1,
        )

        # Should return fallback spec
        assert isinstance(spec, ExperimentSpec)
        assert "fallback" in spec.experiment_name
        assert "Fallback" in spec.reasoning

    def test_design_experiment_fallback_on_api_error(
        self, mock_gemini_client, sample_data_profile, mock_genai
    ):
        """Test fallback when Gemini API fails."""
        mock_gemini_client.model.generate_content.side_effect = Exception("API Error")

        designer = ExperimentDesigner(mock_gemini_client)
        spec = designer.design_experiment(
            data_profile=sample_data_profile,
            previous_results=[],
            task_type="classification",
            iteration=1,
        )

        # Should return fallback spec for classification
        assert isinstance(spec, ExperimentSpec)
        assert "fallback" in spec.experiment_name
        # Classification fallback models
        assert spec.model_type in [
            "RandomForestClassifier",
            "GradientBoostingClassifier",
            "LogisticRegression",
        ]


class TestConstraintsParser:
    """Test cases for constraints parsing."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    def test_parse_empty_constraints(self, mock_gemini_client):
        """Test parsing empty constraints."""
        designer = ExperimentDesigner(mock_gemini_client)
        parsed = designer.parse_constraints("")

        assert parsed.primary_metric is None
        assert parsed.preferred_models == []
        assert parsed.avoided_models == []

    def test_parse_metric_constraint(self, mock_gemini_client):
        """Test parsing metric constraints."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = """# Experiment Constraints

## Metrics
- Primary metric: RMSE
"""
        parsed = designer.parse_constraints(constraints)

        assert parsed.primary_metric == "rmse"

    def test_parse_model_preferences(self, mock_gemini_client):
        """Test parsing model preferences."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = """# Experiment Constraints

## Models
- Prefer tree-based models
- Avoid SVM
"""
        parsed = designer.parse_constraints(constraints)

        assert len(parsed.preferred_models) > 0
        assert "RandomForest" in parsed.preferred_models or any(
            "tree" in m.lower() for m in parsed.preferred_models
        )
        assert "SVC" in parsed.avoided_models or "SVR" in parsed.avoided_models

    def test_parse_preprocessing_hints(self, mock_gemini_client):
        """Test parsing preprocessing hints."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = """# Experiment Constraints

## Preprocessing
- Log-transform the target variable
- Handle missing values with imputation
"""
        parsed = designer.parse_constraints(constraints)

        assert len(parsed.preprocessing_hints) == 2
        assert "Log-transform the target variable" in parsed.preprocessing_hints

    def test_parse_termination_rules(self, mock_gemini_client):
        """Test parsing termination rules."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = """# Experiment Constraints

## Termination
- Stop if no improvement for 5 iterations
"""
        parsed = designer.parse_constraints(constraints)

        assert parsed.termination_rules.get("plateau_threshold") == 5

    def test_parse_full_constraints_file(self, mock_gemini_client):
        """Test parsing a complete constraints file."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = """# Experiment Constraints

## Metrics
- Primary metric: R2

## Models
- Prefer tree-based models
- Prefer boosting methods
- Avoid linear models

## Preprocessing
- Log-transform the target variable
- Use standard scaling

## Termination
- Stop if no improvement for 3 iterations
"""
        parsed = designer.parse_constraints(constraints)

        assert parsed.primary_metric == "r2"
        assert len(parsed.preferred_models) > 0
        assert "plateau_threshold" in parsed.termination_rules
        assert parsed.termination_rules["plateau_threshold"] == 3


class TestMetricSelection:
    """Test cases for metric selection."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    def test_select_metric_regression_default(self, mock_gemini_client):
        """Test default metric selection for regression."""
        designer = ExperimentDesigner(mock_gemini_client)
        metric = designer.select_primary_metric("regression")

        assert metric == "rmse"

    def test_select_metric_classification_default(self, mock_gemini_client):
        """Test default metric selection for classification."""
        designer = ExperimentDesigner(mock_gemini_client)
        metric = designer.select_primary_metric("classification")

        assert metric == "f1"

    def test_select_metric_from_constraints(self, mock_gemini_client):
        """Test metric selection from constraints."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = ParsedConstraints(primary_metric="mae")
        metric = designer.select_primary_metric("regression", constraints)

        assert metric == "mae"

    def test_select_metric_constraints_override_default(self, mock_gemini_client):
        """Test that constraints override default metric."""
        designer = ExperimentDesigner(mock_gemini_client)

        constraints = ParsedConstraints(primary_metric="accuracy")
        metric = designer.select_primary_metric("classification", constraints)

        assert metric == "accuracy"


class TestFallbackSpec:
    """Test cases for fallback experiment specs."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    def test_fallback_regression_models_rotate(self, mock_gemini_client):
        """Test that fallback regression models rotate."""
        designer = ExperimentDesigner(mock_gemini_client)

        models_seen = set()
        for i in range(1, 5):
            spec = designer._get_fallback_spec("regression", i)
            models_seen.add(spec.model_type)

        # Should see multiple different models
        assert len(models_seen) > 1

    def test_fallback_classification_models_rotate(self, mock_gemini_client):
        """Test that fallback classification models rotate."""
        designer = ExperimentDesigner(mock_gemini_client)

        models_seen = set()
        for i in range(1, 4):
            spec = designer._get_fallback_spec("classification", i)
            models_seen.add(spec.model_type)

        # Should see multiple different models
        assert len(models_seen) > 1

    def test_fallback_includes_error_reason(self, mock_gemini_client):
        """Test that fallback spec includes error reason."""
        designer = ExperimentDesigner(mock_gemini_client)

        spec = designer._get_fallback_spec(
            "regression", 1, error_reason="API rate limit exceeded"
        )

        assert "rate limit" in spec.reasoning.lower()

    def test_fallback_has_valid_structure(self, mock_gemini_client):
        """Test that fallback spec has all required fields."""
        designer = ExperimentDesigner(mock_gemini_client)

        spec = designer._get_fallback_spec("regression", 1)

        assert spec.experiment_name
        assert spec.hypothesis
        assert spec.model_type
        assert isinstance(spec.model_params, dict)
        assert isinstance(spec.preprocessing, PreprocessingConfig)


class TestSystemPrompt:
    """Test cases for system prompt."""

    def test_system_prompt_contains_key_elements(self):
        """Test that system prompt contains key elements."""
        prompt = EXPERIMENT_DESIGNER_SYSTEM_PROMPT

        assert "hypothesis" in prompt.lower()
        assert "model" in prompt.lower()
        assert "preprocessing" in prompt.lower()
        assert "json" in prompt.lower()

    def test_system_prompt_lists_available_models(self):
        """Test that system prompt lists available models."""
        prompt = EXPERIMENT_DESIGNER_SYSTEM_PROMPT

        assert "RandomForest" in prompt
        assert "GradientBoosting" in prompt
        assert "LinearRegression" in prompt or "LogisticRegression" in prompt
