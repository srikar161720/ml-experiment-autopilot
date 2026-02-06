"""Tests for the HypothesisGenerator component."""

import pytest
from unittest.mock import Mock, patch

from src.cognitive.hypothesis_generator import (
    HypothesisGenerator,
    HYPOTHESIS_GENERATOR_SYSTEM_PROMPT,
)
from src.cognitive.gemini_client import GeminiClient
from src.orchestration.state import (
    ExperimentResult,
    ExperimentState,
    ExperimentConfig,
    TaskType,
    AnalysisResult,
    MetricComparison,
    TrendPattern,
    Hypothesis,
    HypothesisSet,
)
from src.config import GeminiConfig


class TestHypothesisGenerator:
    """Test cases for HypothesisGenerator class."""

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
    def sample_experiment_config(self):
        """Create sample experiment configuration."""
        return ExperimentConfig(
            data_path="/data/test.csv",
            target_column="target",
            task_type=TaskType.REGRESSION,
            max_iterations=10,
            primary_metric="rmse",
            improvement_threshold=0.005,
        )

    @pytest.fixture
    def sample_analysis_result(self):
        """Create a sample AnalysisResult for testing."""
        return AnalysisResult(
            experiment_name="rf_experiment",
            iteration=1,
            success=True,
            primary_metric=MetricComparison(
                metric_name="rmse",
                current_value=0.40,
                baseline_value=0.50,
                best_value=0.50,
                change_from_baseline_pct=20.0,
                is_improvement=True,
                is_new_best=True,
            ),
            trend_pattern=TrendPattern.IMPROVING,
            key_observations=[
                "RandomForest improved RMSE by 20%",
                "Tree-based models outperform linear",
            ],
            reasoning="Ensemble methods captured non-linear patterns",
        )

    @pytest.fixture
    def failed_analysis_result(self):
        """Create a failed AnalysisResult for testing."""
        return AnalysisResult(
            experiment_name="failed_experiment",
            iteration=1,
            success=False,
            trend_pattern=TrendPattern.INITIAL,
            key_observations=["Experiment failed: convergence error"],
            reasoning="Experiment did not complete successfully.",
        )

    @pytest.fixture
    def sample_state(self, sample_experiment_config):
        """Create sample experiment state with baseline."""
        baseline = ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="LinearRegression",
            metrics={"rmse": 0.50, "r2": 0.70},
            hypothesis="Establish baseline",
            success=True,
        )

        state = ExperimentState(config=sample_experiment_config)
        state.experiments = [baseline]
        state.current_iteration = 1
        state.best_metric = 0.50
        state.best_experiment = "baseline"
        return state

    def test_init(self, mock_gemini_client):
        """Test HypothesisGenerator initialization."""
        generator = HypothesisGenerator(mock_gemini_client)

        assert generator.client == mock_gemini_client
        assert generator._generation_count == 0

    def test_generate_success(
        self, mock_gemini_client, sample_state, sample_analysis_result, mock_genai
    ):
        """Test successful hypothesis generation."""
        mock_response = Mock()
        mock_response.text = '''{
            "analysis_summary": "RandomForest showed 20% improvement",
            "exploration_vs_exploitation": "balanced",
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "Try GradientBoosting with tuned learning rate",
                    "rationale": "Ensemble methods are working well",
                    "suggested_model": "GradientBoostingRegressor",
                    "suggested_params": {"n_estimators": 200, "learning_rate": 0.05},
                    "confidence_score": 0.7,
                    "priority": 1
                },
                {
                    "hypothesis_id": "h2",
                    "statement": "Try feature engineering with polynomial features",
                    "rationale": "May capture additional non-linear patterns",
                    "suggested_model": "RandomForestRegressor",
                    "suggested_params": {"n_estimators": 300},
                    "confidence_score": 0.5,
                    "priority": 2
                }
            ],
            "reasoning": "Building on successful tree-based approach"
        }'''
        mock_gemini_client.model.generate_content.return_value = mock_response

        generator = HypothesisGenerator(mock_gemini_client)
        result = generator.generate(sample_analysis_result, sample_state)

        assert isinstance(result, HypothesisSet)
        assert result.iteration == 1
        assert len(result.hypotheses) == 2
        assert result.hypotheses[0].priority == 1
        assert result.hypotheses[0].suggested_model == "GradientBoostingRegressor"
        assert result.exploration_vs_exploitation == "balanced"
        assert generator._generation_count == 1

    def test_generate_failed_analysis(
        self, mock_gemini_client, sample_state, failed_analysis_result
    ):
        """Test hypothesis generation for a failed experiment."""
        generator = HypothesisGenerator(mock_gemini_client)
        result = generator.generate(failed_analysis_result, sample_state)

        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].priority == 1
        assert "retry" in result.hypotheses[0].statement.lower() or "simpler" in result.hypotheses[0].statement.lower()
        assert result.exploration_vs_exploitation == "exploit"

    def test_generate_fallback_on_api_error(
        self, mock_gemini_client, sample_state, sample_analysis_result, mock_genai
    ):
        """Test fallback when Gemini API fails."""
        mock_gemini_client.model.generate_content.side_effect = Exception("API Error")

        generator = HypothesisGenerator(mock_gemini_client)
        result = generator.generate(sample_analysis_result, sample_state)

        assert isinstance(result, HypothesisSet)
        assert "Fallback" in result.reasoning
        assert len(result.hypotheses) == 2

    def test_generate_increments_count(
        self, mock_gemini_client, sample_state, failed_analysis_result
    ):
        """Test that generation count increments correctly."""
        generator = HypothesisGenerator(mock_gemini_client)

        assert generator.get_generation_count() == 0
        generator.generate(failed_analysis_result, sample_state)
        assert generator.get_generation_count() == 1
        generator.generate(failed_analysis_result, sample_state)
        assert generator.get_generation_count() == 2


class TestHypothesisPrompt:
    """Test hypothesis prompt building."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def sample_config(self):
        """Create sample experiment config."""
        return ExperimentConfig(
            data_path="/data/test.csv",
            target_column="target",
            task_type=TaskType.REGRESSION,
            primary_metric="rmse",
            improvement_threshold=0.005,
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for prompt testing."""
        return AnalysisResult(
            experiment_name="test_exp",
            iteration=2,
            success=True,
            primary_metric=MetricComparison(
                metric_name="rmse",
                current_value=0.40,
                baseline_value=0.50,
                change_from_baseline_pct=20.0,
                is_improvement=True,
                is_new_best=True,
            ),
            trend_pattern=TrendPattern.IMPROVING,
            key_observations=["Model improved significantly"],
            reasoning="Good progress",
        )

    def test_prompt_contains_analysis_data(
        self, mock_gemini_client, sample_config, sample_analysis
    ):
        """Test that prompt includes analysis observations."""
        state = ExperimentState(config=sample_config)
        generator = HypothesisGenerator(mock_gemini_client)

        prompt = generator._build_hypothesis_prompt(sample_analysis, state)

        assert "Model improved significantly" in prompt
        assert "improving" in prompt
        assert "rmse" in prompt

    def test_prompt_contains_experiment_history(
        self, mock_gemini_client, sample_config, sample_analysis
    ):
        """Test that prompt includes experiment history."""
        state = ExperimentState(config=sample_config)
        state.experiments = [
            ExperimentResult(
                experiment_name="baseline",
                iteration=0,
                model_type="LinearRegression",
                metrics={"rmse": 0.50},
                success=True,
            )
        ]

        generator = HypothesisGenerator(mock_gemini_client)
        prompt = generator._build_hypothesis_prompt(sample_analysis, state)

        assert "baseline" in prompt
        assert "LinearRegression" in prompt

    def test_prompt_requests_json_schema(
        self, mock_gemini_client, sample_config, sample_analysis
    ):
        """Test that prompt specifies required JSON schema."""
        state = ExperimentState(config=sample_config)
        generator = HypothesisGenerator(mock_gemini_client)

        prompt = generator._build_hypothesis_prompt(sample_analysis, state)

        assert "hypothesis_id" in prompt
        assert "confidence_score" in prompt
        assert "priority" in prompt
        assert "suggested_model" in prompt

    def test_prompt_includes_strategy_hint(
        self, mock_gemini_client, sample_config, sample_analysis
    ):
        """Test that prompt includes strategy guidance."""
        state = ExperimentState(config=sample_config)
        generator = HypothesisGenerator(mock_gemini_client)

        prompt = generator._build_hypothesis_prompt(sample_analysis, state)

        assert "Strategy Guidance" in prompt


class TestHypothesisParsing:
    """Test hypothesis response parsing."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for parsing tests."""
        return AnalysisResult(
            experiment_name="test_exp",
            iteration=3,
            success=True,
            trend_pattern=TrendPattern.IMPROVING,
        )

    def test_parse_valid_response(self, mock_gemini_client, sample_analysis):
        """Test parsing a well-formed response."""
        generator = HypothesisGenerator(mock_gemini_client)

        response = {
            "analysis_summary": "Test summary",
            "exploration_vs_exploitation": "exploit",
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "Try XGBoost",
                    "rationale": "Better gradient boosting",
                    "suggested_model": "XGBRegressor",
                    "suggested_params": {"n_estimators": 100},
                    "confidence_score": 0.8,
                    "priority": 1,
                }
            ],
            "reasoning": "Test reasoning",
        }

        result = generator._parse_hypothesis_response(response, sample_analysis)

        assert isinstance(result, HypothesisSet)
        assert result.iteration == 3
        assert result.analysis_summary == "Test summary"
        assert len(result.hypotheses) == 1
        assert result.hypotheses[0].statement == "Try XGBoost"
        assert result.hypotheses[0].confidence_score == 0.8

    def test_parse_clamps_confidence(self, mock_gemini_client, sample_analysis):
        """Test that confidence values are clamped to [0, 1]."""
        generator = HypothesisGenerator(mock_gemini_client)

        response = {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "Test",
                    "rationale": "Test",
                    "confidence_score": 1.5,  # Over max
                    "priority": 1,
                },
                {
                    "hypothesis_id": "h2",
                    "statement": "Test2",
                    "rationale": "Test2",
                    "confidence_score": -0.2,  # Under min
                    "priority": 2,
                },
            ],
        }

        result = generator._parse_hypothesis_response(response, sample_analysis)

        assert result.hypotheses[0].confidence_score == 1.0
        assert result.hypotheses[1].confidence_score == 0.0

    def test_parse_clamps_priority(self, mock_gemini_client, sample_analysis):
        """Test that priority values are clamped to [1, 3]."""
        generator = HypothesisGenerator(mock_gemini_client)

        response = {
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "statement": "Test",
                    "rationale": "Test",
                    "confidence_score": 0.5,
                    "priority": 0,  # Under min
                },
                {
                    "hypothesis_id": "h2",
                    "statement": "Test2",
                    "rationale": "Test2",
                    "confidence_score": 0.5,
                    "priority": 5,  # Over max
                },
            ],
        }

        result = generator._parse_hypothesis_response(response, sample_analysis)

        assert result.hypotheses[0].priority == 1
        assert result.hypotheses[1].priority == 3

    def test_parse_empty_hypotheses(self, mock_gemini_client, sample_analysis):
        """Test handling of empty hypotheses list."""
        generator = HypothesisGenerator(mock_gemini_client)

        response = {
            "analysis_summary": "No clear direction",
            "hypotheses": [],
            "reasoning": "No hypotheses generated",
        }

        result = generator._parse_hypothesis_response(response, sample_analysis)

        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 0
        assert result.get_top_hypothesis() is None


class TestFallbackHypotheses:
    """Test fallback hypothesis generation."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def regression_config(self):
        """Create regression config."""
        return ExperimentConfig(
            data_path="/data/test.csv",
            target_column="target",
            task_type=TaskType.REGRESSION,
            primary_metric="rmse",
        )

    @pytest.fixture
    def classification_config(self):
        """Create classification config."""
        return ExperimentConfig(
            data_path="/data/test.csv",
            target_column="target",
            task_type=TaskType.CLASSIFICATION,
            primary_metric="accuracy",
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create a sample analysis for fallback tests."""
        return AnalysisResult(
            experiment_name="test_exp",
            iteration=1,
            success=True,
            trend_pattern=TrendPattern.INITIAL,
        )

    def test_fallback_regression(
        self, mock_gemini_client, regression_config, sample_analysis
    ):
        """Test fallback hypotheses for regression task."""
        state = ExperimentState(config=regression_config)
        generator = HypothesisGenerator(mock_gemini_client)

        result = generator._get_fallback_hypotheses(sample_analysis, state)

        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 2
        # Check that regression models are suggested
        models = [h.suggested_model for h in result.hypotheses]
        regression_keywords = ["Regressor", "Ridge", "Lasso", "Linear"]
        assert any(
            any(kw in model for kw in regression_keywords)
            for model in models
        )

    def test_fallback_classification(
        self, mock_gemini_client, classification_config, sample_analysis
    ):
        """Test fallback hypotheses for classification task."""
        state = ExperimentState(config=classification_config)
        generator = HypothesisGenerator(mock_gemini_client)

        result = generator._get_fallback_hypotheses(sample_analysis, state)

        assert isinstance(result, HypothesisSet)
        assert len(result.hypotheses) == 2
        # Check that classification models are suggested
        models = [h.suggested_model for h in result.hypotheses]
        classification_keywords = ["Classifier", "Logistic"]
        assert any(
            any(kw in model for kw in classification_keywords)
            for model in models
        )

    def test_fallback_has_valid_structure(
        self, mock_gemini_client, regression_config, sample_analysis
    ):
        """Test all required fields are present in fallback hypotheses."""
        state = ExperimentState(config=regression_config)
        generator = HypothesisGenerator(mock_gemini_client)

        result = generator._get_fallback_hypotheses(sample_analysis, state)

        for hypothesis in result.hypotheses:
            assert hypothesis.hypothesis_id
            assert hypothesis.statement
            assert hypothesis.rationale
            assert hypothesis.suggested_model
            assert 0.0 <= hypothesis.confidence_score <= 1.0
            assert 1 <= hypothesis.priority <= 3

    def test_fallback_rotates_suggestions(
        self, mock_gemini_client, regression_config
    ):
        """Test that different iterations get different suggestions."""
        state = ExperimentState(config=regression_config)
        generator = HypothesisGenerator(mock_gemini_client)

        analysis_iter1 = AnalysisResult(
            experiment_name="exp1", iteration=1, success=True,
            trend_pattern=TrendPattern.INITIAL,
        )
        analysis_iter2 = AnalysisResult(
            experiment_name="exp2", iteration=2, success=True,
            trend_pattern=TrendPattern.INITIAL,
        )

        result1 = generator._get_fallback_hypotheses(analysis_iter1, state)
        result2 = generator._get_fallback_hypotheses(analysis_iter2, state)

        # Primary hypotheses should differ across iterations
        assert result1.hypotheses[0].suggested_model != result2.hypotheses[0].suggested_model


class TestSystemPrompt:
    """Test system prompt content."""

    def test_prompt_contains_hypothesis_instructions(self):
        """Test prompt explains hypothesis generation."""
        prompt = HYPOTHESIS_GENERATOR_SYSTEM_PROMPT.lower()
        assert "hypothesis" in prompt
        assert "testable" in prompt

    def test_prompt_defines_confidence_scoring(self):
        """Test prompt includes confidence scoring guide."""
        prompt = HYPOTHESIS_GENERATOR_SYSTEM_PROMPT.lower()
        assert "confidence" in prompt
        assert "0.8" in prompt or "0.8-1.0" in HYPOTHESIS_GENERATOR_SYSTEM_PROMPT

    def test_prompt_defines_priority_levels(self):
        """Test prompt defines priority levels."""
        prompt = HYPOTHESIS_GENERATOR_SYSTEM_PROMPT
        assert "1:" in prompt or "priority" in prompt.lower()
        assert "2:" in prompt
        assert "3:" in prompt

    def test_prompt_mentions_json(self):
        """Test prompt requests JSON output."""
        assert "json" in HYPOTHESIS_GENERATOR_SYSTEM_PROMPT.lower()

    def test_prompt_mentions_exploration_exploitation(self):
        """Test prompt discusses exploration vs exploitation."""
        prompt = HYPOTHESIS_GENERATOR_SYSTEM_PROMPT.lower()
        assert "exploration" in prompt or "explore" in prompt
        assert "exploitation" in prompt or "exploit" in prompt
