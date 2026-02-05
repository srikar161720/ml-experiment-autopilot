"""Tests for the ResultsAnalyzer component."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.cognitive.results_analyzer import (
    ResultsAnalyzer,
    RESULTS_ANALYZER_SYSTEM_PROMPT,
)
from src.cognitive.gemini_client import GeminiClient
from src.orchestration.state import (
    ExperimentResult,
    ExperimentState,
    ExperimentConfig,
    DataProfile,
    TaskType,
    AnalysisResult,
    MetricComparison,
    TrendPattern,
    PreprocessingConfig,
)
from src.config import GeminiConfig


class TestResultsAnalyzer:
    """Test cases for ResultsAnalyzer class."""

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
    def baseline_result(self):
        """Create a baseline experiment result."""
        return ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="LinearRegression",
            metrics={"rmse": 0.50, "r2": 0.70, "mae": 0.40},
            hypothesis="Establish baseline",
            success=True,
        )

    @pytest.fixture
    def improved_result(self):
        """Create an improved experiment result."""
        return ExperimentResult(
            experiment_name="rf_experiment",
            iteration=1,
            model_type="RandomForestRegressor",
            metrics={"rmse": 0.40, "r2": 0.80, "mae": 0.30},
            hypothesis="Test RandomForest",
            success=True,
        )

    @pytest.fixture
    def degraded_result(self):
        """Create a degraded experiment result."""
        return ExperimentResult(
            experiment_name="bad_experiment",
            iteration=2,
            model_type="DecisionTreeRegressor",
            metrics={"rmse": 0.60, "r2": 0.60, "mae": 0.50},
            hypothesis="Test DecisionTree",
            success=True,
        )

    @pytest.fixture
    def failed_result(self):
        """Create a failed experiment result."""
        return ExperimentResult(
            experiment_name="failed_experiment",
            iteration=1,
            model_type="SVR",
            success=False,
            error_message="Training failed: convergence error",
        )

    @pytest.fixture
    def sample_state(self, sample_experiment_config, baseline_result):
        """Create sample experiment state with baseline."""
        state = ExperimentState(config=sample_experiment_config)
        state.experiments = [baseline_result]
        state.current_iteration = 1
        state.best_metric = 0.50
        state.best_experiment = "baseline"
        return state

    def test_init(self, mock_gemini_client):
        """Test ResultsAnalyzer initialization."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        assert analyzer.client == mock_gemini_client
        assert analyzer._analysis_count == 0

    def test_analyze_success(self, mock_gemini_client, sample_state, improved_result, mock_genai):
        """Test successful results analysis."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = '''{
            "key_observations": [
                "RandomForest improved RMSE by 20%",
                "Tree-based models outperform linear"
            ],
            "reasoning": "The ensemble method captured non-linear patterns"
        }'''
        mock_gemini_client.model.generate_content.return_value = mock_response

        analyzer = ResultsAnalyzer(mock_gemini_client)
        analysis = analyzer.analyze(improved_result, sample_state)

        assert isinstance(analysis, AnalysisResult)
        assert analysis.experiment_name == "rf_experiment"
        assert analysis.success is True
        assert analysis.primary_metric is not None
        assert analyzer._analysis_count == 1

    def test_analyze_failed_experiment(self, mock_gemini_client, sample_state, failed_result):
        """Test analysis of a failed experiment."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        analysis = analyzer.analyze(failed_result, sample_state)

        assert analysis.success is False
        assert "failed" in analysis.key_observations[0].lower()
        assert "convergence" in analysis.key_observations[0].lower()

    def test_analyze_fallback_on_api_error(
        self, mock_gemini_client, sample_state, improved_result, mock_genai
    ):
        """Test fallback when Gemini API fails."""
        mock_gemini_client.model.generate_content.side_effect = Exception("API Error")

        analyzer = ResultsAnalyzer(mock_gemini_client)
        analysis = analyzer.analyze(improved_result, sample_state)

        # Should return fallback analysis without crashing
        assert isinstance(analysis, AnalysisResult)
        assert "Fallback" in analysis.reasoning
        assert len(analysis.key_observations) > 0


class TestMetricComparison:
    """Test metric comparison logic."""

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

    def test_lower_is_better_rmse(self, mock_gemini_client, sample_config):
        """Test correct comparison for RMSE (lower is better)."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        baseline = ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="Linear",
            metrics={"rmse": 0.50},
            success=True,
        )

        improved = ExperimentResult(
            experiment_name="improved",
            iteration=1,
            model_type="RF",
            metrics={"rmse": 0.40},  # Lower = better
            success=True,
        )

        state = ExperimentState(config=sample_config)
        state.experiments = [baseline]
        state.best_metric = 0.50

        comparison = analyzer._compute_metric_comparison(improved, state, "rmse")

        assert comparison.current_value == 0.40
        assert comparison.baseline_value == 0.50
        assert comparison.is_improvement is True
        assert comparison.change_from_baseline_pct > 0  # Positive = improvement

    def test_higher_is_better_r2(self, mock_gemini_client, sample_config):
        """Test correct comparison for R2 (higher is better)."""
        sample_config.primary_metric = "r2"
        analyzer = ResultsAnalyzer(mock_gemini_client)

        baseline = ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="Linear",
            metrics={"r2": 0.70},
            success=True,
        )

        improved = ExperimentResult(
            experiment_name="improved",
            iteration=1,
            model_type="RF",
            metrics={"r2": 0.85},  # Higher = better
            success=True,
        )

        state = ExperimentState(config=sample_config)
        state.experiments = [baseline]
        state.best_metric = 0.70

        comparison = analyzer._compute_metric_comparison(improved, state, "r2")

        assert comparison.current_value == 0.85
        assert comparison.is_improvement is True
        assert comparison.change_from_baseline_pct > 0

    def test_percentage_change_calculation(self, mock_gemini_client, sample_config):
        """Test percentage change calculations."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        baseline = ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="Linear",
            metrics={"rmse": 0.50},
            success=True,
        )

        improved = ExperimentResult(
            experiment_name="improved",
            iteration=1,
            model_type="RF",
            metrics={"rmse": 0.45},  # 10% improvement
            success=True,
        )

        state = ExperimentState(config=sample_config)
        state.experiments = [baseline]
        state.best_metric = 0.50

        comparison = analyzer._compute_metric_comparison(improved, state, "rmse")

        # 10% improvement: (0.50 - 0.45) / 0.50 = 0.10 = 10%
        assert comparison.change_from_baseline_pct == pytest.approx(10.0, rel=0.01)

    def test_is_new_best(self, mock_gemini_client, sample_config):
        """Test detection of new best metric."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        baseline = ExperimentResult(
            experiment_name="baseline",
            iteration=0,
            model_type="Linear",
            metrics={"rmse": 0.50},
            success=True,
        )

        new_best = ExperimentResult(
            experiment_name="new_best",
            iteration=1,
            model_type="RF",
            metrics={"rmse": 0.35},  # Significantly better
            success=True,
        )

        state = ExperimentState(config=sample_config)
        state.experiments = [baseline]
        state.best_metric = 0.50

        comparison = analyzer._compute_metric_comparison(new_best, state, "rmse")

        assert comparison.is_new_best is True


class TestTrendDetection:
    """Test trend pattern detection."""

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

    def test_detect_improving_trend(self, mock_gemini_client, sample_config):
        """Test detection of improving trend."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        # Create experiments with consistent improvement
        experiments = [
            ExperimentResult(
                experiment_name=f"exp_{i}",
                iteration=i,
                model_type="Model",
                metrics={"rmse": 0.50 - i * 0.05},  # Improving
                success=True,
            )
            for i in range(4)
        ]

        state = ExperimentState(config=sample_config)
        state.experiments = experiments

        trend = analyzer._detect_trend_pattern(state, "rmse")

        assert trend == TrendPattern.IMPROVING

    def test_detect_degrading_trend(self, mock_gemini_client, sample_config):
        """Test detection of degrading trend."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        # Create experiments with consistent degradation
        experiments = [
            ExperimentResult(
                experiment_name=f"exp_{i}",
                iteration=i,
                model_type="Model",
                metrics={"rmse": 0.30 + i * 0.05},  # Degrading
                success=True,
            )
            for i in range(4)
        ]

        state = ExperimentState(config=sample_config)
        state.experiments = experiments

        trend = analyzer._detect_trend_pattern(state, "rmse")

        assert trend == TrendPattern.DEGRADING

    def test_detect_plateau_trend(self, mock_gemini_client, sample_config):
        """Test detection of plateau."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        # Create experiments with minimal change
        experiments = [
            ExperimentResult(
                experiment_name=f"exp_{i}",
                iteration=i,
                model_type="Model",
                metrics={"rmse": 0.40 + i * 0.001},  # Very small change
                success=True,
            )
            for i in range(4)
        ]

        state = ExperimentState(config=sample_config)
        state.experiments = experiments

        trend = analyzer._detect_trend_pattern(state, "rmse")

        assert trend == TrendPattern.PLATEAU

    def test_initial_trend_insufficient_data(self, mock_gemini_client, sample_config):
        """Test initial pattern when not enough data."""
        analyzer = ResultsAnalyzer(mock_gemini_client)

        # Only 2 experiments - not enough for trend
        experiments = [
            ExperimentResult(
                experiment_name=f"exp_{i}",
                iteration=i,
                model_type="Model",
                metrics={"rmse": 0.40},
                success=True,
            )
            for i in range(2)
        ]

        state = ExperimentState(config=sample_config)
        state.experiments = experiments

        trend = analyzer._detect_trend_pattern(state, "rmse")

        assert trend == TrendPattern.INITIAL


class TestIsLowerBetter:
    """Test metric direction detection."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient."""
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    def test_rmse_lower_is_better(self, mock_gemini_client):
        """Test RMSE is correctly identified as lower-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("rmse") is True
        assert analyzer._is_lower_better("RMSE") is True

    def test_mse_lower_is_better(self, mock_gemini_client):
        """Test MSE is correctly identified as lower-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("mse") is True

    def test_mae_lower_is_better(self, mock_gemini_client):
        """Test MAE is correctly identified as lower-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("mae") is True

    def test_r2_higher_is_better(self, mock_gemini_client):
        """Test R2 is correctly identified as higher-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("r2") is False

    def test_accuracy_higher_is_better(self, mock_gemini_client):
        """Test accuracy is correctly identified as higher-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("accuracy") is False

    def test_f1_higher_is_better(self, mock_gemini_client):
        """Test F1 is correctly identified as higher-is-better."""
        analyzer = ResultsAnalyzer(mock_gemini_client)
        assert analyzer._is_lower_better("f1") is False


class TestSystemPrompt:
    """Test system prompt content."""

    def test_prompt_contains_metric_interpretation(self):
        """Test prompt explains metric directions."""
        prompt = RESULTS_ANALYZER_SYSTEM_PROMPT.lower()
        assert "lower is better" in prompt
        assert "higher is better" in prompt

    def test_prompt_contains_pattern_definitions(self):
        """Test prompt defines patterns."""
        prompt = RESULTS_ANALYZER_SYSTEM_PROMPT.lower()
        assert "improving" in prompt
        assert "degrading" in prompt
        assert "plateau" in prompt

    def test_prompt_mentions_json(self):
        """Test prompt requests JSON output."""
        assert "json" in RESULTS_ANALYZER_SYSTEM_PROMPT.lower()
