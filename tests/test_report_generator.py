"""Tests for the ReportGenerator component."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.cognitive.report_generator import (
    ReportGenerator,
    REPORT_GENERATOR_SYSTEM_PROMPT,
    SECTION_DELIMITER,
)
from src.cognitive.gemini_client import GeminiClient, GeminiInvalidResponseError
from src.orchestration.state import (
    ExperimentResult,
    ExperimentState,
    ExperimentConfig,
    DataProfile,
    TaskType,
)
from src.config import GeminiConfig


# ── Shared helpers ──────────────────────────────────────────────────────


def _make_delimited_response() -> str:
    """Build a valid 4-section delimited Gemini response for mocking."""
    sections = [
        (
            "This is the executive summary. The experiment session explored "
            "multiple models on the house_prices_train dataset for regression. "
            "The best result was achieved by rf_tuned with RMSE of 0.30, "
            "representing a 40.0% improvement over the linear baseline."
        ),
        (
            "The methodology followed an iterative, hypothesis-driven approach. "
            "Starting with a LinearRegression baseline, the system explored "
            "tree-based models including RandomForest variants.\n\n"
            "Each iteration analyzed the previous results to design the next experiment."
        ),
        (
            "- Tree-based models consistently outperformed the linear baseline.\n"
            "- Tuning n_estimators to 200 yielded the best RMSE.\n"
            "- The plateau was reached after iteration 2."
        ),
        (
            "- Consider trying XGBoost or LightGBM for further improvement.\n"
            "- Feature engineering may unlock additional performance.\n"
            "- Evaluate the best model on a held-out test set."
        ),
    ]
    return f"\n{SECTION_DELIMITER}\n".join(sections)


def _make_sample_data_profile() -> DataProfile:
    """Create a realistic DataProfile fixture."""
    return DataProfile(
        n_rows=1460,
        n_columns=81,
        columns=["SalePrice", "OverallQual", "GrLivArea", "Neighborhood"],
        column_types={"SalePrice": "float64", "OverallQual": "int64",
                       "GrLivArea": "int64", "Neighborhood": "object"},
        numeric_columns=["SalePrice", "OverallQual", "GrLivArea"],
        categorical_columns=["Neighborhood"],
        target_column="SalePrice",
        target_type="continuous",
        missing_values={"OverallQual": 0, "GrLivArea": 0, "Neighborhood": 5},
        missing_percentages={"OverallQual": 0.0, "GrLivArea": 0.0, "Neighborhood": 0.34},
        numeric_stats={
            "SalePrice": {"mean": 180921.0, "std": 79442.0, "min": 34900.0, "max": 755000.0},
        },
        categorical_stats={
            "Neighborhood": {"nunique": 25, "top": "NAmes"},
        },
        target_stats={"mean": 180921.0, "std": 79442.0, "min": 34900.0, "max": 755000.0},
    )


def _make_sample_experiment_config() -> ExperimentConfig:
    """Create a regression ExperimentConfig."""
    return ExperimentConfig(
        data_path="/data/house_prices_train.csv",
        target_column="SalePrice",
        task_type=TaskType.REGRESSION,
        max_iterations=10,
        primary_metric="rmse",
        improvement_threshold=0.005,
    )


def _make_sample_state() -> ExperimentState:
    """Create a populated ExperimentState with 3 experiments."""
    config = _make_sample_experiment_config()

    baseline = ExperimentResult(
        experiment_name="baseline",
        iteration=0,
        model_type="LinearRegression",
        metrics={"rmse": 0.50, "r2": 0.70},
        hypothesis="Establish baseline with simple model",
        success=True,
        execution_time=2.5,
    )
    rf_basic = ExperimentResult(
        experiment_name="rf_basic",
        iteration=1,
        model_type="RandomForestRegressor",
        model_params={"n_estimators": 100, "max_depth": 10},
        metrics={"rmse": 0.35, "r2": 0.82},
        hypothesis="RandomForest should capture non-linear patterns",
        reasoning="Tree-based models may outperform linear on this dataset",
        success=True,
        execution_time=5.1,
    )
    rf_tuned = ExperimentResult(
        experiment_name="rf_tuned",
        iteration=2,
        model_type="RandomForestRegressor",
        model_params={"n_estimators": 200, "max_depth": 15},
        metrics={"rmse": 0.30, "r2": 0.87},
        hypothesis="Increasing n_estimators and max_depth improves RF",
        reasoning="Previous RF showed promise, tuning hyperparameters",
        success=True,
        execution_time=8.3,
    )

    state = ExperimentState(config=config)
    state.experiments = [baseline, rf_basic, rf_tuned]
    state.current_iteration = 3
    state.best_metric = 0.30
    state.best_experiment = "rf_tuned"
    state.termination_reason = "Performance plateau detected"
    state.data_profile = _make_sample_data_profile()

    return state


# ── TestReportGenerator ─────────────────────────────────────────────────


class TestReportGenerator:
    """Main integration tests for ReportGenerator."""

    @pytest.fixture
    def mock_config(self):
        return GeminiConfig(
            api_key="test_key",
            model="test-model",
            temperature=1.0,
            max_retries=3,
            retry_delay=0.1,
        )

    @pytest.fixture
    def mock_genai(self):
        with patch("src.cognitive.gemini_client.genai") as mock:
            yield mock

    @pytest.fixture
    def mock_gemini_client(self, mock_config, mock_genai):
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        return GeminiClient(mock_config)

    @pytest.fixture
    def sample_state(self):
        return _make_sample_state()

    def test_init(self, mock_gemini_client):
        """Test ReportGenerator stores the client."""
        generator = ReportGenerator(mock_gemini_client)
        assert generator.client is mock_gemini_client

    def test_generate_success(
        self, mock_gemini_client, sample_state, temp_output_dir
    ):
        """Test full report generation with mocked Gemini."""
        mock_response = Mock()
        mock_response.text = _make_delimited_response()
        mock_gemini_client.model.generate_content.return_value = mock_response

        generator = ReportGenerator(mock_gemini_client)
        report_path = generator.generate(sample_state, temp_output_dir)

        # File was created in reports/ subdirectory
        assert report_path.exists()
        assert report_path.parent.name == "reports"
        assert report_path.suffix == ".md"

        content = report_path.read_text()

        # All major headers present
        assert "# ML Experiment Report:" in content
        assert "## Executive Summary" in content
        assert "## Dataset Overview" in content
        assert "## Methodology" in content
        assert "## Experiment Results" in content
        assert "### Best Model" in content
        assert "## Key Insights" in content
        assert "## Recommendations" in content
        assert "## Appendix" in content

        # Experiment data from state appears
        assert "baseline" in content
        assert "rf_tuned" in content

    def test_generate_fallback_on_api_error(
        self, mock_gemini_client, sample_state, temp_output_dir
    ):
        """Test report generation falls back gracefully on Gemini failure."""
        mock_gemini_client.model.generate_content.side_effect = Exception(
            "API Error"
        )

        generator = ReportGenerator(mock_gemini_client)
        report_path = generator.generate(sample_state, temp_output_dir)

        assert report_path.exists()
        content = report_path.read_text()

        # Fallback narrative still produces valid report
        assert "## Executive Summary" in content
        assert "rf_tuned" in content
        assert "regression" in content.lower()

    def test_generate_no_successful_experiments(
        self, mock_gemini_client, sample_state, temp_output_dir
    ):
        """Test report generation when all experiments failed."""
        # Replace experiments with failed ones
        failed = ExperimentResult(
            experiment_name="failed_exp",
            iteration=1,
            model_type="BadModel",
            success=False,
            error_message="Convergence error",
            hypothesis="This will fail",
        )
        sample_state.experiments = [failed]
        sample_state.best_metric = None
        sample_state.best_experiment = None

        # Gemini will fail too since there's no meaningful data
        mock_gemini_client.model.generate_content.side_effect = Exception("err")

        generator = ReportGenerator(mock_gemini_client)
        report_path = generator.generate(sample_state, temp_output_dir)

        assert report_path.exists()
        content = report_path.read_text()
        assert "FAILED" in content
        assert "No successful experiments" in content

    def test_generate_single_experiment(
        self, mock_gemini_client, sample_state, temp_output_dir
    ):
        """Test report generation with only a baseline experiment."""
        # Keep only baseline
        sample_state.experiments = [sample_state.experiments[0]]
        sample_state.current_iteration = 1

        mock_response = Mock()
        mock_response.text = _make_delimited_response()
        mock_gemini_client.model.generate_content.return_value = mock_response

        generator = ReportGenerator(mock_gemini_client)
        report_path = generator.generate(sample_state, temp_output_dir)

        assert report_path.exists()
        content = report_path.read_text()
        assert "baseline" in content


# ── TestLocalSections ───────────────────────────────────────────────────


class TestLocalSections:
    """Test locally-built Markdown sections."""

    @pytest.fixture
    def mock_gemini_client(self):
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def generator(self, mock_gemini_client):
        return ReportGenerator(mock_gemini_client)

    @pytest.fixture
    def sample_state(self):
        return _make_sample_state()

    def test_build_dataset_section(self, generator, sample_state):
        """Test dataset section contains profile stats."""
        section = generator._build_dataset_section(sample_state)

        assert "1,460" in section  # formatted row count
        assert "81" in section  # column count
        assert "SalePrice" in section  # target column
        assert "continuous" in section  # target type
        assert "Numeric Features" in section
        assert "Categorical Features" in section

    def test_build_dataset_section_no_profile(self, generator, sample_state):
        """Test dataset section with no data profile."""
        sample_state.data_profile = None
        section = generator._build_dataset_section(sample_state)
        assert "not available" in section.lower()

    def test_build_results_table(self, generator, sample_state):
        """Test results table includes all experiments."""
        table = generator._build_results_table(sample_state)

        # Header row present
        assert "RMSE" in table  # primary metric header
        assert "Status" in table

        # All experiments in table
        assert "baseline" in table
        assert "rf_basic" in table
        assert "rf_tuned" in table

        # Check success status
        assert "OK" in table

    def test_build_results_table_with_failures(self, generator, sample_state):
        """Test results table correctly shows failed experiments."""
        failed = ExperimentResult(
            experiment_name="failed_exp",
            iteration=3,
            model_type="BadModel",
            success=False,
            error_message="Error",
            hypothesis="Will fail",
        )
        sample_state.experiments.append(failed)

        table = generator._build_results_table(sample_state)

        assert "FAILED" in table
        assert "--" in table  # metric placeholder for failed

    def test_build_best_model_section(self, generator, sample_state):
        """Test best model section contains correct details."""
        section = generator._build_best_model_section(sample_state)

        assert "RandomForestRegressor" in section
        assert "rf_tuned" in section
        assert "0.30" in section  # best metric value
        assert "n_estimators" in section  # hyperparameter
        assert "Increasing n_estimators" in section  # hypothesis

    def test_build_best_model_section_no_best(self, generator, sample_state):
        """Test best model section when no best experiment exists."""
        sample_state.best_experiment = None
        sample_state.best_metric = None

        section = generator._build_best_model_section(sample_state)
        assert "No successful experiments" in section

    def test_build_appendix(self, generator, sample_state):
        """Test appendix contains all experiment details."""
        appendix = generator._build_appendix_section(sample_state)

        # All experiment names present
        assert "baseline" in appendix
        assert "rf_basic" in appendix
        assert "rf_tuned" in appendix

        # Model types present
        assert "LinearRegression" in appendix
        assert "RandomForestRegressor" in appendix

        # Success status
        assert "Success" in appendix

    def test_build_run_metadata(self, generator, sample_state):
        """Test run metadata footer."""
        metadata = generator._build_run_metadata(sample_state)

        assert sample_state.session_id in metadata
        assert "Gemini 3" in metadata
        assert "ML Experiment Autopilot" in metadata

    def test_extract_dataset_name(self, generator, sample_state):
        """Test dataset name extraction from path."""
        name = generator._extract_dataset_name(sample_state)
        assert name == "house_prices_train"


# ── TestReportPrompt ────────────────────────────────────────────────────


class TestReportPrompt:
    """Test report prompt building."""

    @pytest.fixture
    def mock_gemini_client(self):
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def generator(self, mock_gemini_client):
        return ReportGenerator(mock_gemini_client)

    @pytest.fixture
    def sample_state(self):
        return _make_sample_state()

    def test_prompt_contains_experiment_data(self, generator, sample_state):
        """Test prompt includes experiment names and models."""
        prompt = generator._build_report_prompt(sample_state)

        assert "baseline" in prompt
        assert "rf_basic" in prompt
        assert "rf_tuned" in prompt
        assert "LinearRegression" in prompt
        assert "RandomForestRegressor" in prompt

    def test_prompt_contains_delimiter_instruction(self, generator, sample_state):
        """Test prompt instructs Gemini to use section delimiter."""
        prompt = generator._build_report_prompt(sample_state)
        assert SECTION_DELIMITER in prompt

    def test_prompt_contains_dataset_info(self, generator, sample_state):
        """Test prompt includes dataset profile information."""
        prompt = generator._build_report_prompt(sample_state)

        assert "1460" in prompt  # row count
        assert "SalePrice" in prompt  # target column
        assert "regression" in prompt  # task type
        assert "rmse" in prompt  # primary metric


# ── TestNarrativeParsing ────────────────────────────────────────────────


class TestNarrativeParsing:
    """Test Gemini response parsing."""

    @pytest.fixture
    def mock_gemini_client(self):
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def generator(self, mock_gemini_client):
        return ReportGenerator(mock_gemini_client)

    def test_parse_valid_response(self, generator):
        """Test parsing a well-formed 4-section response."""
        response_text = _make_delimited_response()
        result = generator._parse_narrative_response(response_text)

        assert "executive_summary" in result
        assert "methodology" in result
        assert "key_insights" in result
        assert "recommendations" in result

        # Content from each section
        assert "executive summary" in result["executive_summary"].lower()
        assert "iterative" in result["methodology"].lower()
        assert "Tree-based" in result["key_insights"]
        assert "XGBoost" in result["recommendations"]

    def test_parse_missing_sections(self, generator):
        """Test parsing when fewer than 4 sections are returned."""
        # Only 2 sections
        text = (
            "This is a sufficiently long executive summary paragraph that exceeds "
            "the fifty character minimum required by the validation logic."
            f"\n{SECTION_DELIMITER}\n"
            "This is the methodology section."
        )
        result = generator._parse_narrative_response(text)

        assert result["executive_summary"] != ""
        assert result["methodology"] != ""
        assert result["key_insights"] == ""
        assert result["recommendations"] == ""

    def test_parse_empty_response(self, generator):
        """Test that empty response raises error."""
        with pytest.raises(GeminiInvalidResponseError, match="Empty response"):
            generator._parse_narrative_response("")

    def test_parse_short_executive_summary(self, generator):
        """Test that too-short executive summary raises error."""
        short_text = "Too short."
        with pytest.raises(
            GeminiInvalidResponseError, match="too short"
        ):
            generator._parse_narrative_response(short_text)


# ── TestFallbackNarrative ───────────────────────────────────────────────


class TestFallbackNarrative:
    """Test template-based fallback narrative."""

    @pytest.fixture
    def mock_gemini_client(self):
        with patch("src.cognitive.gemini_client.genai"):
            return Mock(spec=GeminiClient)

    @pytest.fixture
    def generator(self, mock_gemini_client):
        return ReportGenerator(mock_gemini_client)

    @pytest.fixture
    def sample_state(self):
        return _make_sample_state()

    def test_fallback_has_all_keys(self, generator, sample_state):
        """Test fallback returns all 4 required narrative keys."""
        result = generator._get_fallback_narrative(sample_state)

        assert "executive_summary" in result
        assert "methodology" in result
        assert "key_insights" in result
        assert "recommendations" in result

        # All values are non-empty strings
        for key, value in result.items():
            assert isinstance(value, str)
            assert len(value) > 0

    def test_fallback_mentions_best_model(self, generator, sample_state):
        """Test fallback narrative references the best experiment."""
        result = generator._get_fallback_narrative(sample_state)

        assert "rf_tuned" in result["executive_summary"]
        assert "rmse" in result["executive_summary"].lower() or "0.30" in result["executive_summary"]


# ── TestSystemPrompt ────────────────────────────────────────────────────


class TestSystemPrompt:
    """Test the system prompt content."""

    def test_prompt_contains_section_names(self):
        """Test system prompt references all required report sections."""
        prompt = REPORT_GENERATOR_SYSTEM_PROMPT

        assert "Executive Summary" in prompt or "EXECUTIVE SUMMARY" in prompt
        assert "Methodology" in prompt or "METHODOLOGY" in prompt
        assert "Key Insights" in prompt or "KEY INSIGHTS" in prompt
        assert "Recommendations" in prompt or "RECOMMENDATIONS" in prompt
