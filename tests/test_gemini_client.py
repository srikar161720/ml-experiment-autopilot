"""Tests for the GeminiClient (mocked)."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiResponse,
    GeminiError,
    GeminiRateLimitError,
    GeminiInvalidResponseError,
    create_experiment_designer_prompt,
)
from src.config import GeminiConfig


class TestGeminiClient:
    """Test cases for GeminiClient (mocked API calls)."""

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

    def test_init(self, mock_config, mock_genai):
        """Test client initialization."""
        client = GeminiClient(mock_config)

        mock_genai.configure.assert_called_once_with(api_key="test_key")
        assert client.config == mock_config

    def test_generate_success(self, mock_config, mock_genai):
        """Test successful generation."""
        # Setup mock response
        mock_response = Mock()
        mock_response.text = "Generated text response"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)
        response = client.generate("Test prompt")

        assert isinstance(response, GeminiResponse)
        assert response.text == "Generated text response"

    def test_generate_with_system_instruction(self, mock_config, mock_genai):
        """Test generation with system instruction."""
        mock_response = Mock()
        mock_response.text = "Response with system instruction"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)
        response = client.generate(
            "Test prompt",
            system_instruction="You are a helpful assistant",
        )

        assert response.text == "Response with system instruction"
        # Verify GenerativeModel was called with system instruction
        assert mock_genai.GenerativeModel.call_count >= 1

    def test_generate_json_success(self, mock_config, mock_genai):
        """Test JSON generation and parsing."""
        mock_response = Mock()
        mock_response.text = '{"key": "value", "number": 42}'

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)
        result = client.generate_json("Generate JSON")

        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_generate_json_with_markdown(self, mock_config, mock_genai):
        """Test JSON parsing when wrapped in markdown code blocks."""
        mock_response = Mock()
        mock_response.text = '```json\n{"key": "value"}\n```'

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)
        result = client.generate_json("Generate JSON")

        assert result["key"] == "value"

    def test_generate_json_invalid(self, mock_config, mock_genai):
        """Test error on invalid JSON response."""
        mock_response = Mock()
        mock_response.text = "This is not valid JSON"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)

        with pytest.raises(GeminiInvalidResponseError):
            client.generate_json("Generate JSON")

    def test_conversation_history(self, mock_config, mock_genai):
        """Test conversation history management."""
        mock_response = Mock()
        mock_response.text = "Response"

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)

        assert client.get_history_length() == 0

        client.generate("First message")
        assert client.get_history_length() == 2  # user + model

        client.generate("Second message")
        assert client.get_history_length() == 4

        client.clear_history()
        assert client.get_history_length() == 0

    def test_add_message(self, mock_config, mock_genai):
        """Test manually adding messages to history."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(mock_config)

        client.add_message("user", "Test message")
        assert client.get_history_length() == 1

        client.add_message("model", "Response")
        assert client.get_history_length() == 2


class TestExperimentDesignerPrompt:
    """Test cases for the experiment designer prompt creation."""

    def test_basic_prompt(self):
        """Test creating a basic prompt."""
        data_profile = {
            "n_rows": 100,
            "n_columns": 5,
            "columns": ["a", "b", "c", "d", "target"],
        }

        prompt = create_experiment_designer_prompt(
            data_profile=data_profile,
            previous_results=[],
            task_type="regression",
        )

        assert "Dataset Profile" in prompt
        assert "regression" in prompt
        assert "No previous experiments" in prompt
        assert "JSON" in prompt

    def test_prompt_with_results(self):
        """Test prompt with previous results."""
        data_profile = {"n_rows": 100}
        previous_results = [
            {"name": "baseline", "model": "LinearRegression", "metrics": {"rmse": 0.5}},
            {"name": "rf_1", "model": "RandomForest", "metrics": {"rmse": 0.4}},
        ]

        prompt = create_experiment_designer_prompt(
            data_profile=data_profile,
            previous_results=previous_results,
            task_type="regression",
        )

        assert "baseline" in prompt
        assert "LinearRegression" in prompt
        assert "RandomForest" in prompt

    def test_prompt_with_constraints(self):
        """Test prompt with user constraints."""
        data_profile = {"n_rows": 100}

        prompt = create_experiment_designer_prompt(
            data_profile=data_profile,
            previous_results=[],
            constraints="Use only tree-based models",
            task_type="classification",
        )

        assert "Constraints" in prompt
        assert "tree-based" in prompt
        assert "classification" in prompt
