"""Gemini API client with retry logic and Thought Signature support."""

import json
import time
from typing import Optional, Literal
from dataclasses import dataclass, field

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.config import GeminiConfig, get_config


ThinkingLevel = Literal["low", "medium", "high"]


@dataclass
class ConversationMessage:
    """A message in the conversation history."""

    role: Literal["user", "model"]
    content: str


@dataclass
class GeminiResponse:
    """Response from Gemini API."""

    text: str
    thinking: Optional[str] = None
    raw_response: Optional[object] = None


class GeminiError(Exception):
    """Base exception for Gemini-related errors."""

    pass


class GeminiRateLimitError(GeminiError):
    """Rate limit exceeded."""

    pass


class GeminiInvalidResponseError(GeminiError):
    """Invalid or unparseable response from Gemini."""

    pass


class GeminiClient:
    """Client for interacting with Gemini 3 API with Thought Signatures.

    Key features:
    - Temperature fixed at 1.0 (required for Thought Signatures)
    - Retry logic with exponential backoff
    - Multi-turn conversation support
    - Thinking level configuration

    Usage:
        client = GeminiClient()
        response = client.generate("Design an ML experiment...")

        # For multi-turn conversations
        client.add_message("user", "What about trying XGBoost?")
        response = client.generate_with_history()
    """

    def __init__(self, config: Optional[GeminiConfig] = None):
        """Initialize the Gemini client.

        Args:
            config: Optional GeminiConfig. If not provided, loads from environment.
        """
        self.config = config or get_config().gemini

        # Configure the API
        genai.configure(api_key=self.config.api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(self.config.model)

        # Conversation history for multi-turn support
        self.conversation_history: list[ConversationMessage] = []

    def _get_generation_config(
        self, thinking_level: ThinkingLevel = "high"
    ) -> genai.GenerationConfig:
        """Get generation config with appropriate settings.

        Args:
            thinking_level: Level of thinking/reasoning to use.

        Returns:
            GenerationConfig with appropriate parameters.
        """
        # Temperature MUST be 1.0 for Thought Signatures
        return genai.GenerationConfig(
            temperature=self.config.temperature,
            # Additional thinking configuration may be added here
            # based on Gemini 3 API specifics
        )

    def _build_prompt_with_history(self, prompt: str) -> list[dict]:
        """Build prompt including conversation history.

        Args:
            prompt: The new user prompt.

        Returns:
            List of message dicts for the API.
        """
        messages = []

        # Add conversation history
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "parts": [msg.content]})

        # Add the new prompt
        messages.append({"role": "user", "parts": [prompt]})

        return messages

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        thinking_level: ThinkingLevel = "high",
        use_history: bool = False,
    ) -> GeminiResponse:
        """Generate a response from Gemini.

        Args:
            prompt: The user prompt.
            system_instruction: Optional system-level instruction.
            thinking_level: Level of thinking (low/medium/high).
            use_history: Whether to include conversation history.

        Returns:
            GeminiResponse with text and optional thinking.

        Raises:
            GeminiError: If all retries fail.
        """
        generation_config = self._get_generation_config(thinking_level)

        # Create model with system instruction if provided
        if system_instruction:
            model = genai.GenerativeModel(
                self.config.model, system_instruction=system_instruction
            )
        else:
            model = self.model

        # Build content based on whether we use history
        if use_history:
            content = self._build_prompt_with_history(prompt)
        else:
            content = prompt

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.config.max_retries):
            try:
                response = model.generate_content(
                    content, generation_config=generation_config
                )

                # Extract response text
                response_text = response.text

                # Add to conversation history
                self.conversation_history.append(
                    ConversationMessage(role="user", content=prompt)
                )
                self.conversation_history.append(
                    ConversationMessage(role="model", content=response_text)
                )

                return GeminiResponse(
                    text=response_text,
                    thinking=None,  # Extract thinking from response if available
                    raw_response=response,
                )

            except google_exceptions.ResourceExhausted as e:
                # Rate limit - wait and retry
                last_exception = GeminiRateLimitError(str(e))
                wait_time = self.config.retry_delay * (2**attempt)
                time.sleep(wait_time)

            except google_exceptions.InvalidArgument as e:
                # Bad request - don't retry
                raise GeminiError(f"Invalid request: {e}")

            except Exception as e:
                # Other errors - retry with backoff
                last_exception = GeminiError(str(e))
                wait_time = self.config.retry_delay * (2**attempt)
                time.sleep(wait_time)

        # All retries exhausted
        raise last_exception or GeminiError("Unknown error after retries")

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        thinking_level: ThinkingLevel = "high",
    ) -> dict:
        """Generate and parse a JSON response from Gemini.

        Args:
            prompt: The user prompt (should request JSON output).
            system_instruction: Optional system-level instruction.
            thinking_level: Level of thinking (low/medium/high).

        Returns:
            Parsed JSON dict.

        Raises:
            GeminiInvalidResponseError: If response cannot be parsed as JSON.
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only. No additional text."

        response = self.generate(
            json_prompt,
            system_instruction=system_instruction,
            thinking_level=thinking_level,
        )

        # Try to parse JSON
        try:
            # Clean up the response - remove markdown code blocks if present
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            return json.loads(text)
        except json.JSONDecodeError as e:
            raise GeminiInvalidResponseError(
                f"Failed to parse JSON response: {e}\nResponse was: {response.text[:500]}"
            )

    def add_message(self, role: Literal["user", "model"], content: str):
        """Add a message to the conversation history.

        Args:
            role: The role (user or model).
            content: The message content.
        """
        self.conversation_history.append(ConversationMessage(role=role, content=content))

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history_length(self) -> int:
        """Get the number of messages in conversation history."""
        return len(self.conversation_history)


def create_experiment_designer_prompt(
    data_profile: dict,
    previous_results: list[dict],
    constraints: Optional[str] = None,
    task_type: str = "regression",
) -> str:
    """Create a prompt for the experiment designer.

    Args:
        data_profile: Dataset profile from DataProfiler.
        previous_results: List of previous experiment results.
        constraints: Optional user constraints.
        task_type: 'classification' or 'regression'.

    Returns:
        Formatted prompt string.
    """
    prompt = f"""You are an ML experiment designer. Design the next experiment based on the data and results.

## Dataset Profile
{json.dumps(data_profile, indent=2)}

## Task Type
{task_type}

## Previous Experiments
{json.dumps(previous_results, indent=2) if previous_results else "No previous experiments."}

"""
    if constraints:
        prompt += f"""## User Constraints
{constraints}

"""

    prompt += """## Your Task
Design the next experiment. Consider:
1. What worked and didn't work in previous experiments
2. What hypothesis are you testing
3. What model and preprocessing to use

Respond with a JSON object:
{
    "experiment_name": "descriptive_name",
    "hypothesis": "What you're testing and why",
    "model_type": "sklearn model class name (e.g., RandomForestRegressor)",
    "model_params": {"param": "value"},
    "preprocessing": {
        "missing_values": "drop|mean|median|mode",
        "scaling": "standard|minmax|none",
        "encoding": "onehot|ordinal"
    },
    "reasoning": "Detailed explanation of your choices"
}"""

    return prompt
