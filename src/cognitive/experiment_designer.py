"""ExperimentDesigner - Gemini-powered ML experiment design.

This component uses Gemini 3 to design experiments based on data profile,
previous results, and user constraints. It maintains reasoning continuity
across iterations through multi-turn conversation.
"""

import json
import re
from typing import Optional
from dataclasses import dataclass

from src.cognitive.gemini_client import (
    GeminiClient,
    GeminiError,
    GeminiInvalidResponseError,
)
from src.orchestration.state import (
    DataProfile,
    ExperimentResult,
    ExperimentSpec,
    PreprocessingConfig,
)


# System prompt for experiment design
EXPERIMENT_DESIGNER_SYSTEM_PROMPT = """You are an expert ML researcher designing experiments. Your goal is to systematically improve model performance through hypothesis-driven experimentation.

PRINCIPLES:
1. Each experiment tests a specific hypothesis derived from previous observations
2. Learn from both successes and failures in previous iterations
3. Consider data characteristics when selecting models and preprocessing
4. Apply appropriate preprocessing based on the data profile
5. Balance exploration (trying new approaches) with exploitation (refining what works)
6. Avoid repeating experiments that have already been tried

AVAILABLE MODELS:
- Regression: LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor, SVR, KNeighborsRegressor
- Classification: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, SVC, KNeighborsClassifier, GaussianNB

PREPROCESSING OPTIONS:
- missing_values: "drop" | "mean" | "median" | "mode" | "constant"
- scaling: "standard" | "minmax" | "none"
- encoding: "onehot" | "ordinal"
- target_transform: "log" | "none" (for regression only)

Always respond with valid JSON matching the required schema."""


@dataclass
class ParsedConstraints:
    """Parsed user constraints from Markdown."""

    primary_metric: Optional[str] = None
    preferred_models: list[str] = None
    avoided_models: list[str] = None
    preprocessing_hints: list[str] = None
    termination_rules: dict = None
    raw_text: str = ""

    def __post_init__(self):
        if self.preferred_models is None:
            self.preferred_models = []
        if self.avoided_models is None:
            self.avoided_models = []
        if self.preprocessing_hints is None:
            self.preprocessing_hints = []
        if self.termination_rules is None:
            self.termination_rules = {}


class ExperimentDesigner:
    """Designs ML experiments using Gemini 3.

    Key features:
    - Hypothesis-driven experiment design
    - Learns from previous experiment results
    - Respects user constraints
    - Maintains conversation context for Thought Signature continuity
    - Intelligent fallback when Gemini fails

    Usage:
        designer = ExperimentDesigner(gemini_client)
        spec = designer.design_experiment(
            data_profile=profile,
            previous_results=results,
            task_type="regression",
            constraints="Prefer tree-based models",
        )
    """

    def __init__(self, gemini_client: GeminiClient):
        """Initialize the experiment designer.

        Args:
            gemini_client: Shared GeminiClient instance for API calls.
                           Sharing preserves conversation history.
        """
        self.client = gemini_client
        self._parsed_constraints: Optional[ParsedConstraints] = None
        self._design_count = 0

    def design_experiment(
        self,
        data_profile: DataProfile,
        previous_results: list[ExperimentResult],
        task_type: str,
        constraints: Optional[str] = None,
        iteration: int = 1,
    ) -> ExperimentSpec:
        """Design the next experiment based on data and history.

        Args:
            data_profile: Profile of the dataset from DataProfiler.
            previous_results: List of previous experiment results.
            task_type: "classification" or "regression".
            constraints: Optional user constraints as Markdown text.
            iteration: Current iteration number.

        Returns:
            ExperimentSpec ready for code generation.

        Raises:
            GeminiError: If API call fails and no fallback available.
        """
        self._design_count += 1

        # Parse constraints if provided and not already parsed
        if constraints and self._parsed_constraints is None:
            self._parsed_constraints = self.parse_constraints(constraints)

        # Build the prompt
        prompt = self._build_design_prompt(
            data_profile=data_profile,
            previous_results=previous_results,
            task_type=task_type,
            iteration=iteration,
        )

        try:
            response = self.client.generate_json(
                prompt=prompt,
                system_instruction=EXPERIMENT_DESIGNER_SYSTEM_PROMPT,
                thinking_level="high",
            )

            return self._parse_design_response(response, iteration)

        except GeminiInvalidResponseError as e:
            # Try to extract useful info from invalid response
            return self._get_fallback_spec(task_type, iteration, str(e))

        except GeminiError as e:
            # API error - use fallback
            return self._get_fallback_spec(task_type, iteration, str(e))

    def parse_constraints(self, constraints_text: str) -> ParsedConstraints:
        """Parse Markdown constraints text into structured config.

        Args:
            constraints_text: Raw Markdown text from constraints file.

        Returns:
            ParsedConstraints with extracted values.
        """
        parsed = ParsedConstraints(raw_text=constraints_text)

        if not constraints_text:
            return parsed

        text_lower = constraints_text.lower()
        lines = constraints_text.split('\n')

        current_section = None

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Detect section headers
            if line_stripped.startswith('##'):
                header = line_stripped.lstrip('#').strip().lower()
                if 'metric' in header:
                    current_section = 'metrics'
                elif 'model' in header:
                    current_section = 'models'
                elif 'preprocess' in header:
                    current_section = 'preprocessing'
                elif 'terminat' in header:
                    current_section = 'termination'
                else:
                    current_section = None
                continue

            # Skip empty lines and main headers
            if not line_stripped or line_stripped.startswith('#'):
                continue

            # Parse content based on section
            if current_section == 'metrics':
                # Look for "primary metric: X" pattern
                if 'primary' in line_lower and ':' in line_lower:
                    metric = line_stripped.split(':')[-1].strip()
                    metric = metric.lower().replace(' ', '_')
                    parsed.primary_metric = metric
                # Also check for common metric names
                elif any(m in line_lower for m in ['rmse', 'mse', 'mae', 'r2', 'f1', 'accuracy', 'precision', 'recall', 'auc', 'log_loss']):
                    for m in ['rmse', 'mse', 'mae', 'r2', 'f1', 'accuracy', 'precision', 'recall', 'auc', 'log_loss']:
                        if m in line_lower:
                            parsed.primary_metric = m
                            break

            elif current_section == 'models':
                if 'prefer' in line_lower:
                    # Extract model preferences
                    models = self._extract_model_names(line_stripped)
                    parsed.preferred_models.extend(models)
                elif 'avoid' in line_lower or 'don\'t' in line_lower or 'do not' in line_lower:
                    # Extract models to avoid
                    models = self._extract_model_names(line_stripped)
                    parsed.avoided_models.extend(models)

            elif current_section == 'preprocessing':
                # Store preprocessing hints as-is
                if line_stripped.startswith('-'):
                    hint = line_stripped.lstrip('- ').strip()
                    parsed.preprocessing_hints.append(hint)

            elif current_section == 'termination':
                # Parse termination rules
                if 'no improvement' in line_lower:
                    # Look for number of iterations
                    numbers = re.findall(r'\d+', line_stripped)
                    if numbers:
                        parsed.termination_rules['plateau_threshold'] = int(numbers[0])

        return parsed

    def _extract_model_names(self, text: str) -> list[str]:
        """Extract model names from text.

        Args:
            text: Text containing model names.

        Returns:
            List of identified model types.
        """
        models = []
        text_lower = text.lower()

        # Model families/types
        model_keywords = {
            'tree': ['RandomForest', 'DecisionTree', 'GradientBoosting'],
            'forest': ['RandomForest'],
            'boosting': ['GradientBoosting'],
            'linear': ['LinearRegression', 'LogisticRegression', 'Ridge', 'Lasso'],
            'svm': ['SVC', 'SVR'],
            'knn': ['KNeighbors'],
            'naive bayes': ['GaussianNB'],
            'neural': [],  # Not supported yet
        }

        for keyword, model_types in model_keywords.items():
            if keyword in text_lower:
                models.extend(model_types)

        return list(set(models))

    def select_primary_metric(
        self,
        task_type: str,
        constraints: Optional[ParsedConstraints] = None,
    ) -> str:
        """Select the primary metric based on task and constraints.

        Args:
            task_type: "classification" or "regression".
            constraints: Parsed constraints (may contain metric preference).

        Returns:
            Primary metric name.
        """
        # Check constraints first
        if constraints and constraints.primary_metric:
            return constraints.primary_metric

        # Default by task type
        if task_type == "regression":
            return "rmse"
        else:
            return "f1"

    def _build_design_prompt(
        self,
        data_profile: DataProfile,
        previous_results: list[ExperimentResult],
        task_type: str,
        iteration: int,
    ) -> str:
        """Build the prompt for experiment design.

        Args:
            data_profile: Dataset profile.
            previous_results: Previous experiment results.
            task_type: Task type.
            iteration: Current iteration.

        Returns:
            Formatted prompt string.
        """
        # Summarize data profile
        profile_summary = {
            "n_rows": data_profile.n_rows,
            "n_columns": data_profile.n_columns,
            "numeric_columns": len(data_profile.numeric_columns),
            "categorical_columns": len(data_profile.categorical_columns),
            "target_column": data_profile.target_column,
            "target_type": data_profile.target_type,
            "missing_values_total": sum(data_profile.missing_values.values()),
            "columns_with_missing": [
                col for col, count in data_profile.missing_values.items() if count > 0
            ],
        }

        # Add target statistics
        if data_profile.target_stats:
            profile_summary["target_stats"] = data_profile.target_stats

        # Summarize previous results (last 5)
        results_summary = []
        for exp in previous_results[-5:]:
            results_summary.append({
                "name": exp.experiment_name,
                "model": exp.model_type,
                "metrics": exp.metrics,
                "hypothesis": exp.hypothesis,
                "success": exp.success,
                "error": exp.error_message if not exp.success else None,
            })

        # Build prompt
        prompt = f"""## Experiment Design Request

### Iteration: {iteration}

### Task Type: {task_type}

### Dataset Profile
```json
{json.dumps(profile_summary, indent=2)}
```

### Previous Experiments
"""

        if results_summary:
            prompt += f"""```json
{json.dumps(results_summary, indent=2)}
```
"""
        else:
            prompt += "No previous experiments. This is the first iteration.\n"

        # Add constraints if available
        if self._parsed_constraints:
            prompt += "\n### User Constraints\n"
            if self._parsed_constraints.primary_metric:
                prompt += f"- Primary metric: {self._parsed_constraints.primary_metric}\n"
            if self._parsed_constraints.preferred_models:
                prompt += f"- Preferred models: {', '.join(self._parsed_constraints.preferred_models)}\n"
            if self._parsed_constraints.avoided_models:
                prompt += f"- Avoid models: {', '.join(self._parsed_constraints.avoided_models)}\n"
            if self._parsed_constraints.preprocessing_hints:
                prompt += "- Preprocessing hints:\n"
                for hint in self._parsed_constraints.preprocessing_hints:
                    prompt += f"  - {hint}\n"

        prompt += f"""
### Your Task
Design experiment #{iteration}. Based on the data profile and previous results:

1. Formulate a clear hypothesis for what you want to test
2. Select an appropriate model (don't repeat models that have already been tried unless with different params)
3. Choose preprocessing that addresses the data's characteristics
4. Explain your reasoning

Respond with valid JSON:
```json
{{
    "experiment_name": "descriptive_name_iteration{iteration}",
    "hypothesis": "Clear statement of what you're testing and why",
    "model_type": "ModelClassName",
    "model_params": {{"param": "value"}},
    "preprocessing": {{
        "missing_values": "mean|median|mode|drop",
        "scaling": "standard|minmax|none",
        "encoding": "onehot|ordinal",
        "target_transform": "log|none"
    }},
    "reasoning": "Detailed explanation of your choices based on data profile and previous results"
}}
```"""

        return prompt

    def _parse_design_response(
        self,
        response: dict,
        iteration: int,
    ) -> ExperimentSpec:
        """Parse Gemini's design response into ExperimentSpec.

        Args:
            response: Parsed JSON response from Gemini.
            iteration: Current iteration.

        Returns:
            ExperimentSpec ready for code generation.

        Raises:
            ValueError: If response is missing required fields.
        """
        # Validate required fields
        required_fields = ["experiment_name", "hypothesis", "model_type"]
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")

        # Parse preprocessing config
        preprocessing_data = response.get("preprocessing", {})
        preprocessing = PreprocessingConfig(
            missing_values=preprocessing_data.get("missing_values", "median"),
            scaling=preprocessing_data.get("scaling", "standard"),
            encoding=preprocessing_data.get("encoding", "onehot"),
            target_transform=preprocessing_data.get("target_transform"),
        )

        return ExperimentSpec(
            experiment_name=response["experiment_name"],
            hypothesis=response["hypothesis"],
            model_type=response["model_type"],
            model_params=response.get("model_params", {}),
            preprocessing=preprocessing,
            reasoning=response.get("reasoning", ""),
        )

    def _get_fallback_spec(
        self,
        task_type: str,
        iteration: int,
        error_reason: str = "",
    ) -> ExperimentSpec:
        """Get a fallback experiment spec when Gemini fails.

        Args:
            task_type: Task type.
            iteration: Current iteration.
            error_reason: Why fallback is needed.

        Returns:
            Simple fallback ExperimentSpec.
        """
        # Rotate through models based on iteration
        if task_type == "regression":
            models = [
                ("RandomForestRegressor", {"n_estimators": 100, "max_depth": 10}),
                ("GradientBoostingRegressor", {"n_estimators": 100, "learning_rate": 0.1}),
                ("Ridge", {"alpha": 1.0}),
                ("Lasso", {"alpha": 0.1}),
            ]
        else:
            models = [
                ("RandomForestClassifier", {"n_estimators": 100, "max_depth": 10}),
                ("GradientBoostingClassifier", {"n_estimators": 100, "learning_rate": 0.1}),
                ("LogisticRegression", {"max_iter": 1000}),
            ]

        model_idx = (iteration - 1) % len(models)
        model_type, model_params = models[model_idx]

        return ExperimentSpec(
            experiment_name=f"fallback_{model_type.lower()}_{iteration}",
            hypothesis=f"Fallback experiment with {model_type}",
            model_type=model_type,
            model_params=model_params,
            preprocessing=PreprocessingConfig(),
            reasoning=f"Fallback due to Gemini error: {error_reason[:200]}",
        )

    def get_design_count(self) -> int:
        """Get the number of experiments designed so far."""
        return self._design_count

    def get_parsed_constraints(self) -> Optional[ParsedConstraints]:
        """Get the parsed constraints if available."""
        return self._parsed_constraints
