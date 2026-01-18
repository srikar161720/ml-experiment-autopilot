"""Code generation for ML experiments using Jinja2 templates."""

import ast
from pathlib import Path
from datetime import datetime
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from src.config import TEMPLATES_DIR, EXPERIMENTS_DIR
from src.orchestration.state import ExperimentSpec, PreprocessingConfig


class CodeGenerationError(Exception):
    """Error during code generation."""

    pass


class CodeGenerator:
    """Generate Python experiment scripts from templates.

    Uses Jinja2 templates to create runnable sklearn training scripts.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the code generator.

        Args:
            templates_dir: Path to templates directory. Defaults to project templates/.
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR

        # Set up Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(
        self,
        spec: ExperimentSpec,
        data_path: Path,
        target_column: str,
        task_type: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate an experiment script from a specification.

        Args:
            spec: Experiment specification from Gemini.
            data_path: Path to the dataset.
            target_column: Name of target column.
            task_type: 'classification' or 'regression'.
            output_dir: Output directory for generated script.

        Returns:
            Path to the generated script.

        Raises:
            CodeGenerationError: If generation or validation fails.
        """
        output_dir = output_dir or EXPERIMENTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select template based on task type
        template_name = self._get_template_name(task_type)
        template = self.env.get_template(template_name)

        # Prepare template context
        context = self._build_context(spec, data_path, target_column, task_type)

        # Render template
        code = template.render(**context)

        # Validate generated code
        self._validate_code(code)

        # Save to file
        script_path = output_dir / f"{spec.experiment_name}.py"
        script_path.write_text(code)

        return script_path

    def generate_baseline(
        self,
        data_path: Path,
        target_column: str,
        task_type: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Generate a baseline experiment script.

        Args:
            data_path: Path to the dataset.
            target_column: Name of target column.
            task_type: 'classification' or 'regression'.
            output_dir: Output directory for generated script.

        Returns:
            Path to the generated script.
        """
        # Create baseline spec
        if task_type == "regression":
            spec = ExperimentSpec(
                experiment_name="baseline_linear_regression",
                hypothesis="Establish baseline performance with a simple linear model",
                model_type="LinearRegression",
                model_params={},
                preprocessing=PreprocessingConfig(
                    missing_values="median",
                    scaling="standard",
                    encoding="onehot",
                ),
                reasoning="Starting with a simple linear model to establish baseline metrics",
            )
        else:
            spec = ExperimentSpec(
                experiment_name="baseline_logistic_regression",
                hypothesis="Establish baseline performance with logistic regression",
                model_type="LogisticRegression",
                model_params={},
                preprocessing=PreprocessingConfig(
                    missing_values="median",
                    scaling="standard",
                    encoding="onehot",
                ),
                reasoning="Starting with logistic regression to establish baseline metrics",
            )

        return self.generate(spec, data_path, target_column, task_type, output_dir)

    def _get_template_name(self, task_type: str) -> str:
        """Get the appropriate template name for the task type."""
        if task_type == "regression":
            return "sklearn_regressor.py.jinja"
        else:
            return "sklearn_classifier.py.jinja"

    def _build_context(
        self,
        spec: ExperimentSpec,
        data_path: Path,
        target_column: str,
        task_type: str,
    ) -> dict:
        """Build the context dictionary for template rendering."""
        # Convert model params to string format for template
        model_params_str = self._format_model_params(spec.model_params)

        return {
            "experiment_name": spec.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "data_path": str(data_path.absolute()),
            "target_column": target_column,
            "task_type": task_type,
            "model_type": spec.model_type,
            "model_params": spec.model_params,
            "model_params_str": model_params_str,
            "preprocessing": spec.preprocessing,
            "hypothesis": spec.hypothesis,
            "reasoning": spec.reasoning,
        }

    def _format_model_params(self, params: dict) -> str:
        """Format model parameters as a string for template insertion.

        Args:
            params: Dictionary of model parameters.

        Returns:
            Formatted string like "n_estimators=100, max_depth=5, "
        """
        if not params:
            return ""

        parts = []
        for key, value in params.items():
            if isinstance(value, str):
                parts.append(f'{key}="{value}"')
            else:
                parts.append(f"{key}={value}")

        return ", ".join(parts) + ", "

    def _validate_code(self, code: str):
        """Validate generated Python code for syntax errors.

        Args:
            code: The generated Python code.

        Raises:
            CodeGenerationError: If code has syntax errors.
        """
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise CodeGenerationError(
                f"Generated code has syntax error at line {e.lineno}: {e.msg}"
            )


def create_experiment_from_gemini_response(response: dict) -> ExperimentSpec:
    """Create an ExperimentSpec from Gemini's JSON response.

    Args:
        response: Parsed JSON response from Gemini.

    Returns:
        ExperimentSpec object.

    Raises:
        ValueError: If response is missing required fields.
    """
    required_fields = ["experiment_name", "hypothesis", "model_type"]
    for field in required_fields:
        if field not in response:
            raise ValueError(f"Gemini response missing required field: {field}")

    # Parse preprocessing config
    preprocessing_dict = response.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        missing_values=preprocessing_dict.get("missing_values", "median"),
        scaling=preprocessing_dict.get("scaling", "standard"),
        encoding=preprocessing_dict.get("encoding", "onehot"),
        target_transform=preprocessing_dict.get("target_transform"),
    )

    return ExperimentSpec(
        experiment_name=response["experiment_name"],
        hypothesis=response["hypothesis"],
        model_type=response["model_type"],
        model_params=response.get("model_params", {}),
        preprocessing=preprocessing,
        reasoning=response.get("reasoning", ""),
    )
