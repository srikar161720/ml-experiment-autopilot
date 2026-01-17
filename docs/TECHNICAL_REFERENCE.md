# ML Experiment Autopilot - Technical Reference

## Document Purpose
This document contains practical implementation details: dependencies, code snippets, API references, and configuration templates ready for use during development.

---

## 1. Dependencies

### requirements.txt

```
# Core - Gemini Integration
google-genai>=1.0.0

# ML Frameworks
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0

# Experiment Tracking
mlflow>=2.9.0

# Data Handling
pandas>=2.1.0
numpy>=1.25.0
pyarrow>=14.0.0

# Data Profiling
scipy>=1.11.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# CLI & Configuration
typer>=0.9.0
pydantic>=2.5.0
python-dotenv>=1.0.0

# Console Output
rich>=13.7.0

# Templating
jinja2>=3.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "from google import genai; print('Gemini SDK OK')"
python -c "import mlflow; print(f'MLflow {mlflow.__version__} OK')"
python -c "import typer; print('Typer OK')"
```

---

## 2. Configuration

### .env.example

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3-pro-preview

# MLflow Configuration
MLFLOW_TRACKING_URI=./outputs/mlruns
MLFLOW_EXPERIMENT_NAME=ml-experiment-autopilot

# Experiment Runner Configuration
EXPERIMENT_TIMEOUT_SECONDS=300
MAX_ITERATIONS=20
RANDOM_STATE=42

# Termination Criteria
PLATEAU_THRESHOLD=3
IMPROVEMENT_THRESHOLD=0.005
TIME_BUDGET_SECONDS=3600

# Logging
LOG_LEVEL=INFO
```

### config.py

```python
"""Configuration management for ML Experiment Autopilot."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeminiConfig:
    """Gemini API configuration."""
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"))
    temperature: float = 1.0  # REQUIRED - do not change
    default_thinking_level: str = "high"
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class ExperimentConfig:
    """Experiment execution configuration."""
    timeout_seconds: int = int(os.getenv("EXPERIMENT_TIMEOUT_SECONDS", "300"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "20"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    cv_folds: int = 5
    plateau_threshold: int = int(os.getenv("PLATEAU_THRESHOLD", "3"))
    improvement_threshold: float = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.005"))
    time_budget: int = int(os.getenv("TIME_BUDGET_SECONDS", "3600"))


@dataclass
class PathConfig:
    """Path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs" / "experiments")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs" / "reports")
    mlruns_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "outputs" / "mlruns")
    state_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / ".autopilot_state")
    templates_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "templates")

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [
            self.outputs_dir, self.experiments_dir, self.reports_dir,
            self.mlruns_dir, self.state_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "./outputs/mlruns")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-experiment-autopilot")
    )


@dataclass
class Config:
    """Main configuration container."""
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


# Global config instance
config = Config()
```

---

## 3. Pydantic State Models

### orchestration/state.py

```python
"""Pydantic models for experiment state management."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import time


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ExperimentPhase(str, Enum):
    INITIALIZING = "initializing"
    DATA_PROFILING = "data_profiling"
    BASELINE_MODELING = "baseline_modeling"
    EXPERIMENT_DESIGN = "experiment_design"
    CODE_GENERATION = "code_generation"
    EXPERIMENT_EXECUTION = "experiment_execution"
    RESULTS_ANALYSIS = "results_analysis"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ITERATION_DECISION = "iteration_decision"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentSpec(BaseModel):
    """Specification for a single experiment."""
    experiment_name: str
    hypothesis: str
    model_type: str
    model_params: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    feature_columns: Optional[List[str]] = None
    reasoning: str


class ExperimentResult(BaseModel):
    """Results from a single experiment."""
    experiment_name: str
    iteration: int
    spec: ExperimentSpec
    metrics: Dict[str, float]
    training_time: float
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


class DataProfile(BaseModel):
    """Profile of the input dataset."""
    n_rows: int
    n_columns: int
    columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]
    target_column: str
    target_distribution: Dict[str, Any]
    missing_values: Dict[str, int]
    statistics: Dict[str, Dict[str, float]]
    cardinality: Dict[str, int]


class ExperimentState(BaseModel):
    """Complete state of an experiment session."""
    session_id: str
    data_path: str
    target_column: str
    task_type: TaskType
    constraints: Optional[str] = None
    
    # Data info
    data_profile: Optional[DataProfile] = None
    
    # Experiment tracking
    experiments: List[ExperimentResult] = Field(default_factory=list)
    current_iteration: int = 0
    
    # Best performance tracking
    primary_metric: Optional[str] = None
    best_metric_value: Optional[float] = None
    best_experiment_name: Optional[str] = None
    iterations_without_improvement: int = 0
    
    # Timing
    start_time: float = Field(default_factory=time.time)
    
    # Current phase
    phase: ExperimentPhase = ExperimentPhase.INITIALIZING
    
    # Gemini context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    
    # Termination
    agent_recommends_stop: bool = False
    termination_reason: Optional[str] = None
    
    def save(self, path: Path):
        """Save state to JSON file."""
        path.write_text(self.model_dump_json(indent=2))
    
    @classmethod
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from JSON file."""
        return cls.model_validate_json(path.read_text())
    
    def update_best(self, result: ExperimentResult, metric_name: str, higher_is_better: bool = False):
        """Update best performance if this result is better."""
        current_value = result.metrics.get(metric_name)
        if current_value is None:
            return
        
        if self.best_metric_value is None:
            self.best_metric_value = current_value
            self.best_experiment_name = result.experiment_name
            self.iterations_without_improvement = 0
            return
        
        # Check improvement (considering direction)
        if higher_is_better:
            improved = current_value > self.best_metric_value * (1 + 0.005)
        else:
            improved = current_value < self.best_metric_value * (1 - 0.005)
        
        if improved:
            self.best_metric_value = current_value
            self.best_experiment_name = result.experiment_name
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
```

---

## 4. Gemini Client Implementation

### cognitive/gemini_client.py

```python
"""Gemini 3 API client with Thought Signature management."""

import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from google import genai
from google.genai import types

from config import config


class GeminiError(Exception):
    """Exception for Gemini API errors."""
    pass


class GeminiClient:
    """
    Client for Gemini 3 API with automatic retry and error handling.
    
    Usage:
        client = GeminiClient()
        response = client.generate("Design an experiment...")
        
        # For multi-turn conversations
        conversation = client.create_conversation()
        response1 = conversation.send("Analyze this data...")
        response2 = conversation.send("What should we try next?")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.gemini.api_key
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY env var.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = config.gemini.model
        self.max_retries = config.gemini.max_retries
        self.retry_delay = config.gemini.retry_delay
    
    def generate(
        self,
        prompt: str,
        thinking_level: str = "high",
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a single response (no conversation history).
        
        Args:
            prompt: The user prompt
            thinking_level: "high", "medium", or "low"
            system_instruction: Optional system prompt
            
        Returns:
            Dict with 'text', 'thought_signature', and 'raw_response'
        """
        generation_config = types.GenerateContentConfig(
            temperature=config.gemini.temperature,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level
            )
        )
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=generation_config,
                    system_instruction=system_instruction
                )
                
                return {
                    "text": response.text,
                    "thought_signature": getattr(response, 'thought_signature', None),
                    "raw_response": response
                }
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Gemini API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise GeminiError(f"Gemini API failed after {self.max_retries} attempts: {e}")
    
    def generate_json(
        self,
        prompt: str,
        thinking_level: str = "high",
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON response.
        
        Returns:
            Parsed JSON dict
            
        Raises:
            GeminiError if JSON parsing fails
        """
        response = self.generate(prompt, thinking_level, system_instruction)
        text = response["text"]
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise GeminiError(f"Failed to parse JSON from Gemini response: {e}\nResponse: {text[:500]}")
    
    def create_conversation(self, system_instruction: Optional[str] = None) -> 'Conversation':
        """Create a new multi-turn conversation."""
        return Conversation(self, system_instruction)


class Conversation:
    """
    Manages a multi-turn conversation with Thought Signature circulation.
    """
    
    def __init__(self, client: GeminiClient, system_instruction: Optional[str] = None):
        self.client = client
        self.system_instruction = system_instruction
        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
    
    def send(self, message: str, thinking_level: str = "high") -> Dict[str, Any]:
        """
        Send a message in the conversation.
        
        Returns:
            Dict with 'text', 'thought_signature', 'turn_count'
        """
        # Build contents with history
        contents = []
        for turn in self.history:
            contents.append({"role": turn["role"], "parts": [{"text": turn["content"]}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        
        generation_config = types.GenerateContentConfig(
            temperature=config.gemini.temperature,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking_level
            )
        )
        
        for attempt in range(self.client.max_retries):
            try:
                response = self.client.client.models.generate_content(
                    model=self.client.model,
                    contents=contents,
                    config=generation_config,
                    system_instruction=self.system_instruction
                )
                
                # Update history
                self.history.append({"role": "user", "content": message})
                self.history.append({"role": "model", "content": response.text})
                self.turn_count += 1
                
                return {
                    "text": response.text,
                    "thought_signature": getattr(response, 'thought_signature', None),
                    "turn_count": self.turn_count
                }
                
            except Exception as e:
                if attempt < self.client.max_retries - 1:
                    wait_time = self.client.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise GeminiError(f"Conversation failed: {e}")
    
    def get_history_summary(self) -> str:
        """Get a summary of conversation for logging."""
        return f"Conversation with {self.turn_count} turns, {len(self.history)} messages"
```

---

## 5. Data Profiler Implementation

### execution/data_profiler.py

```python
"""Data profiling for ML datasets."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from orchestration.state import DataProfile, TaskType


class DataProfiler:
    """
    Generates comprehensive profiles of tabular datasets.
    """
    
    def __init__(self, data_path: Path, target_column: str, task_type: TaskType):
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type
        self.df: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """Load dataset from file."""
        suffix = self.data_path.suffix.lower()
        
        if suffix == '.csv':
            self.df = pd.read_csv(self.data_path)
        elif suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        return self.df
    
    def profile(self) -> DataProfile:
        """Generate complete data profile."""
        if self.df is None:
            self.load_data()
        
        df = self.df
        
        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)
        
        # Calculate statistics for numeric columns
        statistics = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                statistics[col] = {
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75)),
                    "skewness": float(col_data.skew()),
                }
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}
        
        # Cardinality for categorical columns
        cardinality = {col: int(df[col].nunique()) for col in categorical_cols}
        
        # Target distribution
        target_data = df[self.target_column]
        if self.task_type == TaskType.CLASSIFICATION:
            target_distribution = {
                "type": "classification",
                "n_classes": int(target_data.nunique()),
                "class_counts": target_data.value_counts().to_dict(),
            }
        else:
            target_distribution = {
                "type": "regression",
                "mean": float(target_data.mean()),
                "median": float(target_data.median()),
                "std": float(target_data.std()),
                "min": float(target_data.min()),
                "max": float(target_data.max()),
                "skewness": float(target_data.skew()),
            }
        
        return DataProfile(
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=df.columns.tolist(),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            target_column=self.target_column,
            target_distribution=target_distribution,
            missing_values=missing_values,
            statistics=statistics,
            cardinality=cardinality,
        )
    
    def get_profile_summary(self) -> str:
        """Get human-readable profile summary for Gemini."""
        profile = self.profile()
        
        lines = [
            f"Dataset: {self.data_path.name}",
            f"Shape: {profile.n_rows} rows, {profile.n_columns} columns",
            f"Target: {profile.target_column} ({self.task_type.value})",
            "",
            f"Numeric features: {len(profile.numeric_columns)}",
            f"Categorical features: {len(profile.categorical_columns)}",
            "",
        ]
        
        if profile.missing_values:
            lines.append("Missing values:")
            for col, count in sorted(profile.missing_values.items(), key=lambda x: -x[1])[:5]:
                pct = count / profile.n_rows * 100
                lines.append(f"  - {col}: {count} ({pct:.1f}%)")
            lines.append("")
        
        if profile.target_distribution.get("type") == "classification":
            lines.append(f"Classes: {profile.target_distribution['n_classes']}")
            for cls, count in list(profile.target_distribution['class_counts'].items())[:5]:
                lines.append(f"  - {cls}: {count}")
        else:
            lines.append(f"Target range: {profile.target_distribution['min']:.2f} - {profile.target_distribution['max']:.2f}")
            lines.append(f"Target mean: {profile.target_distribution['mean']:.2f}")
            if profile.target_distribution['skewness'] > 1:
                lines.append(f"Note: Target is right-skewed (skewness={profile.target_distribution['skewness']:.2f})")
        
        return "\n".join(lines)
```

---

## 6. Code Generator Templates

### templates/sklearn_regressor.py.jinja

```python
"""Auto-generated experiment script."""

import json
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
{% if model_type == "RandomForestRegressor" %}
from sklearn.ensemble import RandomForestRegressor
{% elif model_type == "GradientBoostingRegressor" %}
from sklearn.ensemble import GradientBoostingRegressor
{% elif model_type == "LinearRegression" %}
from sklearn.linear_model import LinearRegression
{% elif model_type == "Ridge" %}
from sklearn.linear_model import Ridge
{% endif %}

# Configuration
DATA_PATH = "{{ data_path }}"
TARGET_COLUMN = "{{ target_column }}"
RANDOM_STATE = {{ random_state }}

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

{% if log_transform_target %}
# Log transform target (handling zeros)
y = np.log1p(y)
{% endif %}

# Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='{{ numeric_impute_strategy | default("median") }}')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Model
model = {{ model_type }}(
{% for key, value in model_params.items() %}
    {{ key }}={{ value }},
{% endfor %}
    random_state=RANDOM_STATE
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
rmse_scores = -cv_scores

# Output results
results = {
    "rmse_mean": float(rmse_scores.mean()),
    "rmse_std": float(rmse_scores.std()),
    "r2_mean": float(cross_val_score(pipeline, X, y, cv=5, scoring='r2').mean()),
}

print(json.dumps(results))
```

---

## 7. Prompt Templates

### Experiment Designer Prompt

```python
EXPERIMENT_DESIGNER_SYSTEM = """You are an expert ML experiment designer. Your role is to design the next experiment based on the current state of an automated ML pipeline.

You have deep knowledge of:
- Machine learning algorithms and when to use them
- Feature engineering techniques
- Hyperparameter tuning strategies
- Common pitfalls like overfitting and data leakage

Your experiment designs should be:
1. Informed by previous results - learn from what worked
2. Strategic - explore new directions, don't repeat failures
3. Practical - implementable with sklearn, XGBoost, or LightGBM
4. Well-reasoned - explain your thinking clearly

Always respond with valid JSON matching the required schema."""


EXPERIMENT_DESIGNER_PROMPT = """## Current State

### Dataset Profile
{data_profile}

### Experiments Completed ({n_experiments} total)
{experiment_history}

### Current Best Performance
- Primary Metric: {primary_metric}
- Best Value: {best_value}
- Achieved By: {best_experiment_name}

### Hypotheses from Previous Iteration
{previous_hypotheses}

### User Constraints
{constraints}

## Available Models
- sklearn: LinearRegression, Ridge, RandomForestRegressor, GradientBoostingRegressor
- sklearn: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
{additional_models}

## Your Task

Design the next experiment. Consider:
1. What approaches haven't been tried yet?
2. What patterns in the results suggest promising directions?
3. Are there preprocessing opportunities?
4. Should we try different models or hyperparameters?

## Output Format

Respond with ONLY a JSON object:

{{
  "experiment_name": "descriptive_name_no_spaces",
  "hypothesis": "What you expect to learn",
  "model_type": "ExactModelClassName",
  "model_params": {{}},
  "preprocessing": {{
    "numeric_impute_strategy": "median",
    "log_transform_target": false
  }},
  "reasoning": "Why you chose this configuration"
}}"""
```

### Hypothesis Generator Prompt

```python
HYPOTHESIS_GENERATOR_PROMPT = """## Latest Experiment Results

### Experiment: {experiment_name}
{experiment_details}

### Metrics
{metrics}

### Comparison to Previous Best
- Previous Best: {previous_best}
- This Experiment: {current_value}
- Improvement: {improvement}

### Full Experiment History
{experiment_history}

## Your Task

Analyze these results:

1. **Why did this experiment perform this way?**
   - What factors contributed to the results?
   - Any surprises?

2. **What should we try next?**
   - What directions look promising?
   - What hasn't been explored?

3. **Should we continue iterating?**
   - Are we seeing diminishing returns?
   - Have we explored sufficiently?

Respond with JSON:

{{
  "performance_analysis": "Why this experiment performed as it did",
  "key_insights": ["insight1", "insight2"],
  "next_directions": ["direction1", "direction2"],
  "continue_recommendation": true,
  "confidence": 0.8,
  "reasoning": "Why continue or stop"
}}"""
```

---

## 8. Console Output Utilities

### utils/display.py

```python
"""Rich console output utilities."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def print_header(text: str):
    """Print a styled header."""
    console.print(Panel(text, style="bold blue"))


def print_success(text: str):
    """Print a success message."""
    console.print(f"[green]✓[/green] {text}")


def print_error(text: str):
    """Print an error message."""
    console.print(f"[red]✗[/red] {text}")


def print_iteration_start(iteration: int, max_iterations: int, turn_count: int):
    """Print iteration header with thought signature indicator."""
    header = f"🧠 ITERATION {iteration}/{max_iterations}"
    status = f"🔗 Thought Signature Active | Context: {turn_count} turns"
    console.print(Panel(
        status,
        title=header,
        border_style="blue"
    ))


def print_reasoning(reasoning: str):
    """Print Gemini's reasoning."""
    console.print(Panel(
        reasoning,
        title="💭 Gemini's Reasoning",
        border_style="cyan"
    ))


def print_experiment_result(name: str, metrics: dict, is_best: bool = False):
    """Print experiment results."""
    table = Table(title=f"Results: {name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green" if is_best else "white")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            table.add_row(metric, f"{value:.4f}")
        else:
            table.add_row(metric, str(value))
    
    if is_best:
        console.print(Panel(table, border_style="green", title="🏆 New Best!"))
    else:
        console.print(table)


def create_progress():
    """Create a progress context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
```

---

## 9. Model Registry

### execution/model_registry.py

```python
"""Model registry with configurations."""

MODEL_REGISTRY = {
    # Regression - sklearn
    "LinearRegression": {
        "module": "sklearn.linear_model",
        "task_type": "regression",
        "default_params": {},
        "tunable_params": {}
    },
    "Ridge": {
        "module": "sklearn.linear_model",
        "task_type": "regression",
        "default_params": {"random_state": 42},
        "tunable_params": {"alpha": [0.1, 1.0, 10.0, 100.0]}
    },
    "RandomForestRegressor": {
        "module": "sklearn.ensemble",
        "task_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "GradientBoostingRegressor": {
        "module": "sklearn.ensemble",
        "task_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    
    # Classification - sklearn
    "LogisticRegression": {
        "module": "sklearn.linear_model",
        "task_type": "classification",
        "default_params": {"random_state": 42, "max_iter": 1000},
        "tunable_params": {"C": [0.1, 1.0, 10.0]}
    },
    "RandomForestClassifier": {
        "module": "sklearn.ensemble",
        "task_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None]
        }
    },
    "GradientBoostingClassifier": {
        "module": "sklearn.ensemble",
        "task_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    
    # Phase 2 - XGBoost
    "XGBRegressor": {
        "module": "xgboost",
        "task_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "XGBClassifier": {
        "module": "xgboost",
        "task_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    
    # Phase 2 - LightGBM
    "LGBMRegressor": {
        "module": "lightgbm",
        "task_type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "LGBMClassifier": {
        "module": "lightgbm",
        "task_type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbosity": -1},
        "tunable_params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
}


def get_models_for_task(task_type: str) -> list:
    """Get all models suitable for a given task type."""
    return [name for name, cfg in MODEL_REGISTRY.items() 
            if cfg["task_type"] == task_type]


def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]
```

---

## 10. Quick Start Commands

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd ml-experiment-autopilot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Gemini API key

# 3. Download sample data
mkdir -p data/sample
# Download House Prices from Kaggle to data/sample/

# 4. Run tests
pytest tests/ -v

# 5. Start MLflow UI (separate terminal)
mlflow ui --backend-store-uri ./outputs/mlruns

# 6. Run the autopilot
python -m src.main run \
  --data data/sample/house_prices_train.csv \
  --target SalePrice \
  --task regression \
  --verbose
```

---

*Technical Reference v2.0*
*Updated: January 2026*
*All decisions finalized*
