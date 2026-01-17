# ML Experiment Autopilot - Technical Specification

## Document Purpose
This document contains the detailed technical architecture, component specifications, and system design for the ML Experiment Autopilot project.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML EXPERIMENT AUTOPILOT                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   INPUT     │    │              ORCHESTRATION LAYER                 │   │
│  │             │    │  ┌────────────────────────────────────────────┐  │   │
│  │ • Dataset   │───▶│  │         Experiment Loop Controller         │  │   │
│  │ • Target    │    │  │  • State Management (JSON + Pydantic)      │  │   │
│  │ • Task Type │    │  │  • Thought Signature Circulation           │  │   │
│  │ • Constraints│   │  │  • Termination Criteria Evaluation         │  │   │
│  │   (optional)│    │  │  • Error Recovery & Retry Logic            │  │   │
│  └─────────────┘    │  └────────────────────────────────────────────┘  │   │
│                     └──────────────────────────────────────────────────┘   │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        COGNITIVE CORE (Gemini 3)                     │  │
│  │                                                                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│  │  │  Experiment │  │  Hypothesis │  │   Results   │  │   Report   │  │  │
│  │  │   Designer  │  │  Generator  │  │  Analyzer   │  │  Generator │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │  │
│  │                                                                      │  │
│  │  [Thought Signatures maintained across all cognitive operations]     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         EXECUTION LAYER                              │  │
│  │                                                                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │  │
│  │  │    Data      │  │     Code     │  │    Experiment Runner     │   │  │
│  │  │   Profiler   │  │   Generator  │  │    (Subprocess)          │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        PERSISTENCE LAYER                             │  │
│  │                                                                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │  │
│  │  │   MLflow     │  │  JSON State  │  │      Artifact Store      │   │  │
│  │  │  (Local)     │  │   Files      │  │  (Models, Plots, Code)   │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                            OUTPUT                                    │  │
│  │  • Final trained model    • MLflow experiment with full history     │  │
│  │  • Markdown report        • Reproducible training code              │  │
│  │  • Visualizations         • Reasoning documentation                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CLI Interface Specification

### Command Structure

```bash
autopilot run [OPTIONS]
```

### Arguments

| Argument | Required | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `--data`, `-d` | Yes | Path | — | Path to dataset (CSV/Parquet) |
| `--target`, `-t` | Yes | String | — | Target column name |
| `--task` | Yes | Choice | — | `classification` or `regression` |
| `--constraints`, `-c` | No | Path | None | Path to constraints file (Markdown) |
| `--max-iterations`, `-n` | No | Int | 20 | Maximum experiment iterations |
| `--time-budget` | No | Int | 3600 | Time budget in seconds |
| `--output-dir`, `-o` | No | Path | Auto | Output directory |
| `--verbose`, `-v` | No | Flag | False | Show detailed Gemini reasoning |
| `--resume` | No | Path | None | Resume from saved state file |

### Implementation

```python
# src/main.py
import typer
from pathlib import Path
from typing import Optional
from enum import Enum

app = typer.Typer(
    name="autopilot",
    help="ML Experiment Autopilot - Autonomous ML experimentation powered by Gemini"
)

class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"

@app.command()
def run(
    data: Path = typer.Option(..., "--data", "-d", help="Path to dataset"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    task: TaskType = typer.Option(..., "--task", help="classification or regression"),
    constraints: Optional[Path] = typer.Option(None, "--constraints", "-c"),
    max_iterations: int = typer.Option(20, "--max-iterations", "-n"),
    time_budget: int = typer.Option(3600, "--time-budget"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
):
    """Run the ML Experiment Autopilot on a dataset."""
    # Implementation here
    pass

if __name__ == "__main__":
    app()
```

---

## 3. Component Specifications

### 3.1 Orchestration Layer

**Purpose**: Manages the experiment lifecycle, maintains state, coordinates components.

#### State Machine

```python
from enum import Enum

class ExperimentPhase(Enum):
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
```

#### Termination Criteria

| Criterion | Default | Override Method |
|-----------|---------|-----------------|
| Max iterations | 20 | `--max-iterations` or constraints file |
| Plateau threshold | 3 consecutive iterations | Constraints file |
| Improvement threshold | 0.5% relative | Constraints file |
| Time budget | 3600 seconds | `--time-budget` or constraints file |
| Target metric | None | Constraints file |
| Agent decision | Enabled | Constraints file |

#### Implementation

```python
def should_terminate(state: ExperimentState) -> tuple[bool, str]:
    """Check if experiment loop should terminate."""
    
    if state.iteration >= state.config.max_iterations:
        return True, "Maximum iterations reached"
    
    elapsed = time.time() - state.start_time
    if elapsed > state.config.time_budget:
        return True, "Time budget exhausted"
    
    if state.iterations_without_improvement >= state.config.plateau_threshold:
        return True, "Performance plateau detected"
    
    if state.config.target_value and state.best_metric >= state.config.target_value:
        return True, "Target metric achieved"
    
    if state.agent_recommends_stop:
        return True, "Agent determined further improvement unlikely"
    
    return False, ""
```

#### Error Recovery Strategy
- Retry failed API calls with exponential backoff (max 3 retries)
- Checkpoint state before each major operation
- Graceful degradation: skip failed experiment, log error, continue
- Catastrophic failure: save state, generate partial report, exit cleanly

---

### 3.2 Cognitive Core (Gemini 3 Integration)

**Purpose**: The "brain" of the system using Gemini 3 for all reasoning tasks.

#### Critical Configuration

```python
GEMINI_CONFIG = {
    "model": "gemini-3-pro-preview",  # Update with actual model string
    "temperature": 1.0,  # REQUIRED - do not lower
    "default_thinking_level": "high",
}
```

#### Cognitive Components

| Component | Purpose | Thinking Level |
|-----------|---------|----------------|
| **ExperimentDesigner** | Designs next experiment based on history | `high` |
| **HypothesisGenerator** | Analyzes results, generates hypotheses | `high` |
| **ResultsAnalyzer** | Compares results, detects patterns | `medium` to `high` |
| **ReportGenerator** | Creates final Markdown report | `high` |

#### Metric Selection Logic

```
User specifies --task
        │
        ▼
┌───────────────────────┐
│ Constraints provided? │
└───────────────────────┘
      │           │
     Yes          No
      │           │
      ▼           ▼
 Parse for    Gemini selects
 metric       based on data
 preference   profile
      │           │
      ▼           ▼
 ┌─────────┐     │
 │ Found?  │     │
 └─────────┘     │
   │     │       │
  Yes    No      │
   │     └───────┤
   ▼             ▼
 Use user's   Use Gemini's
 preference   selection
```

**Fallback Defaults** (if Gemini API fails):
- Regression: RMSE
- Classification: F1

---

### 3.3 Execution Layer

#### Data Profiler

**Input**: Dataset path  
**Output**: Structured profile (JSON)

**Profile Contents**:
- Schema: column names, types, target identification
- Statistics: mean, median, std, min, max, quartiles
- Missing values: count and percentage per column
- Categorical: unique values, cardinality
- Correlations: correlation matrix for numeric columns
- Distribution: skewness, kurtosis for numeric columns

#### Code Generator

**Strategy**: Template-based with Jinja2

**Supported Operations (Phase 1)**:

| Category | Operations |
|----------|------------|
| Missing Values | Drop rows, impute (mean/median/mode/constant) |
| Numeric Scaling | StandardScaler, MinMaxScaler, None |
| Categorical Encoding | OneHotEncoder, OrdinalEncoder |
| Target Transform | Log transform (regression) |

**Phase 2 Additions**:
- Feature selection (variance threshold, correlation filter)
- Polynomial features (degree 2)

#### Experiment Runner

**Method**: Subprocess execution

```python
import subprocess
import json

def run_experiment(script_path: Path, timeout: int) -> dict:
    """Execute experiment script and capture results."""
    result = subprocess.run(
        ["python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=script_path.parent
    )
    
    if result.returncode != 0:
        raise ExperimentError(result.stderr)
    
    # Parse metrics from stdout (JSON format)
    return json.loads(result.stdout)
```

---

### 3.4 Persistence Layer

#### State Management (JSON + Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExperimentResult(BaseModel):
    """Results from a single experiment."""
    experiment_name: str
    iteration: int
    model_type: str
    model_params: dict
    metrics: dict
    hypothesis: str
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ExperimentState(BaseModel):
    """Complete state of an experiment session."""
    session_id: str
    data_path: str
    target_column: str
    task_type: str
    constraints: Optional[str] = None
    
    data_profile: Optional[dict] = None
    experiments: List[ExperimentResult] = Field(default_factory=list)
    
    current_iteration: int = 0
    best_metric: Optional[float] = None
    best_experiment: Optional[str] = None
    iterations_without_improvement: int = 0
    
    start_time: float = Field(default_factory=time.time)
    phase: str = "initializing"
    
    gemini_conversation_history: List[dict] = Field(default_factory=list)
```

#### MLflow Integration (Local)

```python
import mlflow

def setup_mlflow(experiment_name: str, tracking_uri: str = "./outputs/mlruns"):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_experiment(result: ExperimentResult):
    """Log experiment to MLflow."""
    with mlflow.start_run(run_name=result.experiment_name):
        mlflow.log_params(result.model_params)
        mlflow.log_metrics(result.metrics)
        mlflow.set_tag("hypothesis", result.hypothesis)
        mlflow.set_tag("reasoning", result.reasoning)
```

---

## 4. Workflow Diagram

```
┌─────────────────────┐
│  1. DATA PROFILING  │ ◄─── Single execution at start
│                     │
│  • Load dataset     │
│  • Generate profile │
│  • Identify issues  │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ 2. BASELINE MODEL   │ ◄─── Establish performance floor
│                     │
│  • Simple model     │
│  • Default params   │
│  • Record baseline  │
└─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              ITERATION LOOP                         │
│                                                     │
│  ┌─────────────────────┐                           │
│  │ 3. EXPERIMENT       │                           │
│  │    DESIGN           │                           │
│  │                     │                           │
│  │ Gemini designs next │                           │
│  │ experiment          │                           │
│  └─────────────────────┘                           │
│            │                                        │
│            ▼                                        │
│  ┌─────────────────────┐                           │
│  │ 4. CODE GENERATION  │                           │
│  │                     │                           │
│  │ Generate Python     │                           │
│  │ training script     │                           │
│  └─────────────────────┘                           │
│            │                                        │
│            ▼                                        │
│  ┌─────────────────────┐                           │
│  │ 5. EXECUTION        │                           │
│  │                     │                           │
│  │ Run in subprocess   │──▶ [MLflow Logging]       │
│  │ Capture results     │                           │
│  └─────────────────────┘                           │
│            │                                        │
│            ▼                                        │
│  ┌─────────────────────┐                           │
│  │ 6. RESULTS ANALYSIS │                           │
│  │                     │                           │
│  │ • Compare to prior  │                           │
│  │ • Detect patterns   │                           │
│  │ • Update best       │                           │
│  └─────────────────────┘                           │
│            │                                        │
│            ▼                                        │
│  ┌─────────────────────┐                           │
│  │ 7. HYPOTHESIS       │                           │
│  │    GENERATION       │                           │
│  │                     │                           │
│  │ • Why did this work?│                           │
│  │ • What to try next? │                           │
│  └─────────────────────┘                           │
│            │                                        │
│            ▼                                        │
│  ┌─────────────────────┐     ┌──────────────────┐  │
│  │ 8. CONTINUE?        │─NO─▶│ EXIT LOOP        │  │
│  │                     │     │                  │  │
│  │ Check termination   │     │ • Max iterations │  │
│  │ criteria            │     │ • Plateau        │  │
│  └─────────────────────┘     │ • Time budget    │  │
│            │                 │ • Target met     │  │
│           YES                │ • Agent decision │  │
│            │                 └──────────────────┘  │
│            └───────────────────────────────────────┤
│                                                    │
└────────────────────────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │ 9. REPORT GENERATION    │
                │                         │
                │ • Executive summary     │
                │ • Methodology           │
                │ • Results & insights    │
                │ • Recommendations       │
                └─────────────────────────┘
                              │
                              ▼
                             END
```

---

## 5. Constraints File Specification

### Format

Markdown file with optional sections:

```markdown
# Experiment Constraints

## Metrics
- Primary metric: RMSE
- Also track: R², MAE

## Models
- Prefer: tree-based models (RandomForest, XGBoost)
- Avoid: SVM (too slow)

## Preprocessing
- Log-transform the target variable
- Use median imputation for missing values

## Termination
- Stop if RMSE doesn't improve by 0.01 for 5 iterations
- Maximum 15 iterations

## Other
- Final model must be interpretable
```

### Parsing

Gemini receives the constraints as context and incorporates them into experiment design decisions.

---

## 6. Supported Models

### Phase 1 (MVP)

| Library | Models | Task Types |
|---------|--------|------------|
| scikit-learn | LogisticRegression, RandomForestClassifier, GradientBoostingClassifier | Classification |
| scikit-learn | LinearRegression, RandomForestRegressor, GradientBoostingRegressor | Regression |

### Phase 2 (Enhanced)

| Library | Models | Task Types |
|---------|--------|------------|
| XGBoost | XGBClassifier, XGBRegressor | Both |
| LightGBM | LGBMClassifier, LGBMRegressor | Both |

---

## 7. Error Handling

### Error Categories

```python
class AutopilotError(Exception):
    """Base exception."""
    pass

class DataError(AutopilotError):
    """Data loading or quality errors."""
    pass

class ExperimentError(AutopilotError):
    """Experiment execution errors."""
    pass

class GeminiError(AutopilotError):
    """Gemini API errors."""
    pass

class StateError(AutopilotError):
    """State management errors."""
    pass
```

### Recovery Strategies

| Error Type | Strategy |
|------------|----------|
| Gemini 429 (Rate Limit) | Exponential backoff, max 3 retries |
| Gemini 400 (Bad Request) | Log error, use fallback/skip |
| Experiment Timeout | Kill process, log partial, continue |
| Experiment Runtime Error | Log error, try alternative |
| Data Loading Error | Fatal - report to user, exit |
| Invalid Gemini JSON | Retry with refined prompt, then fallback |

---

## 8. Thought Signature Demonstration

### Visualization Strategy

Display reasoning chain showing Gemini maintaining context:

```
═══════════════════════════════════════════════════════════════
🧠 ITERATION 5 - GEMINI'S REASONING
═══════════════════════════════════════════════════════════════
🔗 Thought Signature Active | Context: 12 turns | 5 iterations

Based on the previous 4 experiments, I've observed that:
- Tree-based models consistently outperform linear models
- Feature 'OverallQual' has high importance across all models
- Iteration 3's log-transform hypothesis improved RMSE by 8%

For this iteration, I'm testing XGBoost with tuned hyperparameters...
```

### Implementation

```python
def display_reasoning(iteration: int, reasoning: str, turn_count: int):
    """Display Gemini's reasoning with thought signature indicator."""
    console.print(Panel(
        f"🔗 Thought Signature Active | Context: {turn_count} turns\n\n{reasoning}",
        title=f"🧠 ITERATION {iteration} - GEMINI'S REASONING",
        border_style="blue"
    ))
```

---

*Technical Specification v2.0*
*Updated: January 2026*
*All decisions finalized*
