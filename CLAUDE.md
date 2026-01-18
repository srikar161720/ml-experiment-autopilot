# CLAUDE.md - ML Experiment Autopilot

> **Purpose**: This file provides Claude Code with comprehensive context to effectively help build the ML Experiment Autopilot project for the Gemini 3 Hackathon.

---

## Project Overview

**ML Experiment Autopilot** is an autonomous agent that designs, executes, and iterates on machine learning experiments without human supervision. It uses Google's Gemini 3 API with Thought Signatures to maintain reasoning continuity across hundreds of API calls.

**Hackathon**: Gemini 3 Hackathon by Google DeepMind & Devpost  
**Deadline**: February 9, 2026 @ 5:00pm PST  
**Prize Pool**: $100,000  
**Target Track**: "The Marathon Agent" — autonomous systems for long-running tasks with self-correction

### Key Differentiators from AutoML Tools

| AutoML (H2O, AutoGluon) | ML Experiment Autopilot |
|-------------------------|------------------------|
| Black box | Explains every decision with reasoning |
| No hypothesis testing | Generates and tests hypotheses |
| Generic error messages | Reasons about why experiments fail |
| Auto-generated tables | Publication-ready narrative reports |
| Configuration files | Natural language constraints |

---

## Current Status

- **Phase**: 2 (Intelligence) — Ready to begin
- **Current Focus**: ExperimentDesigner, ResultsAnalyzer, HypothesisGenerator components
- **Last Completed**: Phase 1 Foundation (Session 1, January 17, 2026)
- **Overall Progress**: ~30% (Phase 1 complete, Phases 2-4 remaining)

### Progress Tracking

Update this section as work progresses:

```
[x] Phase 1: Foundation (Days 1-7) ✅ COMPLETE (Day 1)
    [x] Day 1-2: Project setup, Gemini integration
    [x] Day 3-4: Data profiler, code generation
    [x] Day 5-7: Experiment runner, basic loop
[ ] Phase 2: Intelligence (Days 8-14)
    [ ] ExperimentDesigner component
    [ ] ResultsAnalyzer component
    [ ] HypothesisGenerator component
    [ ] Constraints file parsing
    [ ] Multi-turn conversation management
[ ] Phase 3: Robustness & Polish (Days 15-21)
[ ] Phase 4: Demo & Submission (Days 22-25)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ML EXPERIMENT AUTOPILOT                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              ORCHESTRATION LAYER                        │ │
│  │  • ExperimentController - Main loop & state machine     │ │
│  │  • State management - JSON with Pydantic validation     │ │
│  │  • Termination criteria evaluation                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              COGNITIVE CORE (Gemini 3)                  │ │
│  │  • ExperimentDesigner - Designs next experiment         │ │
│  │  • HypothesisGenerator - Analyzes results, hypothesizes │ │
│  │  • ResultsAnalyzer - Compares, detects patterns         │ │
│  │  • ReportGenerator - Creates final Markdown report      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 EXECUTION LAYER                         │ │
│  │  • DataProfiler - Schema, stats, missing values         │ │
│  │  • CodeGenerator - Template-based sklearn/XGBoost code  │ │
│  │  • ExperimentRunner - Subprocess execution, capture     │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                PERSISTENCE LAYER                        │ │
│  │  • MLflow tracking (local)                              │ │
│  │  • JSON state files                                     │ │
│  │  • Artifact storage (models, plots, code)               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

```
Input: Dataset + Problem Statement + (Optional) Constraints
                    │
                    ▼
            Data Profiling
                    │
                    ▼
            Baseline Model
                    │
                    ▼
    ┌───────────────────────────────┐
    │       ITERATION LOOP          │
    │  Design → Generate → Execute  │
    │  → Analyze → Hypothesize →    │
    │  → Decide (continue/stop)     │
    └───────────────────────────────┘
                    │
                    ▼
          Report Generation
                    │
                    ▼
Output: Best Model + Report + MLflow Experiment + Code
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Execution Environment** | Subprocess | Simple, fast to implement, sufficient isolation for hackathon |
| **State Persistence** | JSON + Pydantic | Human-readable, easy debugging, type-safe |
| **Report Format** | Markdown only | Sufficient for demo, avoids PDF complexity |
| **MLflow Hosting** | Local | No network dependency, faster, zero demo risk |
| **CLI Framework** | Typer | Modern, type-hint based, auto-generates help |
| **Problem Type** | User-specified (`--task`) | Explicit is clearer than auto-detection |
| **Metric Selection** | Gemini decides (with fallbacks) | Intelligent default, user can override in constraints |
| **Preprocessing** | Gemini decides per experiment | Flexible, learns from results |

### Demo Datasets (Priority Order)

1. **House Prices (Kaggle)** — Primary, regression, 1460 rows
2. **Titanic (Kaggle)** — Secondary, classification, 891 rows
3. **Credit Card Fraud** — Stretch only (requires class imbalance handling)

---

## Complete Project Structure

```
ml-experiment-autopilot/
├── CLAUDE.md                         # This file - Claude Code context
├── PROGRESS.md                       # Session logs tracking progress (gitignored)
├── README.md                         # Public-facing documentation
├── LICENSE                           # MIT License
├── pyproject.toml                    # Project metadata
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
├── .env                              # Actual environment variables (gitignored)
├── .gitignore                        # Git ignore patterns
│
├── docs/                             # Planning & design documents
│   ├── PROJECT_BRIEF.md              # High-level context, hackathon details
│   ├── TECHNICAL_SPECIFICATION.md    # Detailed architecture, component specs
│   ├── IMPLEMENTATION_PLAN.md        # Phased timeline, tasks, milestones
│   ├── TECHNICAL_REFERENCE.md        # Code snippets, dependencies, templates
│   └── CONTINUATION_GUIDE.md         # Quick orientation for new sessions
│
├── src/                              # Main source code
│   ├── __init__.py
│   ├── main.py                       # CLI entry point (Typer app)
│   ├── config.py                     # Configuration management
│   │
│   ├── orchestration/                # Experiment lifecycle management
│   │   ├── __init__.py
│   │   ├── controller.py             # ExperimentController - main loop
│   │   └── state.py                  # Pydantic state models, save/load
│   │
│   ├── cognitive/                    # Gemini-powered reasoning
│   │   ├── __init__.py
│   │   ├── gemini_client.py          # Gemini API wrapper with retries
│   │   ├── experiment_designer.py    # Designs next experiment
│   │   ├── hypothesis_generator.py   # Analyzes results, generates hypotheses
│   │   ├── results_analyzer.py       # Compares results, detects patterns
│   │   └── report_generator.py       # Creates final Markdown report
│   │
│   ├── execution/                    # Data processing & code execution
│   │   ├── __init__.py
│   │   ├── data_profiler.py          # Dataset analysis
│   │   ├── code_generator.py         # Template-based code generation
│   │   ├── experiment_runner.py      # Subprocess execution
│   │   └── model_registry.py         # Model configurations
│   │
│   ├── persistence/                  # Storage & tracking
│   │   ├── __init__.py
│   │   ├── mlflow_tracker.py         # MLflow integration
│   │   └── artifact_store.py         # Model/plot/code storage
│   │
│   └── utils/                        # Utility Functions
│       ├── __init__.py
│       └── display.py                # Rich console output utilities
│
├── templates/                        # Code generation templates
│   ├── base_experiment.py.jinja      # Base training script template
│   ├── sklearn_classifier.py.jinja
│   ├── sklearn_regressor.py.jinja
│   ├── xgboost_model.py.jinja        # Phase 2
│   └── lightgbm_model.py.jinja       # Phase 2
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── data/                         # Test datasets (small samples)
│   │   ├── house_prices_sample.csv   # ~100 rows for fast testing
│   │   └── titanic_sample.csv        # ~100 rows for fast testing
│   ├── test_data_profiler.py         # Data profiler tests
│   ├── test_code_generator.py        # Code generator tests
│   ├── test_experiment_runner.py     # Experiment runner tests
│   ├── test_gemini_client.py         # Gemini client tests (mocked)
│   ├── test_state.py                 # State management tests
│   └── integration/                  # Integration tests (real API)
│       └── test_full_pipeline.py     # End-to-end tests
│
├── scripts/                          # Helper scripts
│   ├── setup.sh                      # Environment setup
│   ├── download_data.sh              # Download demo datasets
│   ├── run_demo.sh                   # Run demo workflow
│   ├── pre_demo_check.sh             # Pre-demo validation
│   └── clean.sh                      # Clean generated files
│
├── data/                             # Datasets (gitignored except samples)
│   ├── .gitkeep                      # Keep directory in git
│   └── sample/                       # Sample datasets for demos
│       ├── house_prices_train.csv    # Primary demo dataset
│       └── titanic_train.csv         # Secondary demo dataset
│
└── outputs/                          # Generated outputs (gitignored)
    ├── .gitkeep                      # Keep directory in git
    ├── experiments/                  # Generated experiment code
    │   └── .gitkeep
    ├── reports/                      # Generated Markdown reports
    │   └── .gitkeep
    ├── models/                       # Saved model files
    │   └── .gitkeep
    └── mlruns/                       # MLflow tracking directory
        └── .gitkeep
```

---

## CLI Interface

### Command Structure

```bash
# Minimal invocation
autopilot run --data train.csv --target SalePrice --task regression

# Full invocation
autopilot run \
  --data train.csv \
  --target SalePrice \
  --task regression \
  --constraints constraints.md \
  --max-iterations 15 \
  --time-budget 3600 \
  --output-dir ./my_experiment \
  --verbose
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

### Constraints File Format

Optional Markdown file for user preferences:

```markdown
# Experiment Constraints

## Metrics
- Primary metric: RMSE

## Models
- Prefer tree-based models
- Avoid SVM

## Preprocessing
- Log-transform the target variable

## Termination
- Stop if no improvement for 5 iterations
```

---

## Gemini 3 Integration

### Critical Configuration

```python
# ALWAYS use these settings
GEMINI_CONFIG = {
    "model": "gemini-3-pro-preview",  # Gemini 3 Pro Preview (free tier)
    "temperature": 1.0,  # REQUIRED - lower values degrade reasoning
    "thinking_level": "high",  # For complex tasks
}
```

### Thought Signature Rules

1. **Temperature MUST be 1.0** — lower values cause reasoning degradation
2. **Thought Signatures required for function calling** — 400 error if missing
3. **SDK handles signatures automatically** in chat interface
4. Use multi-turn conversation to maintain context

### Thinking Levels

| Level | Use Case |
|-------|----------|
| `high` | Experiment design, hypothesis generation, analysis |
| `medium` | Standard reasoning tasks |
| `low` | Simple tasks, validation |

### Error Handling

- Retry with exponential backoff (max 3 retries)
- Fallback defaults if API fails
- Log all Gemini interactions for debugging

---

## Code Patterns to Follow

### 1. Pydantic for All Data Models

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class ExperimentSpec(BaseModel):
    """Specification for a single experiment."""
    experiment_name: str
    hypothesis: str
    model_type: str
    model_params: dict = Field(default_factory=dict)
    preprocessing: dict = Field(default_factory=dict)
    reasoning: str
```

### 2. Rich for Console Output

```python
from rich.console import Console
from rich.panel import Panel

console = Console()

def print_iteration_header(iteration: int, total: int):
    console.print(Panel(
        f"Iteration {iteration}/{total}",
        style="bold blue"
    ))
```

### 3. All Gemini Calls Through GeminiClient

```python
# NEVER call Gemini API directly
# ALWAYS use the GeminiClient wrapper

from cognitive.gemini_client import GeminiClient

client = GeminiClient()
response = client.generate(
    prompt="Design an experiment...",
    thinking_level="high",
    system_instruction="You are an ML experiment designer..."
)
```

### 4. Template-Based Code Generation

```python
# Use Jinja2 templates, not raw string generation
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("sklearn_regressor.py.jinja")
code = template.render(
    model_type="RandomForestRegressor",
    model_params={"n_estimators": 100},
    # ...
)
```

### 5. Subprocess for Experiment Execution

```python
import subprocess
import json

result = subprocess.run(
    ["python", script_path],
    capture_output=True,
    text=True,
    timeout=timeout_seconds,
    cwd=working_dir
)

# Parse metrics from stdout
metrics = json.loads(result.stdout)
```

---

## Termination Criteria

### Defaults

| Criterion | Default | Can Override via Constraints |
|-----------|---------|------------------------------|
| Max iterations | 20 | Yes |
| Plateau threshold | 3 consecutive no-improvement | Yes |
| Improvement threshold | 0.5% relative | Yes |
| Time budget | 3600 seconds (1 hour) | Yes |
| Target achieved | None | Yes |
| Agent decision | Enabled | Yes |

### Implementation

```python
def should_terminate(state: ExperimentState) -> tuple[bool, str]:
    """Check if experiment loop should terminate."""
    
    # Max iterations
    if state.iteration >= state.config.max_iterations:
        return True, "Maximum iterations reached"
    
    # Time budget
    if time.time() - state.start_time > state.config.time_budget:
        return True, "Time budget exhausted"
    
    # Plateau detection
    if state.iterations_without_improvement >= state.config.plateau_threshold:
        return True, "Performance plateau detected"
    
    # Target achieved
    if state.config.target_metric_value:
        if state.best_metric >= state.config.target_metric_value:
            return True, "Target metric achieved"
    
    # Agent decision
    if state.gemini_recommends_stop:
        return True, "Agent determined further improvement unlikely"
    
    return False, ""
```

---

## Testing Strategy

### Priority Tiers

**Tier 1 (Critical Path)** — Must pass before demo:
- `test_data_profiler_house_prices` — Profiler works on demo dataset
- `test_code_generation_sklearn` — Generated code is valid
- `test_experiment_execution` — Code runs and returns metrics
- `test_gemini_experiment_design` — Gemini returns parseable JSON
- `test_full_loop_3_iterations` — End-to-end works
- `test_mlflow_logging` — Metrics appear in MLflow

**Tier 2 (Robustness)** — Should have:
- `test_gemini_retry_on_rate_limit`
- `test_invalid_gemini_response_handling`
- `test_experiment_timeout`
- `test_state_save_and_resume`

**Tier 3 (Edge Cases)** — Nice to have:
- `test_missing_values_handling`
- `test_high_cardinality_categorical`
- `test_constraints_parsing`

### Running Tests

```bash
# Run all unit tests
pytest tests/ -v --ignore=tests/integration

# Run integration tests (uses real Gemini API)
pytest tests/integration/ -v -m integration

# Run pre-demo check
./scripts/pre_demo_check.sh
```

---

## Things to Avoid

1. **Don't hardcode paths** — Always use `config.py` paths
2. **Don't make raw Gemini API calls** — Use `GeminiClient` wrapper
3. **Don't generate code without validation** — Check syntax before execution
4. **Don't skip error handling** — Every Gemini call needs try/except
5. **Don't use temperature < 1.0** — Degrades Gemini 3 reasoning
6. **Don't store secrets in code** — Use `.env` file
7. **Don't create overly complex preprocessing** — Keep Phase 1 simple
8. **Don't forget MLflow logging** — It's key for the demo

---

## Demo Priorities

### Must Show in Demo

1. **Data profiling output** — Agent understands the dataset
2. **Gemini reasoning** — Agent explains each decision
3. **Iteration progression** — Multiple experiments improving
4. **Thought Signature continuity** — Agent references previous iterations
5. **MLflow dashboard** — All experiments tracked
6. **Final report** — Professional Markdown output

### Thought Signature Visualization

Display reasoning chain showing Gemini maintaining context:

```
═══════════════════════════════════════════════════════════════
🧠 ITERATION 5 - GEMINI'S REASONING
═══════════════════════════════════════════════════════════════
🔗 Thought Signature Active | Context: 12 turns

Based on the previous 4 experiments, I've observed that:
- Tree-based models consistently outperform linear models
- Feature 'OverallQual' has high importance across all models
- Iteration 3's log-transform hypothesis improved RMSE by 8%

For this iteration, I'm testing XGBoost with...
```

---

## Quick Reference Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Gemini API key

# Run the autopilot
python -m src.main run --data data/sample/house_prices_train.csv \
  --target SalePrice --task regression --verbose

# Start MLflow UI
mlflow ui --backend-store-uri ./outputs/mlruns

# Run tests
pytest tests/ -v

# Pre-demo validation
./scripts/pre_demo_check.sh
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Gemini returns invalid JSON | Add retry with prompt refinement, use fallback |
| Experiment times out | Kill subprocess, log partial results, continue |
| MLflow UI not showing experiments | Check `MLFLOW_TRACKING_URI` matches UI `--backend-store-uri` |
| Generated code has syntax error | Validate with `ast.parse()` before execution |
| API rate limit (429) | Exponential backoff, max 3 retries |

---

## Reference Documents

For detailed specifications, see:

- `PROGRESS.md` — Session logs tracking progress and current status
- `docs/PROJECT_BRIEF.md` — High-level context, hackathon details
- `docs/TECHNICAL_SPECIFICATION.md` — Detailed architecture, component specs
- `docs/IMPLEMENTATION_PLAN.md` — Phased timeline, tasks, milestones
- `docs/TECHNICAL_REFERENCE.md` — Code snippets, dependencies, templates
- `docs/CONTINUATION_GUIDE.md` — Quick orientation for new sessions

---

*CLAUDE.md v1.0 — ML Experiment Autopilot*
*Last Updated: January 2026*
