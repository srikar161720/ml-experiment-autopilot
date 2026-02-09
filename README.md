# ML Experiment Autopilot

> **Autonomous ML experimentation powered by Google Gemini 3** — designs, executes, and iterates on machine learning experiments without human supervision, explaining every decision along the way.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-160_passing-brightgreen)
![Gemini 3](https://img.shields.io/badge/Gemini_3-Flash_Preview-4285F4?logo=google&logoColor=white)
![Track](https://img.shields.io/badge/Track-The_Marathon_Agent-orange)

**Gemini 3 Hackathon** by Google DeepMind & Devpost

---

## Table of Contents

- [Why This Matters](#why-this-matters-beyond-black-box-automl)
- [See It In Action](#see-it-in-action)
- [Architecture](#architecture)
- [Gemini 3 Integration](#gemini-3-integration-the-marathon-agent)
- [Quick Start](#quick-start)
- [CLI Reference](#command-line-interface)
- [How It Works](#how-it-works-the-experiment-loop)
- [Features](#features)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [License](#license)

---

## Why This Matters: Beyond Black-Box AutoML

| AutoML Tools (H2O, AutoGluon) | ML Experiment Autopilot |
|-------------------------------|------------------------|
| Black box model selection | Explains every decision with Gemini reasoning |
| No hypothesis testing | Generates and tests data-driven hypotheses |
| Generic "model trained" messages | Reasons about *why* experiments fail or succeed |
| Auto-generated performance tables | Publication-ready narrative reports |
| Configuration files | Natural language constraints |

---

## See It In Action

When you run the autopilot with `--verbose`, you see the agent's reasoning process in real time:

```
╔══════════════════════════════════════════════════════════════╗
║  ITERATION 3 - GEMINI'S REASONING                            ║
║  Thought Signature Active | Context: 12 turns                ║
╚══════════════════════════════════════════════════════════════╝

Based on the previous 2 experiments, I've observed that:
- Tree-based models consistently outperform linear models on this dataset
- Iteration 2's log-transform hypothesis improved RMSE by 80%
- Feature distributions suggest boosting may capture residual patterns

For this iteration, I'm testing XGBoost with tuned learning rate
and max_depth to see if gradient boosting further reduces error...

┌─────────────────────────────────────────────────────────────┐
│ RESULTS ANALYSIS                                            │
├─────────────────────────────────────────────────────────────┤
│ Trend: IMPROVING                                            │
│ RMSE: 0.1332   ★ NEW BEST                                   │
│   82.1% better than baseline                                │
│                                                             │
│ Key Observations:                                           │
│   - Boosting provided 10.3% improvement over bagging        │
│   - Log transformation remains critical for this target     │
│   - Diminishing returns suggest stopping after next round   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ HYPOTHESES FOR NEXT ITERATION                               │
├─────────────────────────────────────────────────────────────┤
│ Strategy: exploit                                           │
│                                                             │
│ 1. [Priority: 1] Fine-tune XGBoost regularization           │
│    Confidence: 0.72 | Models: XGBRegressor                  │
│                                                             │
│ 2. [Priority: 2] Try LightGBM as alternative booster        │
│    Confidence: 0.65 | Models: LGBMRegressor                 │
└─────────────────────────────────────────────────────────────┘
```

> A working demo of the CLI can be viewed at [https://www.youtube.com/watch?v=BM2VCaRToNE](https://www.youtube.com/watch?v=BM2VCaRToNE).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ML EXPERIMENT AUTOPILOT                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 ORCHESTRATION LAYER                    │ │
│  │ ExperimentController — main loop & state machine       │ │
│  │ Pydantic state management — JSON with type validation  │ │
│  │ Termination criteria — plateau, budget, agent decision │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           COGNITIVE CORE  (Gemini 3 Flash)             │ │
│  │                                                        │ │
│  │  ExperimentDesigner — designs next experiment          │ │
│  │  ResultsAnalyzer — compares results, detects trends    │ │
│  │  HypothesisGenerator — hypotheses with confidence      │ │
│  │  ReportGenerator — publication-ready narrative reports │ │
│  │                                                        │ │
│  │  Thought Signatures maintain reasoning continuity      │ │
│  │  across all iterations via shared conversation history │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 EXECUTION LAYER                        │ │
│  │  DataProfiler — schema, stats, missing values          │ │
│  │  CodeGenerator — Jinja2 template-based Python scripts  │ │
│  │  ExperimentRunner — subprocess execution with timeout  │ │
│  │  VisualizationGenerator — matplotlib charts            │ │
│  └────────────────────────────────────────────────────────┘ │
│                             │                               │
│                             ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  PERSISTENCE LAYER                     │ │
│  │  MLflow tracking (local) — metrics, params, artifacts  │ │
│  │  JSON state files — resumable experiment sessions      │ │
│  │  Artifact storage — models, plots, generated code      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Gemini 3 Integration: The Marathon Agent

This project leverages Gemini 3's unique capabilities for long-running autonomous tasks.

### Thought Signatures

All four cognitive components share a single `GeminiClient` instance, maintaining a multi-turn conversation across the entire experiment session. This means Gemini 3 can reference results from iteration 1 when designing iteration 10 — true reasoning continuity across 100+ API calls.

- **Temperature 1.0** — required for high-quality reasoning (per Gemini 3 best practices)
- **`thinking_level: high`** — used for experiment design, analysis, and hypothesis generation

### Four Cognitive Components

| Component | Role | Output |
|-----------|------|--------|
| **ExperimentDesigner** | Designs next experiment based on data profile, history, and constraints | Structured JSON: model, hyperparameters, preprocessing |
| **ResultsAnalyzer** | Compares current results against baseline and best | Trend detection, metric comparison, observations |
| **HypothesisGenerator** | Synthesizes all iterations into ranked next steps | Hypotheses with confidence scores, explore/exploit strategy |
| **ReportGenerator** | Writes final narrative report | Markdown with executive summary, methodology, insights |

### Why This Qualifies for "The Marathon Agent"

- **Autonomous**: Runs 20+ iterations without human intervention
- **Long-Running**: Maintains context across multi-hour execution via Thought Signatures
- **Self-Correcting**: Learns from failures, adjusts strategy, detects performance plateaus
- **Explainable**: Every decision is documented with Gemini's reasoning

---

## Quick Start

### Prerequisites

- Python 3.9+
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier works, but Tier 1 or higher is recommended)

### 1. Setup

```bash
git clone https://github.com/srikar161720/ml-experiment-autopilot.git
cd ml-experiment-autopilot

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
```

Edit `.env` and set your key:

```
GEMINI_API_KEY=your_actual_key_here
```

### 3. Run Your First Experiment

**Regression** (California Housing — 20,640 samples):

```bash
python -m src.main run \
  --data data/sample/california_housing.csv \
  --target MedHouseVal \
  --task regression \
  --max-iterations 3 \
  --verbose
```

**Classification** (Bank Marketing — 11,162 samples):

```bash
python -m src.main run \
  --data data/sample/bank.csv \
  --target deposit \
  --task classification \
  --max-iterations 3 \
  --verbose
```

### 4. View Results

```bash
# MLflow dashboard
mlflow ui --backend-store-uri file:./outputs/mlruns
# Open http://127.0.0.1:5000
```

**Generated outputs:**

| Output | Location |
|--------|----------|
| Markdown reports | `outputs/reports/` |
| Visualizations | `outputs/plots/` |
| Generated experiment code | `outputs/experiments/` |
| MLflow tracking data | `outputs/mlruns/` |

---

## Command-Line Interface

```bash
python -m src.main run --data <path> --target <column> --task <type> [options]
```

### Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--data` | `-d` | Yes | — | Path to dataset (CSV or Parquet) |
| `--target` | `-t` | Yes | — | Target column name |
| `--task` | — | Yes | — | `classification` or `regression` |
| `--constraints` | `-c` | No | None | Path to constraints file (Markdown) |
| `--max-iterations` | `-n` | No | 20 | Maximum experiment iterations (1–100) |
| `--time-budget` | — | No | 3600 | Time budget in seconds (60–86400) |
| `--output-dir` | `-o` | No | Auto | Custom output directory |
| `--verbose` | `-v` | No | False | Show detailed Gemini reasoning |
| `--resume` | — | No | None | Resume from a saved state file |

### Constraints File

Guide Gemini with natural language preferences in a Markdown file:

```markdown
# Experiment Constraints

## Metrics
- Primary metric: RMSE

## Models
- Prefer tree-based models
- Prefer boosting methods

## Preprocessing
- Log-transform the target variable
- Use median imputation for missing values

## Termination
- Stop if no improvement for 3 iterations
```

```bash
python -m src.main run \
  --data data/sample/california_housing.csv \
  --target MedHouseVal \
  --task regression \
  --constraints data/sample/constraints.md \
  --max-iterations 5 \
  --verbose
```

---

## How It Works: The Experiment Loop

```
Input: Dataset + Target + Task Type + (optional) Constraints
                    │
                    ▼
            ┌────────────────┐
            │ DATA PROFILING │  Analyze schema, distributions, missing values
            └───────┬────────┘
                    │
                    ▼
            ┌────────────────┐
            │ BASELINE MODEL │  Simple model to establish performance floor
            └───────┬────────┘
                    │
                    ▼
    ┌────────────────────────────────────┐
    │          ITERATION LOOP            │
    │                                    │
    │  1. Experiment Design (Gemini)     │  ← hypothesis, model, params
    │  2. Code Generation (Jinja2)       │  ← validated Python script
    │  3. Execution (subprocess)         │  ← train, evaluate, capture metrics
    │  4. Results Analysis (Gemini)      │  ← trends, comparisons, insights
    │  5. Hypothesis Generation (Gemini) │  ← ranked next steps
    │  6. Termination Check              │  ← continue or stop?
    │                                    │
    │  Repeat until termination...       │
    └───────────────┬────────────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │ REPORT GENERATION │  Gemini writes narrative Markdown report
          └─────────┬─────────┘
                    │
                    ▼
Output: Best Model + Report + Visualizations + MLflow Experiment + Code
```

### Termination Criteria

The agent decides when to stop based on multiple signals:

| Criterion | Default | Configurable |
|-----------|---------|--------------|
| Max iterations | 20 | `--max-iterations` |
| Performance plateau | 3 consecutive non-improvements | Via constraints |
| Time budget | 3600 seconds | `--time-budget` |
| Target metric achieved | None | Via constraints |
| Agent recommendation | Enabled | Via constraints |

---

## Features

### Supported Models

- **scikit-learn**: LinearRegression, LogisticRegression, RandomForest, GradientBoosting, SVM, and more
- **XGBoost**: XGBRegressor, XGBClassifier
- **LightGBM**: LGBMRegressor, LGBMClassifier

### Intelligent Experiment Design

- **Hypothesis-driven**: Every experiment tests a specific, stated hypothesis
- **Context-aware**: Considers data profile, all previous results, and user constraints
- **Adaptive**: Learns from failures and adjusts strategy (explore vs. exploit)
- **Explainable**: Full reasoning provided for every decision

### Automatic Preprocessing

Gemini decides per experiment — no fixed pipeline:

- Missing value handling (drop, mean, median, mode imputation)
- Feature scaling (standard, min-max, or none)
- Categorical encoding (one-hot, ordinal)
- Target transformations (log, sqrt for skewed distributions)

### Rich Console Output

Colored, panel-based terminal output via the Rich library:

- Blue panels for iteration headers and configuration
- Magenta panels for Gemini's reasoning (with `--verbose`)
- Cyan panels for results analysis and data profiles
- Yellow panels for hypotheses and termination notices
- Green panels for best results and improvements

### Report Generation

Gemini writes a narrative Markdown report containing:

- Executive summary of the experimental journey
- Methodology and approach
- Results table with all experiments
- Best model details with hyperparameters
- Key insights and recommendations
- Embedded visualization charts
- Per-experiment appendix with full details

### Visualization

Auto-generated matplotlib charts saved to `outputs/plots/`:

- Metric progression across iterations (line chart)
- Model comparison (horizontal bar chart)
- Improvement over baseline (bar chart)

### Error Handling

- Graceful recovery from failed experiments (the loop continues)
- State saving on interruption (`Ctrl+C`) for resume capability
- Retry with exponential backoff for Gemini API calls
- Code validation with `ast.parse()` before execution
- Timeout protection for long-running experiment scripts

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | Gemini 3 Flash Preview | Reasoning, design, analysis, reporting |
| CLI | Typer | Type-safe command-line interface |
| Console | Rich | Colored terminal output with panels |
| Validation | Pydantic | Type-safe data models and state |
| Data | pandas, NumPy | Data loading and processing |
| ML | scikit-learn, XGBoost, LightGBM | Model training and evaluation |
| Visualization | matplotlib | Chart generation (headless Agg backend) |
| Tracking | MLflow | Experiment logging, metrics, artifacts |
| Templating | Jinja2 | Code generation from templates |
| Config | python-dotenv | Environment variable management |

---

## Project Structure

```
ml-experiment-autopilot/
├── src/
│   ├── main.py                       # CLI entry point (Typer)
│   ├── config.py                     # Configuration management
│   ├── orchestration/
│   │   ├── controller.py             # ExperimentController — main loop
│   │   └── state.py                  # Pydantic state models
│   ├── cognitive/
│   │   ├── gemini_client.py          # Gemini API wrapper with retries
│   │   ├── experiment_designer.py    # Designs next experiment
│   │   ├── results_analyzer.py       # Compares results, detects trends
│   │   ├── hypothesis_generator.py   # Generates ranked hypotheses
│   │   └── report_generator.py       # Creates narrative Markdown reports
│   ├── execution/
│   │   ├── data_profiler.py          # Dataset analysis
│   │   ├── code_generator.py         # Jinja2 template-based code gen
│   │   ├── experiment_runner.py      # Subprocess execution
│   │   └── visualization_generator.py # matplotlib charts
│   ├── persistence/
│   │   └── mlflow_tracker.py         # MLflow integration
│   └── utils/
│       └── display.py                # Rich console output
├── templates/                        # Jinja2 code generation templates
│   ├── base_experiment.py.jinja
│   ├── sklearn_classifier.py.jinja
│   ├── sklearn_regressor.py.jinja
│   ├── xgboost_model.py.jinja
│   └── lightgbm_model.py.jinja
├── tests/                            # 160 tests
├── data/sample/                      # Demo datasets
│   ├── california_housing.csv        # Regression (20,640 rows)
│   ├── bank.csv                      # Classification (11,162 rows)
│   └── constraints.md               # Sample constraints file
└── outputs/                          # Generated outputs
    ├── experiments/                  # Generated Python scripts
    ├── reports/                      # Markdown reports
    ├── plots/                        # Visualization charts
    ├── models/                       # Saved model files
    └── mlruns/                       # MLflow tracking
```

---

## Testing

```bash
# Run all unit tests
pytest tests/ -v --ignore=tests/integration

# Run integration tests (requires Gemini API key)
pytest tests/integration/ -v -m integration

# Run a specific component's tests
pytest tests/test_data_profiler.py -v
```

### Coverage

| Component | Tests |
|-----------|-------|
| DataProfiler | 11 |
| CodeGenerator | 18 |
| ExperimentRunner | 7 |
| GeminiClient | 11 |
| ExperimentDesigner | 22 |
| ResultsAnalyzer | 21 |
| HypothesisGenerator | 22 |
| ReportGenerator | 24 |
| VisualizationGenerator | 24 |
| **Total** | **160** |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: No module named 'src'` | Run as module: `python -m src.main run ...` |
| `GEMINI_API_KEY not found` | Ensure `.env` file exists with `GEMINI_API_KEY=...` |
| MLflow UI shows no experiments | Verify URI: `mlflow ui --backend-store-uri file:./outputs/mlruns` |
| Experiment timeout | Increase budget: `--time-budget 7200` |
| Gemini rate limit (429) | Automatic retry with exponential backoff (max 3 retries) |
| Generated code syntax error | Inspect `outputs/experiments/` — code is validated with `ast.parse()` |
| Target column not found | Check exact column name (case-sensitive) in your CSV |

---

## Future Work

- **Neural network support** — PyTorch/TensorFlow templates
- **Automated feature engineering** — Gemini-guided feature creation
- **Ensemble selection** — Stacking and blending of top models
- **Parallel execution** — Run multiple experiments concurrently
- **Web dashboard** — Streamlit interface for real-time monitoring
- **Cloud integration** — GCP Vertex AI and AWS SageMaker connectors

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built for the [Gemini 3 Hackathon](https://gemini3hackathon.devpost.com/) by Google DeepMind & Devpost
- Powered by [Google Gemini 3](https://ai.google.dev/gemini-api)
- [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) `california_housing.csv`
- [Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) `bank.csv`

---

**Built with Gemini 3 for The Marathon Agent Track** | **v0.1.0** | **160 Tests Passing**
