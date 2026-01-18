# ML Experiment Autopilot - Implementation Plan

## Document Purpose
This document contains the phased implementation plan, timeline, milestones, and demo strategy for the ML Experiment Autopilot project.

---

## 1. Project Timeline Overview

**Total Available Time**: ~23 days (January 15 - February 9, 2026)

```
Week 1 (Jan 15-21): Foundation
├── Days 1-2: Project setup, Gemini integration
├── Days 3-4: Data profiler, code generation
└── Days 5-7: Experiment runner, basic loop

Week 2 (Jan 22-28): Intelligence
├── Days 8-10: Cognitive components (designer, analyzer)
├── Days 11-12: Hypothesis generation, thought signatures
└── Days 13-14: MLflow integration, state management

Week 3 (Jan 29 - Feb 4): Robustness & Polish
├── Days 15-16: Error handling, edge cases
├── Days 17-18: Multi-model support (XGBoost, LightGBM)
└── Days 19-21: Report generation, visualizations

Week 4 (Feb 5-9): Demo & Submission
├── Days 22-23: Demo preparation, video recording
├── Day 24: Documentation, README polish
└── Day 25: Final submission (Feb 9)
```

---

## 2. Phase 1: Foundation (Days 1-7)

### Goal
Get a basic end-to-end flow working with manual/hardcoded components.

### Day 1-2: Project Setup & Gemini Integration ✅ COMPLETE

**Tasks**:
- [x] Initialize Git repository
- [x] Create project structure (see CLAUDE.md for full tree)
- [x] Set up virtual environment
- [x] Install core dependencies
- [x] Create `.env.example` and `config.py`
- [x] Configure Gemini API authentication
- [x] Implement `GeminiClient` with retry logic
- [x] Test basic Gemini 3 API call with Thought Signatures
- [x] Set up Typer CLI skeleton (`src/main.py`)

**Project Structure** (create these directories):
```
ml-experiment-autopilot/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── docs/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── orchestration/
│   ├── cognitive/
│   ├── execution/
│   └── persistence/
├── templates/
├── tests/
├── scripts/
├── data/sample/
└── outputs/
```

**Deliverables**:
- Working Gemini API connection
- Basic project structure in place
- CLI runs without errors (even if it does nothing yet)

### Day 3-4: Data Profiler & Basic Code Generation ✅ COMPLETE

**Tasks**:
- [x] Implement `DataProfiler` class
  - [x] Schema detection (column names, types)
  - [x] Statistical summary (numeric columns)
  - [x] Missing value analysis
  - [x] Categorical column analysis
  - [x] Target distribution analysis
- [x] Create Jinja2 templates for code generation
  - [x] `base_experiment.py.jinja` - common structure
  - [x] `sklearn_regressor.py.jinja` - regression models
  - [x] `sklearn_classifier.py.jinja` - classification models
- [x] Implement `CodeGenerator` class
  - [x] Load and render templates
  - [x] Support preprocessing configuration
  - [x] Validate generated code (syntax check)
- [x] Download sample datasets (House Prices, Titanic)
- [x] Test with sample dataset

**Deliverables**:
- Data profiler generates complete profile for any tabular dataset
- Code generator produces runnable sklearn scripts
- Generated code passes syntax validation

### Day 5-7: Experiment Runner & Basic Loop ✅ COMPLETE

**Tasks**:
- [x] Implement `ExperimentRunner` class
  - [x] Execute generated Python scripts via subprocess
  - [x] Capture stdout/stderr
  - [x] Parse results JSON from output
  - [x] Handle timeouts
- [x] Implement basic `ExperimentController`
  - [x] Sequential execution: profile → generate → run → collect
  - [x] Simple state tracking (current iteration, results history)
- [x] Implement Pydantic state models (`orchestration/state.py`)
  - [x] `ExperimentResult` model
  - [x] `ExperimentState` model
  - [x] Save/load state to JSON
- [x] Basic MLflow integration
  - [x] Log parameters and metrics
  - [x] Store model artifacts
- [x] End-to-end test with hardcoded experiment sequence

**Deliverables**:
- Can run a single experiment end-to-end
- Results logged to MLflow
- State saved to JSON file

### Phase 1 Milestone ✅ ACHIEVED (Day 1)
✅ **Demo**: Dataset profiled → experiment code generated → executed → results in MLflow

**Completed January 17, 2026** — All Phase 1 tasks completed in a single session with 39 tests passing.

---

## 3. Phase 2: Intelligence (Days 8-14)

### Goal
Add Gemini-powered reasoning to guide the experiment process.

### Day 8-10: Cognitive Components

**Tasks**:
- [ ] Implement `ExperimentDesigner`
  - [ ] Create system prompt for experiment design
  - [ ] Parse JSON experiment specification from Gemini
  - [ ] Handle invalid/unparseable responses
  - [ ] Implement metric selection logic
- [ ] Implement `ResultsAnalyzer`
  - [ ] Create prompts for analyzing outcomes
  - [ ] Compare current results to history
  - [ ] Detect patterns (improving, degrading, plateauing)
- [ ] Implement constraints file parsing
  - [ ] Read Markdown file
  - [ ] Pass to Gemini as context
- [ ] Test cognitive components in isolation

**Deliverables**:
- Experiment Designer generates valid experiment specs
- Results Analyzer provides meaningful analysis
- Constraints are incorporated into design decisions

### Day 11-12: Hypothesis Generation & Thought Signatures

**Tasks**:
- [ ] Implement `HypothesisGenerator`
  - [ ] Analyze why experiments succeeded or failed
  - [ ] Generate testable hypotheses for next iteration
  - [ ] Maintain chain of reasoning across iterations
- [ ] Implement multi-turn conversation management
  - [ ] Track conversation history in state
  - [ ] Verify Thought Signatures maintain context
- [ ] Implement reasoning display for demo
  - [ ] Rich console output showing Gemini's thinking
  - [ ] Thought Signature status indicator

**Deliverables**:
- Hypothesis generator produces actionable insights
- Thought Signatures demonstrably maintain reasoning continuity
- Reasoning is visually displayed for demo

### Day 13-14: MLflow Integration & State Management

**Tasks**:
- [ ] Enhance MLflow integration
  - [ ] Log Gemini reasoning as artifacts
  - [ ] Store experiment specifications
  - [ ] Log hypotheses as tags
- [ ] Implement robust state management
  - [ ] Auto-save state after each iteration
  - [ ] Implement checkpoint/restore functionality
  - [ ] Handle graceful shutdown and resume
- [ ] Implement termination criteria
  - [ ] Max iterations check
  - [ ] Plateau detection
  - [ ] Time budget check
  - [ ] Agent decision integration
- [ ] Connect all components in main loop

**Deliverables**:
- Full experiment history visible in MLflow
- Can pause and resume experiment sessions
- Termination criteria working

### Phase 2 Milestone
✅ **Demo**: Agent runs multiple iterations, Gemini explains decisions, adapts based on results

---

## 4. Phase 3: Robustness & Polish (Days 15-21)

### Goal
Handle real-world complexity and produce professional output.

### Day 15-16: Error Handling & Edge Cases

**Tasks**:
- [ ] Implement comprehensive error handling
  - [ ] Gemini API errors (rate limits, timeouts, invalid responses)
  - [ ] Experiment execution errors (crashes, timeouts)
  - [ ] Data errors (missing values, encoding issues)
- [ ] Add retry logic with exponential backoff
- [ ] Implement graceful degradation (skip failed experiments)
- [ ] Add code validation before execution
- [ ] Test with intentionally problematic inputs

**Deliverables**:
- System handles errors gracefully without crashing
- Clear error messages and recovery paths

### Day 17-18: Multi-Model Support

**Tasks**:
- [ ] Add XGBoost support
  - [ ] Create `xgboost_model.py.jinja` template
  - [ ] Add hyperparameter space definition
  - [ ] Test template generation
- [ ] Add LightGBM support
  - [ ] Create `lightgbm_model.py.jinja` template
  - [ ] Add hyperparameter space definition
  - [ ] Test template generation
- [ ] Update Experiment Designer prompts for new models
- [ ] Test model selection reasoning

**Deliverables**:
- Agent can choose between sklearn, XGBoost, and LightGBM
- Appropriate model selection based on data characteristics

### Day 19-21: Report Generation & Visualizations

**Tasks**:
- [ ] Implement `ReportGenerator`
  - [ ] Executive summary section
  - [ ] Methodology narrative
  - [ ] Results tables and comparisons
  - [ ] Key insights and recommendations
  - [ ] Appendix with experiment details
- [ ] Create visualization generation
  - [ ] Learning curves across iterations
  - [ ] Feature importance plots
  - [ ] Metric progression charts
- [ ] Test report quality with demo datasets

**Report Structure**:
```markdown
# ML Experiment Report: {dataset_name}

## Executive Summary
{one_paragraph_summary}

## Dataset Overview
{data_profile_summary}

## Methodology
{narrative_of_approach}

## Experiment Results

### Performance Summary
| Experiment | Model | {metric} | Notes |
|------------|-------|----------|-------|

### Best Model
{best_model_details}

## Key Insights
{gemini_generated_insights}

## Recommendations
{next_steps}

## Appendix
{detailed_logs}
```

**Deliverables**:
- Professional Markdown reports
- Informative visualizations
- Reports suitable for portfolio

### Phase 3 Milestone
✅ **Demo**: Complete end-to-end run on House Prices with polished report

---

## 5. Phase 4: Demo Preparation (Days 22-25)

### Goal
Create compelling hackathon submission.

### Day 22-23: Demo Preparation

**Tasks**:
- [ ] Perform full demo runs on House Prices dataset
- [ ] Perform full demo run on Titanic dataset (backup)
- [ ] Capture outputs and identify issues
- [ ] Fix any demo-breaking bugs
- [ ] Prepare talking points for video
- [ ] Create demo script with timestamps

**Demo Script** (3 minutes):
```
[0:00-0:20] Introduction
- "This is ML Experiment Autopilot, an autonomous agent that..."
- Problem: manual ML iteration is tedious and inconsistent

[0:20-1:00] Setup & Launch
- Show dataset (House Prices)
- Run command with --verbose flag
- Highlight: natural language constraints file

[1:00-2:00] Agent in Action
- Show data profiling output
- Show Gemini's experiment design reasoning
- Show iteration loop with Thought Signature indicator
- Highlight: agent references previous iterations

[2:00-2:40] Results
- Show MLflow dashboard with all experiments
- Show generated report
- Show best model performance improvement

[2:40-3:00] Conclusion
- Recap: explains decisions, generates hypotheses, produces reports
- Different from AutoML: not a black box
- Future potential
```

**Deliverables**:
- Smooth demo runs on selected datasets
- Video script finalized

### Day 24: Documentation & Video

**Tasks**:
- [ ] Record 3-minute demo video
  - [ ] Screen recording with voiceover
  - [ ] Edit for clarity and pacing
- [ ] Polish README.md
  - [ ] Clear project description
  - [ ] Installation instructions
  - [ ] Usage examples
  - [ ] Architecture overview
- [ ] Write Devpost description (~200 words on Gemini integration)
- [ ] Prepare repository for public viewing
  - [ ] Remove sensitive data/keys
  - [ ] Add MIT license
  - [ ] Clean up code and comments

**Deliverables**:
- Completed demo video
- Polished documentation
- Clean public repository

### Day 25: Final Submission (February 9)

**Tasks**:
- [ ] Final code review
- [ ] Upload video to hosting platform
- [ ] Submit to Devpost:
  - [ ] Project description
  - [ ] Demo video link
  - [ ] GitHub repository link
  - [ ] Any additional links
- [ ] Verify submission is complete
- [ ] Celebrate! 🎉

**Deliverables**:
- ✅ Submitted hackathon entry

---

## 6. Demo Data Strategy

### Primary: House Prices (Kaggle)

**URL**: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

**Why**:
- Universally understood problem
- Rich features requiring preprocessing
- Clear improvement trajectory
- Fast iterations (1460 rows)

**Expected Demo Flow**:
1. Agent profiles data, identifies 81 features, notes missing values
2. Baseline: LinearRegression, RMSE ~0.18 (log scale)
3. Iteration 1: RandomForest, RMSE ~0.15
4. Iteration 2: GradientBoosting, RMSE ~0.14
5. Iteration 3: XGBoost (if Phase 2 complete), RMSE ~0.13
6. Final report generated

### Secondary: Titanic (Kaggle)

**URL**: https://www.kaggle.com/c/titanic

**Why**:
- Classification task (demonstrates both task types)
- Very fast iterations (891 rows)
- Good backup if primary demo has issues

---

## 7. Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Gemini generates invalid code | Medium | High | Code validation layer, retry with feedback |
| API rate limits during demo | Low | Critical | Pre-run demo, have recorded backup |
| Infinite experiment loops | Low | Medium | Hard iteration limits, plateau detection |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 2 takes longer | Medium | High | Phase 3 features can be cut |
| Report generation complex | Medium | Medium | Markdown-only is acceptable |
| Demo recording issues | Low | High | Record on Day 22, leave buffer |

### Contingency Plan

**If behind schedule by Day 15**:
- Skip multi-model support (keep sklearn only)
- Simplify report to basic Markdown tables
- Focus on core loop stability

**If behind schedule by Day 20**:
- Use simpler demo dataset (Titanic)
- Pre-record demo with scripted inputs
- Document limitations honestly

---

## 8. Success Metrics

### Minimum Viable Demo
- [ ] Agent completes 5+ iterations autonomously
- [ ] Each iteration shows clear Gemini reasoning
- [ ] Results improve from baseline
- [ ] Basic report generated
- [ ] Video is under 3 minutes

### Target Demo
- [ ] Agent completes 10+ iterations
- [ ] Handles at least one error gracefully
- [ ] Supports multiple model families
- [ ] Professional report with visualizations
- [ ] Compelling video narrative

### Stretch Goals
- [ ] Docker execution option
- [ ] Web UI for monitoring
- [ ] Real-time streaming of reasoning

---

## 9. Testing Checklist (Pre-Demo)

### Tier 1: Critical Path (Must Pass)
- [x] `test_data_profiler_house_prices` ✅ (11 tests passing)
- [x] `test_code_generation_sklearn` ✅ (10 tests passing)
- [x] `test_experiment_execution` ✅ (7 tests passing)
- [x] `test_gemini_experiment_design` ✅ (11 tests passing - mocked)
- [ ] `test_full_loop_3_iterations`
- [ ] `test_mlflow_logging`

### Tier 2: Robustness
- [ ] `test_gemini_retry_on_rate_limit`
- [ ] `test_invalid_gemini_response_handling`
- [ ] `test_experiment_timeout`
- [ ] `test_state_save_and_resume`

### Pre-Demo Script
```bash
./scripts/pre_demo_check.sh
```

---

## 10. Daily Progress Template

```markdown
## Day N Standup - [Date]

### Completed Yesterday
- 

### Plan for Today
- 

### Blockers
- 

### Notes
- 
```

---

*Implementation Plan v2.0*
*Updated: January 2026*
*All decisions finalized*
